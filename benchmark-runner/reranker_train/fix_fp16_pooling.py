#!/usr/bin/env python3
"""
Stop the mean-pooling head from overflowing float16 on long documents.

THE BUG
-------
ModernBERT's sequence-classification head mean-pools over the sequence:

    mean_pool(final_norm(h)) -> head.dense -> act -> head.norm -> classifier

The pooling sums across positions. In float16 that sum saturates at 65504, and
the next LayerNorm turns the resulting inf into NaN. TEI serves float16 by
default and offers only float16 or float32, so every long document comes back
as `{"error":"score is NaN"}` and Hindsight records a failed rerank.

Measured on our checkpoints before this fix, in TEI's default float16:

    v1 (22 layers)  ok to 1024 tokens, NaN at 2048
    v2 (6 layers)   ok to  768 tokens, NaN at  896

The same weights are correct at every length in bfloat16 and float32, which is
why training never saw it: we train in bf16, which carries float32's exponent
range. The untrained `jhu-clsp/mmBERT-small` backbone is also fine, and so is
`Alibaba-NLP/gte-reranker-modernbert-base`. Our fine-tuning grows the final
hidden states enough to cross the line, and the shallower student crosses it
sooner.

Instrumentation that pinned it down: at 1024 tokens `final_norm` output is
finite with max magnitude 92.06, and `head.dense` input is already inf. Nothing
happens in between except the pooling. Note that no layer output and no
attention score is anywhere near the fp16 ceiling, so a survey of layer
activations or QK^T magnitudes finds nothing. Only the reduction overflows.

THE FIX
-------
`final_norm` is bias-free (`norm_bias: false`), so its output is exactly linear
in its weight. For any k:

    mean(final_norm_w/k @ h) @ (k * dense_w) + dense_b
      == mean(final_norm_w @ h) @ dense_w + dense_b

Dividing the norm weight by k and multiplying the head's dense weight by k is
the identical function with every summed value k times smaller.
`head.dense.bias` is deliberately NOT scaled: it is added after the matmul and
does not scale with the input.

This costs nothing at inference. Same architecture, same dtype, same speed.

k=16 gives ample headroom: it puts a worst-case 8192-token sum around 47,000
against the 65504 ceiling, while keeping values far above fp16's ~6e-5 normal
minimum so nothing underflows at the other end.

  python fix_fp16_pooling.py --src out/final --dst out/final_fp16fix --k 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

# Metadata files sentence-transformers and TEI need that `save_pretrained` on a
# bare HF model does not write.
SIDECAR = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "sentence_bert_config.json", "modules.json", "config_sentence_transformers.json")

PROBE_LENGTHS = (256, 512, 768, 1024, 2048, 4096)


def make_tei_servable(path: Path, reference: Path, max_seq_length: int = 8192) -> None:
    """Re-apply the three TEI metadata patches after `save_pretrained`.

    Easy to forget, and the failure is confusing: transformers 5.x writes RoPE
    settings as a nested `rope_parameters` block, TEI reads flat
    `global_rope_theta` / `local_rope_theta`, and the container exits during
    startup rather than reporting a bad checkpoint. Values are copied from the
    unrescaled checkpoint, which was already known to serve.
    """
    cfg = json.loads((path / "config.json").read_text())
    ref = json.loads((reference / "config.json").read_text())
    rp = cfg.get("rope_parameters") or {}
    cfg.setdefault("global_rope_theta", ref.get("global_rope_theta")
                   or (rp.get("full_attention") or {}).get("rope_theta", 160000))
    cfg.setdefault("local_rope_theta", ref.get("local_rope_theta")
                   or (rp.get("sliding_attention") or {}).get("rope_theta", cfg["global_rope_theta"]))
    cfg.setdefault("local_attention", ref.get("local_attention", 128))
    if "pruned_from_layers" in ref:
        cfg.setdefault("pruned_from_layers", ref["pruned_from_layers"])
    cfg["fp16_pooling_rescale"] = True  # provenance: this checkpoint has been rescaled
    (path / "config.json").write_text(json.dumps(cfg, indent=2))

    sb = path / "sentence_bert_config.json"
    d = json.loads(sb.read_text()) if sb.exists() else {}
    d["max_seq_length"] = max_seq_length
    d.setdefault("do_lower_case", False)
    sb.write_text(json.dumps(d, indent=2))

    (path / "modules.json").write_text(json.dumps(
        [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}], indent=2))


def build_doc(tokenizer, n_tokens: int) -> str:
    words: list[str] = []
    i = 0
    while len(tokenizer.encode(" ".join(words)).ids) < n_tokens:
        words.append(f"item{i}")
        i += 1
    return " ".join(words)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--k", type=float, default=16.0)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="Max allowed |rescaled fp16 - fp32 reference| before this fails")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src, dst = Path(args.src), Path(args.dst)
    tok = AutoTokenizer.from_pretrained(str(src))
    bt = tok.backend_tokenizer
    # A tokenizer that pads to a fixed width would make every probe the same
    # length and the whole verification meaningless.
    bt.no_padding()
    bt.no_truncation()
    query = "what did we decide about the migration"
    docs = {n: build_doc(bt, n) for n in PROBE_LENGTHS}

    def score(model, n: int) -> float:
        enc = tok(query, docs[n], return_tensors="pt", truncation=True, max_length=8192).to("cuda")
        with torch.no_grad():
            return model(**enc).logits.view(-1).float().item()

    # float32 is the ground truth: it is the dtype with enough range that the
    # pooling cannot overflow, so it says what the weights actually compute.
    ref = AutoModelForSequenceClassification.from_pretrained(
        str(src), dtype=torch.float32, num_labels=1, attn_implementation="eager").cuda().eval()
    gold = {n: score(ref, n) for n in PROBE_LENGTHS}
    del ref
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        str(src), dtype=torch.float32, num_labels=1, attn_implementation="eager")
    if model.model.final_norm.bias is not None:
        raise SystemExit(
            "final_norm has a bias, so scaling its weight is not a function-preserving "
            "identity. Scale the bias by the same factor, or do not use this script."
        )
    with torch.no_grad():
        model.model.final_norm.weight.div_(args.k)
        model.head.dense.weight.mul_(args.k)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))
    for name in SIDECAR:
        p = src / name
        if p.exists():
            shutil.copy(p, dst / name)
    make_tei_servable(dst, src)
    del model
    torch.cuda.empty_cache()

    fixed = AutoModelForSequenceClassification.from_pretrained(
        str(dst), dtype=torch.float16, num_labels=1, attn_implementation="eager").cuda().eval()
    print(f"rescale k={args.k:g}\n{'tokens':>7} {'fp32 ref':>11} {'fixed fp16':>11} {'abs diff':>10}")
    worst = 0.0
    failures = []
    for n in PROBE_LENGTHS:
        got = score(fixed, n)
        if got != got:
            print(f"{n:7d} {gold[n]:11.4f} {'NaN':>11} {'-':>10}")
            failures.append(f"{n} tokens still NaN")
            continue
        d = abs(got - gold[n])
        worst = max(worst, d)
        print(f"{n:7d} {gold[n]:11.4f} {got:11.4f} {d:10.5f}")
    if worst > args.tolerance:
        failures.append(f"worst deviation {worst:.5f} exceeds tolerance {args.tolerance}")

    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print(f"\nPASS: worst deviation {worst:.5f}, no NaN, wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
