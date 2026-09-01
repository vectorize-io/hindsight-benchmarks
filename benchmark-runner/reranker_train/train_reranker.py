#!/usr/bin/env python3
"""
Distil a reranker onto mmBERT-small in two stages.

Stage 1 teaches general relevance from five public IR sources — T2Ranking and
DuReader (Chinese), MIRACL (Chinese and English), HotpotQA and MS MARCO
(English), roughly 47% Chinese overall. A masked-LM backbone starts with no
ranking ability at all, so this is the bulk of the learning. Stage 2 adapts to
Hindsight's distribution using synthetic English and Chinese
conversational-memory documents at the real 3000-char chunk size. Both use MarginMSE against a teacher's score margin, which the literature
finds beats pointwise MSE by margins comparable to scaling the backbone.

Sequence length differs by stage on purpose: MS MARCO passages are short, so
stage 1 trains at 512 and runs fast, while stage 2 trains long because that is
where our real chunks live — Chinese 3000-char chunks reach ~2000+ tokens. The
default here is 2560; the v1 run used 2048 after a 40GB A100 ran out of memory.
mmBERT was pretrained with a context-extension phase to 8192, so serving longer
than we trained is supported for the encoder — but the ranking head has never
seen an input longer than the stage-2 window, so treat 8192 as untested.

The output is a standard ModernBertForSequenceClassification checkpoint, which
TEI serves natively — so the result drops into the exact benchmark harness used
for every other model and the numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import MarginMSELoss

STUDENT_DEFAULT = "jhu-clsp/mmBERT-small"


def make_tei_servable(path: Path, max_seq_length: int = 8192) -> None:
    """Make a sentence-transformers checkpoint loadable by TEI 1.8.3.

    Three incompatibilities, all cosmetic to the weights but each fatal or
    noisy at serve time:

    1. sentence_bert_config.json ships without `max_seq_length`; TEI refuses to
       boot ("missing field `max_seq_length`").
    2. transformers 5.x rewrites ModernBERT's RoPE settings into a nested
       `rope_parameters` block, while TEI still reads the flat
       `global_rope_theta` / `local_rope_theta`. Without them the Candle
       backend reports "Model is not supported: missing field
       `global_rope_theta`" even though the weights are fine.
    3. sentence-transformers 6.x writes module paths like
       `sentence_transformers.base.modules.transformer.Transformer`, which TEI
       cannot parse. Non-fatal, but it logs an error on every boot.

    The weights are already a valid ModernBertForSequenceClassification, so
    this is purely a metadata translation.
    """
    cfg = path / "sentence_bert_config.json"
    data = json.loads(cfg.read_text()) if cfg.exists() else {}
    data["max_seq_length"] = max_seq_length
    data.setdefault("do_lower_case", False)
    cfg.write_text(json.dumps(data, indent=2))

    mc = path / "config.json"
    if mc.exists():
        c = json.loads(mc.read_text())
        rp = c.get("rope_parameters") or {}
        if "global_rope_theta" not in c:
            c["global_rope_theta"] = (rp.get("full_attention") or {}).get("rope_theta", 160000)
        if "local_rope_theta" not in c:
            c["local_rope_theta"] = (rp.get("sliding_attention") or {}).get("rope_theta", c["global_rope_theta"])
        c.setdefault("local_attention", 128)
        mc.write_text(json.dumps(c, indent=2))
        print(
            f"  config.json: global_rope_theta={c['global_rope_theta']} "
            f"local_rope_theta={c['local_rope_theta']}",
            flush=True,
        )

    mods = path / "modules.json"
    if mods.exists():
        mods.write_text(json.dumps([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}
        ], indent=2))
    print(f"  patched {path.name} for TEI (max_seq_length={max_seq_length})", flush=True)



def select_layers(n_src: int, n_keep: int, global_every: int) -> list[int]:
    """Choose which source layers survive, preserving each layer's attention role.

    ModernBERT alternates attention types: layer i is global when
    ``i % global_attn_every_n_layers == 0`` and sliding-window (128 tokens)
    otherwise. That role is baked onto the attention module at construction, so
    it travels with the weights when we reindex the ModuleList.

    This constraint is load-bearing for SERVING, which is the part that would
    fail silently. ``prune_to_layers`` rewrites ``config.layer_types``, so
    transformers reloads any selection as what it trained as. TEI 1.8.3 does not
    read ``layer_types`` at all: its ModernBERT backend re-derives each role from
    the layer's position as ``index % global_attn_every_n_layers``. A selection
    whose roles disagree with that arithmetic trains one model and serves a
    different one, with no error anywhere, and the only thing standing between
    that and a published benchmark number is ``tei_parity_check``.

    It is also the better choice on the merits. A layer pretrained to attend over
    a 128-token window learned features suited to that receptive field, and
    reassigning it to full attention puts it to work in a regime it never saw.

    Even spacing with a stride coprime to ``global_every`` avoids this entirely.
    When gcd(stride, global_every) == 1 the stride is invertible modulo
    global_every, so ``(i * stride) % global_every == 0`` exactly when
    ``i % global_every == 0``: every kept layer lands on a new index carrying the
    same role it had. A stride that shares a factor breaks this. Stride 3 with
    global_every 3, for instance, selects only global layers and then assigns
    four of the six a local role.

    We search strides downward from the widest that fits and take the first that
    is role-preserving, falling back to plain even spacing with a loud warning.
    """
    if n_keep < 1 or n_keep > n_src:
        raise ValueError(f"cannot keep {n_keep} of {n_src} layers")
    if n_keep == n_src:
        return list(range(n_src))

    def roles_agree(sel: list[int]) -> bool:
        return all((old % global_every == 0) == (new % global_every == 0)
                   for new, old in enumerate(sel))

    from math import gcd

    widest = (n_src - 1) // (n_keep - 1) if n_keep > 1 else n_src
    for stride in range(widest, 0, -1):
        if gcd(stride, global_every) != 1:
            continue
        sel = [i * stride for i in range(n_keep)]
        if sel[-1] < n_src and roles_agree(sel):
            # Coprimality can force a narrow stride that clusters every kept
            # layer at the bottom of the stack and discards the top entirely.
            # Role preservation is still worth more than span, but a selection
            # that never reaches the final third should not pass silently.
            if sel[-1] <= (n_src - 1) * 2 // 3:
                print(
                    f"  WARNING: layer selection {sel} spans only the first "
                    f"{sel[-1] + 1} of {n_src} layers; the top of the stack is "
                    f"discarded. Consider a different --keep-layers.",
                    flush=True,
                )
            return sel

    if n_keep == 1:
        return [0]
    sel = [round(i * (n_src - 1) / (n_keep - 1)) for i in range(n_keep)]
    print(
        f"  WARNING: no role-preserving stride for {n_keep} of {n_src} layers "
        f"(global every {global_every}); falling back to even spacing {sel}. "
        f"Attention roles will shift when the pruned config is reloaded.",
        flush=True,
    )
    return sel


def prune_to_layers(hf_model, keep: list[int]) -> None:
    """Drop all but `keep` encoder layers, in place, and make the config agree.

    Weight surgery only: each retained block keeps its own attention, MLP and
    norms exactly as pretrained. Nothing is reinitialised or reshaped.

    Three pieces of state decide a ModernBERT layer's behaviour in transformers
    5.x, and all three have to move with the weights:

    * ``config.layer_types`` is the source of truth. It is a list, one entry per
      layer, of "full_attention" or "sliding_attention", and a rebuilt layer
      reads ``config.layer_types[layer_idx]``. Leaving 22 entries on a 6-layer
      config means the reloaded model assigns roles by position in the OLD list,
      so the checkpoint we serve is not the model we trained.
    * ``layer.attention_type`` and ``attn.sliding_window`` are resolved at
      construction and cached on the module, so they travel with the weights.
      They must end up agreeing with the rewritten ``layer_types``.
    * ``layer_idx`` is stamped at construction and left stale by a plain reindex.

    Source layer 0 is special: transformers gives it ``nn.Identity()`` for
    ``attn_norm`` where every other layer gets a real LayerNorm. A pruned stack
    that does not start at source layer 0 would train with a LayerNorm in
    position 0 and reload with an Identity, silently dropping a normalisation
    and its weights.
    """
    import torch.nn as nn

    enc = hf_model.model  # ModernBertForSequenceClassification -> ModernBertModel
    if not hasattr(enc, "layers"):
        raise RuntimeError(f"expected .model.layers on {type(hf_model).__name__}")
    src = enc.layers
    cfg = hf_model.config
    if max(keep) >= len(src):
        raise ValueError(f"layer {max(keep)} requested but model has {len(src)}")
    if keep[0] != 0:
        raise ValueError(
            f"layer selection must start at source layer 0 (got {keep[0]}): only "
            "layer 0 carries nn.Identity() as attn_norm, so any other first layer "
            "trains and reloads as different models"
        )

    old_types = list(getattr(cfg, "layer_types", []))
    if len(old_types) != len(src):
        raise RuntimeError(
            f"config.layer_types has {len(old_types)} entries for {len(src)} layers; "
            "this transformers version does not lay ModernBERT out as expected"
        )
    new_types = [old_types[i] for i in keep]

    enc.layers = nn.ModuleList([src[i] for i in keep])
    cfg.num_hidden_layers = len(keep)
    cfg.layer_types = new_types
    cfg.pruned_from_layers = list(keep)  # provenance for the saved checkpoint

    for new_idx, layer in enumerate(enc.layers):
        layer.layer_idx = new_idx
        if hasattr(layer, "attn"):
            layer.attn.layer_idx = new_idx

    # The role a layer carries must equal the role its new position implies, or
    # a reload rebuilds it differently from how it was trained.
    for new_idx, layer in enumerate(enc.layers):
        carried = getattr(layer, "attention_type", None)
        if carried is not None and carried != cfg.layer_types[new_idx]:
            raise RuntimeError(
                f"layer {new_idx} (from source {keep[new_idx]}) carries "
                f"{carried!r} but config says {cfg.layer_types[new_idx]!r}"
            )
        want_sliding = cfg.layer_types[new_idx] == "sliding_attention"
        has_sliding = getattr(getattr(layer, "attn", None), "sliding_window", None) is not None
        if has_sliding != want_sliding:
            raise RuntimeError(
                f"layer {new_idx} (from source {keep[new_idx]}) sliding_window="
                f"{getattr(layer.attn, 'sliding_window', None)} contradicts "
                f"{cfg.layer_types[new_idx]!r}"
            )

    every = getattr(cfg, "global_attn_every_n_layers", None)
    if every:
        mismatched = [
            (new, old, new_types[new])
            for new, old in enumerate(keep)
            if (new_types[new] == "full_attention") != (new % every == 0)
        ]
        if mismatched:
            raise RuntimeError(
                "layer roles disagree with TEI's positional rule "
                f"(index % {every} == 0 means full attention): {mismatched}. "
                "transformers would reload this correctly from layer_types, but TEI "
                "1.8.3 ignores that field and would serve a different model than "
                "was trained, with no error."
            )

    desc = ", ".join(
        f"{old}->{new}{'G' if new_types[new] == 'full_attention' else 'L'}"
        for new, old in enumerate(keep)
    )
    print(f"  pruned {len(src)} -> {len(keep)} layers: {desc}", flush=True)


def load_triples(path: Path, limit: int | None = None) -> Dataset:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                {
                    "query": r["query"],
                    "positive": r["positive"],
                    "negative": r["negative"],
                    "label": float(r["label"]),
                }
            )
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return Dataset.from_list(rows)


def language_mix(path: Path) -> dict[str, int]:
    mix: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            lang = json.loads(line).get("language", "?")
            mix[lang] = mix.get(lang, 0) + 1
    return mix


def run_stage(
    model: CrossEncoder,
    dataset: Dataset,
    out_dir: Path,
    stage_name: str,
    epochs: float,
    batch_size: int,
    lr: float,
    max_steps: int,
    warmup_ratio: float = 0.1,
    eval_rows: int = 3000,
    eval_every: int = 2500,
    # Save on the same cadence as eval, so every eval point is a checkpoint the
    # best-model logic can actually keep. Saving every 5000 while evaluating
    # every 2500 leaves half the minima with nothing on disk: one ablation arm's
    # best eval landed at step 27,500 with checkpoints only at 25,000 and 30,000.
    # save_total_limit still bounds disk, and with load_best_model_at_end the
    # trainer retains the best alongside the most recent.
    save_every: int = 2500,
) -> CrossEncoder:
    # Hold out a FIXED SMALL slice, not a fraction. At 2M rows a 1% holdout is
    # 20k, and evaluating that every 200 steps costs more wall clock than the
    # training it is monitoring (~155 evals x 313 batches x 2 forwards).
    eval_ds = None
    if eval_rows > 0 and len(dataset) > eval_rows * 3:
        # Split by QUERY, not by row. Up to 6 rows share a query, so a row-level
        # split leaks the same query into both halves and the eval curve then
        # measures memorisation rather than generalisation.
        import random as _random

        queries = sorted(set(dataset["query"]))
        rgen = _random.Random(1234)
        rgen.shuffle(queries)
        n_eval_q = max(1, int(len(queries) * min(0.2, eval_rows / len(dataset))))
        held = set(queries[:n_eval_q])
        idx_eval = [i for i, q in enumerate(dataset["query"]) if q in held]
        idx_train = [i for i, q in enumerate(dataset["query"]) if q not in held]
        eval_ds = dataset.select(idx_eval)
        dataset = dataset.select(idx_train)
        print(
            f"  held out {len(eval_ds)} rows over {len(held)} unseen queries "
            f"(every {eval_every} steps)",
            flush=True,
        )
    loss = MarginMSELoss(model)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / f"ckpt_{stage_name}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        bf16=torch.cuda.is_available(),
        logging_steps=25,
        # Checkpoint periodically: a crash hours in must not lose the run.
        # Must match eval_strategy for load_best_model_at_end to work.
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=2,
        # The eval curve oscillates by ~0.1 on a 3k-row holdout, so the last
        # checkpoint is wherever the noise happened to land rather than the best
        # model: one ablation arm's final four evals spanned 5.74 to 5.93.
        # Without this the trainer ships that coin flip, and save_total_limit
        # quietly deletes the good checkpoint (that arm's 5.740 at step 27,500
        # was gone while 25,000 and 30,000 survived).
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        max_steps=max_steps if max_steps > 0 else -1,
        dataloader_num_workers=4,
        seed=1234,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_every if eval_ds is not None else None,
        per_device_eval_batch_size=batch_size * 2,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=args, train_dataset=dataset, eval_dataset=eval_ds, loss=loss
    )
    print(f"\n=== {stage_name}: {len(dataset)} triples, bs={batch_size}, lr={lr} ===", flush=True)
    trainer.train()
    # Persist the curve so it can be charted without re-parsing stdout.
    hist_path = out_dir / f"loghistory_{stage_name}.json"
    with hist_path.open("w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    print(f"  log history -> {hist_path}", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default=STUDENT_DEFAULT)
    ap.add_argument("--keep-layers", type=int, default=0,
                    help="Prune the student to this many encoder layers (0 = no pruning)")
    ap.add_argument("--data-dir", default=str(Path.home() / "train" / "data"))
    ap.add_argument("--out-dir", default=str(Path.home() / "train" / "out"))
    ap.add_argument("--stage1-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--stage1-batch", type=int, default=64)
    ap.add_argument("--stage2-batch", type=int, default=32)
    ap.add_argument("--stage1-lr", type=float, default=2e-5)
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--stage1-epochs", type=float, default=1.0)
    ap.add_argument("--stage2-epochs", type=float, default=2.0)
    ap.add_argument("--stage1-max-steps", type=int, default=0)
    ap.add_argument("--stage2-max-steps", type=int, default=0)
    # v2 measured its warmup cost directly: the 6-layer student sat near its
    # starting loss until epoch 0.071, escaping only once the schedule had ramped
    # to lr 1.4e-5, while the 22-layer v1 was already learning at 3.9e-6. A
    # smaller student needs more learning rate to move, so 10% of the run spent
    # ramping is ~1,000 wasted steps for it and roughly none for v1.
    ap.add_argument("--warmup-ratio", type=float, default=0.1,
                    help="Fraction of each stage spent ramping the learning rate")
    ap.add_argument("--stage1-maxlen", type=int, default=512)
    ap.add_argument("--stage2-maxlen", type=int, default=2560)
    ap.add_argument("--skip-stage1", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resumed = out_dir / "after_stage1"
    if args.skip_stage1 and args.student == STUDENT_DEFAULT and resumed.exists():
        args.student = str(resumed)
        print(f"--skip-stage1: continuing from {resumed} (not the raw masked LM)", flush=True)
    elif args.skip_stage1 and args.student == STUDENT_DEFAULT:
        raise SystemExit(
            "--skip-stage1 with no after_stage1 checkpoint would fine-tune the RAW masked LM "
            "on stage 2 only. Pass --student explicitly if that is really what you want."
        )

    print(f"Student: {args.student}", flush=True)
    # attn_implementation matters enormously at long sequence lengths. We
    # skipped installing flash-attn (a 30+ minute compile), so ModernBERT falls
    # back to eager attention, whose memory is O(seq^2): at 2560 tokens and
    # batch 16 that is ~27GB of attention matrices and a 40GB A100 OOMs at the
    # first step. SDPA's memory-efficient kernel is O(seq) and fits.
    model = CrossEncoder(
        args.student,
        num_labels=1,
        max_length=args.stage1_maxlen,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    print(f"  attn_implementation: {getattr(model.model.config, 'attn_implementation', '?')}", flush=True)

    def report_size(tag: str) -> tuple[int, int]:
        total = sum(p.numel() for p in model.model.parameters())
        non_embed = total - model.model.get_input_embeddings().weight.numel()
        print(f"  {tag}: {total/1e6:.1f}M total, {non_embed/1e6:.1f}M non-embedding", flush=True)
        return total, non_embed

    report_size("params")

    if args.keep_layers:
        cfg = model.model.config
        n_src = cfg.num_hidden_layers
        if args.keep_layers >= n_src:
            raise SystemExit(
                f"--keep-layers {args.keep_layers} does not prune a {n_src}-layer model"
            )
        # Pruning a checkpoint that already finished stage 1 would throw away
        # most of what stage 1 taught. Prune the raw backbone or not at all.
        if args.skip_stage1:
            raise SystemExit(
                "--keep-layers with --skip-stage1 would prune a model that already "
                "trained at full depth. Prune before stage 1."
            )
        keep = select_layers(n_src, args.keep_layers, cfg.global_attn_every_n_layers)
        prune_to_layers(model.model, keep)
        report_size("params after prune")
        # The classifier head sits on top of the encoder and is untouched by
        # pruning, but a stale cached reference would still point at the old
        # stack. Re-read the depth from the live module as a cross-check.
        live_depth = len(model.model.model.layers)
        if live_depth != args.keep_layers:
            raise SystemExit(f"prune left {live_depth} layers, expected {args.keep_layers}")

    if not args.skip_stage1:
        p1 = data_dir / "stage1.jsonl"
        ds1 = load_triples(p1, args.stage1_limit or None)
        print(f"stage1 language mix: {language_mix(p1)}", flush=True)
        model.max_length = args.stage1_maxlen
        model = run_stage(
            model, ds1, out_dir, "stage1", args.stage1_epochs, args.stage1_batch,
            args.stage1_lr, args.stage1_max_steps, warmup_ratio=args.warmup_ratio,
        )
        model.save_pretrained(str(out_dir / "after_stage1"))
        make_tei_servable(out_dir / "after_stage1")
        print(f"saved -> {out_dir / 'after_stage1'}", flush=True)

    p2 = data_dir / "stage2.jsonl"
    if p2.exists():
        ds2 = load_triples(p2)
        print(f"stage2 language mix: {language_mix(p2)}", flush=True)
        # Longer window here: this is the regime the model will actually serve.
        model.max_length = args.stage2_maxlen
        model = run_stage(
            model, ds2, out_dir, "stage2", args.stage2_epochs, args.stage2_batch,
            args.stage2_lr, args.stage2_max_steps, warmup_ratio=args.warmup_ratio,
        )
    else:
        print(f"WARNING: {p2} missing, skipping domain adaptation", flush=True)

    final = out_dir / "final"
    model.save_pretrained(str(final))
    make_tei_servable(final)
    print(f"\nFinal checkpoint -> {final}", flush=True)

    # Sanity check: a relevant document must outscore an irrelevant one. If this
    # fails the training collapsed and there is no point serving it.
    probe = [
        ("what did we decide about the database migration",
         "We agreed to postpone the database migration until after the release."),
        ("what did we decide about the database migration",
         "The weather was nice so we walked to the park and fed the ducks."),
    ]
    scores = model.predict(probe)
    print(f"probe scores: relevant={scores[0]:.4f} irrelevant={scores[1]:.4f}", flush=True)
    if scores[0] <= scores[1]:
        print("WARNING: relevant did not outscore irrelevant — training likely failed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
