# Frozen reranker data

These files retain the source datasets' licenses. They are not relicensed as repository code. No images are redistributed.

## LoCoMo

`locomo10-top100.json.gz` is an adaptation of the public LoCoMo release by Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang, described in [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753), ACL 2024.

Source: [snap-research/locomo](https://github.com/snap-research/locomo). License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/), including its disclaimer of warranties; the original notice is preserved in [licenses/locomo-CC-BY-NC-4.0.txt](licenses/locomo-CC-BY-NC-4.0.txt).

Changes: flatten dialogue turns into passages with speaker, timestamp, and provided image descriptions; retain original query/evidence annotations; normalize evidence-ID syntax; record exclusions; freeze BM25/BGE top-100 retrieval candidates. Preparation parameters and source/content checksums are embedded in the fixture.

## SciFact

`scifact-test-top100.json.gz` adapts the [BEIR SciFact distribution](https://huggingface.co/datasets/BeIR/scifact), whose dataset card declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This fixture adaptation retains that license. Changes: combine titles and abstracts, select the BEIR test qrels, and freeze BM25/BGE top-100 retrieval candidates.

The original SciFact data is by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi, described in [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974), EMNLP 2020. [Original source](https://github.com/allenai/scifact) and [license notice](licenses/scifact-notice.md): claims and evidence annotations are CC BY 4.0; abstracts come from S2ORC under ODC-By 1.0. Those notices are retained alongside the BEIR distribution notice.

`scifact-evidence-labels.json` is derived directly from the original `claims_dev.jsonl` evidence annotations, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: keep the 188 claims with explicit SUPPORT or CONTRADICT evidence, assign binary relevance to their 209 annotated abstracts, and list the 112 excluded claims. It preserves the original-release checksum and verifies query text against the BEIR fixture. The BEIR test qrels instead contain 339 cited-document positives over all 300 original development claims; these are a different relevance definition.

The licenses above include the source licenses' warranty disclaimers. These data adaptations do not imply endorsement by the original authors.

See [benchmark methodology](../../RERANKER_BENCHMARK.md) for scoring, provenance, known label limitations, and reproduction commands.
