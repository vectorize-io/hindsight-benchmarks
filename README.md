# Hindsight Benchmarks

The code and results behind [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io).

- **Benchmarks:** Hindsight as a whole on the [Agent Memory Benchmark](https://agentmemorybenchmark.ai).
- **Model leaderboards:** which model to run behind each Hindsight operation: retain, recall, consolidation, reflect and refresh pages.

## Layout

```
benchmark-runner/   hindsight-evals: runs Hindsight's system evals against a model
results/            one JSON per model run
visualizer/         the site (Next.js, static export)
```

## Run the evals for a model

The runner pins a Hindsight commit in `benchmark-runner/evals.json`, checks it out, and runs its `hindsight-system-evals` one operation at a time.

```bash
cd benchmark-runner
export OPENROUTER_API_KEY=...
uv run hindsight-evals --model qwen/qwen3.8-flash                           # every operation
uv run hindsight-evals --model qwen/qwen3.8-flash --operations consolidation
uv run hindsight-evals --model google/gemma-4-26b-a4b-it --decision openrouter:cloudflare/clef
```

- **Judge:** Codex. You need the `codex` CLI, logged in.
- **Cost:** the tokens each case really used, at the provider's list price.
- **Results:** written to `results/leaderboard/evals/<provider>-<model>.json`. A run of some operations updates only those operations in the file.

## Run the site

```bash
cd visualizer
npm install
npm run dev     # http://localhost:9998
```
