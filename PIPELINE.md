# Math-Cued ICA Explorer Pipeline

> The commands below describe the original execution in detail. The integrated
> project now exposes the same workflow through `scripts/generate.py`,
> `capture.py`, `fit.py`, `register.py`, `enrich.py`, `validate.py`, and
> `serve.py`. Each takes `--config configs/vibethinker_imo.toml`; the TOML file
> replaces the individual flags shown in historical examples below.

Use `scripts/start_vllm.py --config ...` immediately before generation and
`scripts/stop_vllm.py --config ...` immediately afterward. The server runs in
the background and its PID and log paths are controlled by `[vllm]`.

This document describes the end-to-end workflow for choosing a causal language
model and a problem set, generating responses, capturing an intermediate
residual stream, fitting ICA, building derived feature evidence, importing it
into SQLite, and inspecting it with the v9 Explorer.

The running example uses:

- model: `WeiboAI/VibeThinker-3B`
- dataset: `OpenEvals/IMO-AnswerBench`
- captured site: transformer block residual-post
- captured layer: `layer_20` (zero-based block index 20)
- ICA: full-rank, 2,048 components, 1,000,000 sampled token rows
- feature interface: positive and negative sides of every ICA component, giving
  4,096 nonnegative Explorer features

## Conceptual data flow

```text
model + problem set
        |
        v
vLLM response generation
        |
        v
saved prompt, response, and exact token sequence
        |
        v
Transformers teacher-forced replay with a residual-post hook
        |
        v
one activation bundle per problem and layer
        |
        v
row L2 normalization + centering + whitening + full-rank FastICA
        |
        v
split each signed ICA component into positive/negative features
        |
        v
kurtosis ordering + distributions + histograms + top samples
        |
        v
SQLite feature index + files referenced by SQLite
        |
        v
FastAPI/HTML Explorer
```

Response generation and activation capture are separate stages. Generation is
autoregressive. Capture is a deterministic full-sequence replay over the saved
tokens; it does not regenerate the answer.

## 1. Choose the model, dataset, layer, and decoding policy

Record these choices before running the pipeline:

- exact Hugging Face model ID and revision
- dataset ID, split, revision, and row identifiers
- chat template and user prompt
- generation backend and decoding settings
- activation site and zero-based layer index
- activation dtype
- number and sampling policy of token rows used for ICA
- ICA seed, iteration limit, tolerance, nonlinearity, and whitening solver

For the existing VibeThinker dataset, vLLM generation used greedy decoding:

```text
temperature = 0.0
```

The capture site is the output of a transformer block before the final model
normalization. This is called residual-post in this project. For example,
`--capture-layer 20` captures block index 20, the model's 21st block.

## 2. Generate responses with vLLM

Use a separate vLLM environment. The following example serves one model on one
GPU with its model-faithful 64K context window:

```bash
cd ~/research/Math-Cued-Activation
source ~/vllm-env/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 \
vllm serve WeiboAI/VibeThinker-3B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95
```

In another shell, generate all 400 responses:

```bash
cd ~/research/Math-Cued-Activation

uv run python scripts/generate_imo_text_vllm.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model WeiboAI/VibeThinker-3B \
  --server-name local \
  --sample-size 400 \
  --start-index 0 \
  --temperature 0 \
  --concurrency 6 \
  --generated-text-dir outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B
```

The generation directory contains human-readable text and JSON/token metadata.
Preserve the exact token sequence: activation replay must use the same tokens,
not a newly generated response.

For a new dataset, replace or generalize the IMO-specific dataset loading and
prompt construction in `generate_imo_text_vllm.py` and `run_vibethinker.py`.

## 3. Capture residual-post activations

Stop vLLM before capture so Transformers has clean GPU memory. Capture all
problems at layer 20:

```bash
cd ~/research/Math-Cued-Activation

uv run python scripts/capture_imo_activations.py \
  --model WeiboAI/VibeThinker-3B \
  --sample-size 400 \
  --start-index 0 \
  --capture-layer 20 \
  --capture-prompt-activations \
  --activation-dtype float16 \
  --generated-text-dir outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B \
  --activation-dir ~/data/ICA-data/math-cued-activation
```

The output directory is:

```text
~/data/ICA-data/math-cued-activation/
  OpenEvals__IMO-AnswerBench/
  WeiboAI__VibeThinker-3B/
  layer_20/
```

Each `.pt` bundle stores the prompt, response, token IDs, prompt activations,
generated-token activations, combined activations, and provenance metadata.
Generated-token states are computed with the prompt and all preceding generated
tokens present.

The optional next-token sanity check is designed for the final layer. When
capturing an intermediate layer, it performs an additional final-layer replay;
it does not validate that the intermediate state directly predicts every saved
token.

## 4. Fit full-rank ICA

The current Math-Cued fit samples 1,000,000 activation rows, L2-normalizes every
row, and fits 2,048 components because VibeThinker has hidden size 2,048:

```bash
cd ~/research/Math-Cued-Activation

uv run python scripts/fit_ica_qwen_vibethinker_mixed.py \
  --activation-root ~/data/ICA-data/math-cued-activation \
  --source vibethinker \
  --layer 20 \
  --max-vibethinker-activations 1000000 \
  --seed 0 \
  --max-iter 100 \
  --device cuda
```

Expected checkpoint:

```text
~/research/Math-Cued-Activation/results/ica/
  vibethinker_only_layer20_c2048_iter100.pt
```

For a new model, the number of full-rank components follows the model hidden
size. The current fit script has model-specific slugs and source choices; these
must be generalized or extended for a model other than its known Qwen and
VibeThinker configurations.

## 5. Register the layer in the Explorer run

One model should remain one Explorer model entry. Multiple ICA layers should be
registered under the same run, so the UI model selector remains
`VibeThinker-3B` and the layer selector contains `layer_20` and `layer_32`.

```bash
cd ~/research/ICA-paper/ICA-paper-prep/v9

uv run python diagnostics/math-cued-ica/register_math_cued_ica.py \
  --checkpoint ~/research/Math-Cued-Activation/results/ica/vibethinker_only_layer20_c2048_iter100.pt \
  --run-id math_cued_vibethinker_only_layer32_c2048_iter100 \
  --model-id WeiboAI/VibeThinker-3B \
  --display-name "VibeThinker-3B" \
  --layer layer_20 \
  --add-layer
```

`--add-layer` preserves every other layer and its SQLite annotations/manual
labels. Do not use `--force-db`; it recreates the shared diagnostic database.

Registration creates the feature artifact and placeholder SQLite rows. At this
point feature IDs are still temporary component/sign IDs, and distribution
statistics, mini histograms, and top samples do not exist yet.

## 6. Compute properties and finalize feature IDs

This step must precede all feature-ID-dependent evidence. It computes feature
distributions over 500,000 sampled rows and orders exposed feature IDs by
descending kurtosis. For a VibeThinker-only run, the shared-order calculation
contains one model view and therefore reduces to that run's kurtosis order.

```bash
cd ~/research/ICA-paper/ICA-paper-prep/v9

uv run python diagnostics/math-cued-ica/populate_feature_properties.py \
  --activation-root ~/data/ICA-data/math-cued-activation \
  --run-id math_cued_vibethinker_only_layer32_c2048_iter100 \
  --layer layer_20 \
  --max-rows 500000 \
  --seed 0 \
  --device cuda \
  --dtype float32
```

This updates the feature tensor artifact, ranking/histogram CSVs, layer
metadata, and SQLite feature rows. Do not subsequently run the older standalone
`sort_features_by_kurtosis.py` for this run; property population already applies
the final ordering.

Because reordering changes feature IDs, manual labels and annotations should be
created only after this step.

## 7. Render mini histograms

SQLite stores paths to mini-histogram SVGs; it does not store SVG content.
Render those files after property population:

```bash
uv run python -m ica_lens_v9.features.plot_cli \
  --feature-interface-dir diagnostics/math-cued-ica/runs/math_cued_vibethinker_only_layer32_c2048_iter100/feature_interfaces/split_origin_relu \
  --layers layer_20 \
  --mini-histogram-svgs \
  --force
```

Until this command runs, `/api/mini-histogram/...` returns HTTP 404 even though
the feature row exists.

## 8. Populate top activating samples

Run this only after feature ordering is final:

```bash
uv run python diagnostics/math-cued-ica/populate_top_samples.py \
  --activation-root ~/data/ICA-data/math-cued-activation \
  --run-id math_cued_vibethinker_only_layer32_c2048_iter100 \
  --layer layer_20 \
  --max-rows 500000 \
  --seed 0 \
  --examples 10 \
  --context-window 32 \
  --chunk-size 4096 \
  --write-workers 8 \
  --device cuda \
  --dtype float32
```

Evidence JSON is written under `layer_20_top_samples/`, and SQLite is updated
with each feature's evidence path and JSON payload.

## 9. Start and use Explorer

```bash
cd ~/research/ICA-paper/ICA-paper-prep/v9

ICA_V9_FEATURE_DB=$PWD/diagnostics/math-cued-ica/feature_index.sqlite \
uv run uvicorn server.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Open `http://localhost:8000/`. Select `VibeThinker-3B`, then `layer_20` or
`layer_32`.

The Explorer supports live text probing, top ICA features per token, manual
labels stored only in SQLite, feature details, activation distributions, top
samples, selected-layer logit-lens predictions, and actual final-model
next-token predictions.

## Required execution order

The safe order is:

1. Generate and freeze responses.
2. Capture one or more layers from those exact responses.
3. Fit ICA separately for each layer.
4. Register the checkpoint as a new run or add it to an existing model run.
5. Populate properties and finalize feature IDs.
6. Render mini histograms and ranking plots.
7. Populate top samples.
8. Add manual or automated annotations.
9. Explore through the server.

The canonical integrated commands are:

```bash
uv run python scripts/start_vllm.py --config configs/vibethinker_imo.toml
uv run python scripts/generate.py --config configs/vibethinker_imo.toml
uv run python scripts/stop_vllm.py --config configs/vibethinker_imo.toml
uv run python scripts/capture.py --config configs/vibethinker_imo.toml
uv run python scripts/fit.py --config configs/vibethinker_imo.toml
uv run python scripts/register.py --config configs/vibethinker_imo.toml
uv run python scripts/enrich.py --config configs/vibethinker_imo.toml
uv run python scripts/validate.py --config configs/vibethinker_imo.toml
uv run python scripts/serve.py --config configs/vibethinker_imo.toml
```

Do not produce annotations or top-sample evidence before final feature ordering,
because the meaning of an exposed feature ID changes during the ordering step.

## Absolute path inventory

The current pipeline is split across two repositories and one data root.

### Response generation, replay, and ICA fitting

Repository root:

```text
/home/liusida/research/Math-Cued-Activation
```

Scripts:

```text
/home/liusida/research/Math-Cued-Activation/scripts/generate_imo_text_vllm.py
/home/liusida/research/Math-Cued-Activation/scripts/capture_imo_activations.py
/home/liusida/research/Math-Cued-Activation/scripts/run_vibethinker.py
/home/liusida/research/Math-Cued-Activation/scripts/fit_ica_qwen_vibethinker_mixed.py
```

Generated responses:

```text
/home/liusida/research/Math-Cued-Activation/outputs/imo-answerbench-responses
```

ICA checkpoints:

```text
/home/liusida/research/Math-Cued-Activation/results/ica
```

### Captured activation data

```text
/home/liusida/data/ICA-data/math-cued-activation
```

### Feature processing, SQLite, and Explorer

Repository root:

```text
/home/liusida/research/ICA-paper/ICA-paper-prep/v9
```

Scripts and modules:

```text
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/diagnostics/math-cued-ica/register_math_cued_ica.py
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/diagnostics/math-cued-ica/populate_feature_properties.py
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/diagnostics/math-cued-ica/populate_top_samples.py
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/src/ica_lens_v9/features/plot_cli.py
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/server/app.py
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/server/static/index.html
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/server/static/feature.html
```

SQLite database:

```text
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/diagnostics/math-cued-ica/feature_index.sqlite
```

Registered artifacts, histograms, and top-sample evidence:

```text
/home/liusida/research/ICA-paper/ICA-paper-prep/v9/diagnostics/math-cued-ica/runs
```

## Guidance for a future integrated project

A clean implementation should replace hard-coded model/run maps with a single
versioned pipeline manifest containing:

- model ID, revision, tokenizer, hidden size, and layer count
- dataset ID, revision, split, ID field, prompt field, and answer field
- prompt/chat-template configuration
- generation backend and complete decoding parameters
- response and activation roots
- activation site, layers, and dtype
- ICA preprocessing and fit parameters
- feature ordering and statistics sampling parameters
- SQLite run identity and display name

The integrated project should expose explicit commands such as:

```text
generate -> capture -> fit -> register -> enrich -> serve
```

It should also make every stage resumable, write complete provenance, validate
that all response token sequences replay correctly, use atomic artifact/SQLite
updates, and prevent downstream evidence from being generated before feature
IDs are finalized.
