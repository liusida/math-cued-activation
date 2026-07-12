# Math-Cued Activation

An end-to-end pipeline for generating mathematical reasoning responses,
capturing residual-stream activations, fitting full-rank ICA, building feature
evidence, and exploring the result in the integrated v9 Explorer.

## Quick start

The checked-in reference configuration is
[`configs/vibethinker_imo.toml`](configs/vibethinker_imo.toml). All supported
commands require `--config` and are run from the repository root:

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

Stages resume safely when possible. Use `--force` only to replace outputs owned
by that stage. Registration is additive by default and preserves other layers
and annotations.

## vLLM generation

Generation uses an OpenAI-compatible vLLM server. The supported lifecycle
scripts start it in the background, wait for readiness, and stop it before
Transformers capture needs the GPU:

```bash
uv run python scripts/start_vllm.py --config configs/vibethinker_imo.toml
uv run python scripts/generate.py --config configs/vibethinker_imo.toml
uv run python scripts/stop_vllm.py --config configs/vibethinker_imo.toml
```

Server host, port, dtype, context length, GPU allocation, PID file, log file,
and startup/shutdown timeouts are configured in `[vllm]`.

Stop vLLM before capture so Transformers can use the GPU. Generation freezes
the exact token sequence; capture performs teacher-forced replay and never
regenerates the answer.

## Configuration

The versioned TOML config is the source of truth for model/dataset identity,
prompting, decoding, storage roots, capture layers, ICA parameters, enrichment,
and Explorer identity. Unknown keys, invalid layers, and invalid stage settings
fail before GPU or database work begins. Relative paths resolve from the
repository containing the config; `~` is expanded.

Existing response JSON, activation `.pt`, ICA checkpoint, feature artifact,
evidence JSON, and SQLite layouts remain compatible with the pre-refactor
pipeline.

## Source layout

- `scripts/`: supported user-facing stage commands.
- `src/math_cued_activation/`: shared configuration and pipeline logic.
- `explorer/`: vendored v9 Explorer server, frontend, and direct dependencies.
- `legacy/`: unsupported historical experiments and entrypoints.
- `configs/`: versioned pipeline configurations.

See [`PIPELINE.md`](PIPELINE.md) for stage invariants and artifact details.
See [`explorer/README.md`](explorer/README.md) for the boundary between tracked
Explorer source and ignored runtime artifacts.

Generation and teacher-forced capture are importable pipelines under
`src/math_cued_activation/generation/` and `src/math_cued_activation/capture/`.
ICA still retains a private format-compatibility implementation while it is
migrated incrementally.
