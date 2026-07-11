# Legacy scripts

These files are preserved for reproducibility but are no longer supported
pipeline interfaces. They may retain hard-coded models, datasets, paths, or
dependencies. Supported workflows use the scripts in `../scripts/` with
`--config configs/vibethinker_imo.toml`.

- `scripts/`: historical generation, diagnostics, and plotting entrypoints.
- `tools/`: one-off operational and reproduction helpers.
- `notes/`: historical prompts, model lists, and vLLM notes.
- `charts/`: historical architecture diagrams.

| Previous workflow | Supported replacement |
| --- | --- |
| `generate_imo_text*.py` | `scripts/generate.py` |
| `capture_imo_activations.py` | `scripts/capture.py` |
| `fit_ica_qwen_vibethinker_mixed.py` | `scripts/fit.py` |
| diagnostic inspection/plot scripts | Integrated Explorer or an explicit legacy invocation |

Code under `src/math_cued_activation/` and `scripts/` must not import this
directory.
