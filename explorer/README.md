# Explorer subsystem

This directory contains the integrated feature Explorer.

Tracked source:

- `server/`: FastAPI application and browser UI assets.
- `src/ica_lens_v9/`: feature indexing, probing, plotting, annotation, SAE,
  and model-runtime implementation imported from the former v9 project.
- `tools/`: format-compatible registration and enrichment implementations.
- `config/`: SAE counterpart definitions needed by Explorer endpoints.

Local runtime data:

- `feature_index.sqlite`: feature index, labels, and annotations.
- `runs/`: ICA feature artifacts, plots, histograms, and evidence.
- `annotations/`: migrated annotation source and response files.

Runtime data is intentionally ignored by Git. It can be created by
`scripts/register.py` and `scripts/enrich.py`, or migrated from another
installation. The supported web entrypoint is `scripts/serve.py --config ...`;
do not invoke `server/app.py` directly.

The internal package name `ica_lens_v9` is retained temporarily to preserve
serialized/import compatibility. It does not imply a dependency on the old v9
repository.
