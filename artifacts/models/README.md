# Versioned model artifacts

Store every experiment model under this directory using incremental ids:

- `artifacts/models/model_001/`
- `artifacts/models/model_002/`
- `artifacts/models/model_003/`

Each folder should contain at minimum:

- `lgbm.txt` (trained LightGBM model for that iteration)
- optional notes such as `metrics.txt` or `notes.md`

`scripts/train.py` now supports:

```bash
uv run python scripts/train.py --model-id model_002
```

This saves to `artifacts/models/model_002/lgbm.txt` and also updates
`model/lgbm.txt` as the latest compatibility path for existing tooling.
