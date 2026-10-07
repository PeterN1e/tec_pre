# TEC Forecasting

This repository predicts global TEC maps from a sequence of TEC frames and
solar-geophysical indices.

## Configuration

Hyperparameters have a single source: the YAML files in `configs/`.

- `configs/base.yaml`: shared data, training, and runtime settings
- `configs/models/<slug>.yaml`: per-model settings, merged over the base
  (`model.params` is passed straight to the model constructor, and `training`
  to the trainer)

Model constructors and the trainer read no defaults of their own, so anything
not present in YAML is an error rather than a silently substituted value.
Edit the YAML to change a hyperparameter; `main.py` only asks which model to
run and whether to train or evaluate.

The dataset root can be overridden without touching YAML:

```powershell
$env:TEC_DATA_ROOT = "D:/Dataset_tec_NLY"
```

For scripted sweeps, `core.config.load_config` still accepts dotted overrides
programmatically:

```python
load_config("GA_Predrnn", overrides=["training.lr=0.0005", "model.params.hidden_dim=32"])
```

## Running

```powershell
python main.py
```

Then choose by number: the model (E_P_D asks a second question for its
time-series predictor), and the operation — 1 train + evaluate, 2 train only,
3 evaluate only. The resolved hyperparameters and the two YAML files they came
from are printed before the run starts.

Artifacts are written to `save/<model>/`:

- `config.yaml` — the fully merged config actually used for the run
- `model_state_dict.pth`
- `tec_scaler.pkl`
- `aux_scaler.pkl`
- `logs/`

Prediction and evaluation additionally write `evaluation/<split>/` with the
memory-mapped prediction arrays and `metrics.json`.

## Tests

```powershell
conda run -n tec_prediction python -m unittest tests.test_dataset tests.test_stage0_smoke tests.test_registry_forward tests.test_training_pipeline
```

The tests cover all registered model forward passes, dataset windows across
year boundaries, checkpoint loading, one training epoch, and prediction output.

