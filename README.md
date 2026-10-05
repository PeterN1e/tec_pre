# TEC Forecasting

This repository predicts global TEC maps from a sequence of TEC frames and
solar-geophysical indices.

## Configuration

The unified configuration lives in `configs/`:

- `configs/base.yaml`: shared data, training, and runtime defaults
- `configs/models/*.yaml`: model-specific overrides

Set `TEC_DATA_ROOT` to override the dataset root without editing YAML:

```powershell
$env:TEC_DATA_ROOT = "D:/Dataset_tec_NLY"
```

## Training

```powershell
python -m scripts.train --model ModelCanon
python -m scripts.train --model GA_Predrnn --evaluate-test
python -m scripts.train --model E_P_D --set model.params.predictor=tcn
```

Artifacts are written to `save/<model>/`:

- `config.yaml`
- `model_state_dict.pth`
- `tec_scaler.pkl`
- `aux_scaler.pkl`
- `logs/`

## Prediction and Evaluation

```powershell
python -m scripts.predict --model ModelCanon --split test
python -m scripts.evaluate --model GA_Predrnn --split val
```

Predictions and metrics are written under
`save/<model>/evaluation/<split>/`.

## Tests

```powershell
conda run -n tec_prediction python -m unittest discover -s tests -v
```

The tests cover all registered model forward passes, dataset windows across
year boundaries, checkpoint loading, one training epoch, and prediction output.

