# HTC-DC Net Environment And Config Setup

## Selected Project Model

The selected model is the four-channel off-nadir RGB+NIR EfficientNet-B0 run:

```text
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded
```

Before reproducing the model on another computer, read
`../SELECTED_HTC_DC_MODEL.md`. It records the exact dataset, parameters,
checkpoint, rerun command, and generated files that are excluded from GitHub
and must be transferred or rebuilt separately.

This folder contains project-specific setup files for running the cloned
HTC-DC Net code on the NYC + LA RGB v1 dataset.

For a model-by-model explanation of the HTC-DC Net architecture, dataloader,
inputs, outputs, losses, and smoke-test meaning, see:

```text
data_source/source/ml_models/htc_dc_net_setup/HTC_DC_NET_MODEL_README.md
```

For the future design to use both winter and summer PlanetScope scenes as
model inputs, see:

```text
data_source/source/ml_models/htc_dc_net_setup/TWO_SEASON_PLANETSCOPE_INPUT_README.md
```

For the seven main model/data tuning knobs and how each affects accuracy, see:

```text
data_source/source/ml_models/htc_dc_net_setup/HTC_DC_NET_TUNING_GUIDE.md
```

For the repo-style 40-chip calibration parameters, including how each
parameter enters the HTC-DC Net code and how changing it affects training, see:

```text
data_source/source/ml_models/htc_dc_net_setup/HTC_DC_NET_REPO_PARAMETERS_README.md
```

## External Code

The HTC-DC Net repository is cloned here:

```text
data_source/source/ml_models/external/HTC-DC-Net/
```

Current cloned commit:

```text
adae55edc8be589757cec57f839d59a681d93364
```

Local compatibility fixes applied to the vendored code:

- `build.py` was missing a colon after `def get_model_and_optimizer(...)`,
  which prevented imports from succeeding.
- `dataloaders.py` referenced `use_vis` before assignment in
  `get_train_val_dataloaders`.
- `dataloaders.py` passed extra keyword arguments to `GBHDataset`; the dataset
  constructor now accepts those unused compatibility kwargs and records `rcnn`.
- `htcdc.py` imported local modules as top-level modules only; it now supports
  both top-level and package-relative imports.
- `htcdc.py` required `pytorch3d` for Chamfer distance; it now has a small
  local fallback for smoke tests on machines where PyTorch3D is unavailable.
- `htcdc.py` now sets `trust_repo=True` for the exact upstream Torch Hub
  EfficientNet repository used by the released config, avoiding an interactive
  prompt during non-interactive runs.

## Dataset

The configured dataset is:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/
```

It contains:

```text
image/
mask/
ndsm/
train.txt
val.txt
test.txt
all.txt
chips_manifest.csv
```

The HTC dataloader expects `image_stats.pickle` and `ndsm_stats.pickle` at the
dataset root and reads them with `torch.load`. Run the compatibility prep
script after installing PyTorch:

```bash
python data_source/source/ml_models/htc_dc_net_setup/prepare_htc_dataset_stats.py
```

This writes:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/image_stats.pickle
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/ndsm_stats.pickle
```

## Config Files

Use these config files for the first local smoke test:

```text
data_source/source/ml_models/htc_dc_net_setup/configs/nyc_la_rgb_v1.yaml
data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_v1.yaml
```

They are intentionally small and CPU-oriented:

```text
batch_size: 1
num_workers: 0
max_epochs: 1
device: cpu
```

The first purpose is not full training. The first purpose is proving that the
HTC dataloader, model build, and one forward/backward pass work on our dataset.

## Local Environment

The upstream `environment.yaml` targets Linux/CUDA and older Python/PyTorch
versions. On this Apple Silicon machine, use a fresh Python environment and
install:

```bash
python3 -m venv data_source/source/ml_models/venv_htc_dc_net
data_source/source/ml_models/venv_htc_dc_net/bin/python -m pip install --upgrade pip
data_source/source/ml_models/venv_htc_dc_net/bin/python -m pip install -r data_source/source/ml_models/htc_dc_net_setup/requirements-apple-silicon-smoke.txt
```

For offline Weights & Biases smoke tests:

```bash
export WANDB_MODE=offline
```

The released EfficientNet backbone is fetched through Torch Hub. For a
repeatable local cache location, use:

```bash
export TORCH_HOME=/private/tmp/torch_htc_cache
```

Smoke-test command from the project repo root:

```bash
WANDB_MODE=offline TORCH_HOME=/private/tmp/torch_htc_cache \
  data_source/source/ml_models/venv_htc_dc_net/bin/python data_source/source/ml_models/external/HTC-DC-Net/train.py \
    --config data_source/source/ml_models/htc_dc_net_setup/configs/nyc_la_rgb_v1.yaml \
    --exp_config data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_v1.yaml \
    --overfit
```

## Current Verification Status

- HTC-DC Net repo cloned successfully.
- Project-specific config files created.
- Apple Silicon smoke-test requirements file created.
- Vendored `build.py` syntax error fixed.
- Vendored `dataloaders.py` local compatibility issues fixed.
- Apple Silicon smoke-test environment created at
  `data_source/source/ml_models/venv_htc_dc_net/`.
- Dataset-root `image_stats.pickle` and `ndsm_stats.pickle` were created for
  HTC dataloader compatibility.
- Dataloader smoke test passed on the combined NYC+LA dataset:
  image batches load as `(1, 3, 256, 256)`, nDSM batches as
  `(1, 1, 256, 256)`, masks as `(1, 1, 256, 256)`, and mask values are
  binary `{0, 1}`.
- Released `efficientnetb0` HTC-DC Net config imports and builds successfully.
- One training-style smoke step completed successfully: `model(image, gt)`
  returned losses and predictions, `loss_total.backward()` ran, and
  `optimizer.step()` completed.

The successful smoke step used chip
`los_angeles_20231203_182937_07_2488_chip_000134` and produced these core
objects:

```text
model_class: UBins
optimizer_class: AdamW
prediction keys: bin, ndsm, ndsm_intermediate
loss_total: 130.27557373046875
```
