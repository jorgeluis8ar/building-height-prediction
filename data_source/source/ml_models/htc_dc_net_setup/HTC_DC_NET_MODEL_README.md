# HTC-DC Net Model Anatomy For NYC + LA

This note explains the HTC-DC Net implementation we have vendored for the
building-height project and expands the current smoke-test status before we
move to Step 4.

## Current Implementation Status

The HTC-DC Net setup has passed the minimum implementation checks needed before
running a tiny overfit training experiment.

Verified items:

```text
HTC dataloader can load the combined NYC+LA dataset.
Released efficientnetb0 HTC-DC Net model builds.
One training-style smoke step completed:
  model(image, gt)
  loss_total.backward()
  optimizer.step()
W&B can be run offline using WANDB_MODE=offline.
```

What this means:

- The dataset folder structure is readable by the original HTC dataloader.
- The model can consume our `_IMG.tif`, `_BLG.tif`, and `_AGL.tif` chip
  triplets.
- The released EfficientNet backbone can be downloaded/cached through Torch
  Hub and attached to the HTC-DC Net height head.
- The model returns a valid prediction dictionary and loss dictionary.
- PyTorch can compute gradients through the model and update parameters.
- Weights & Biases will not block local testing if we run with
  `WANDB_MODE=offline`.

This does not yet mean the model is trained, well-calibrated, or producing
useful height predictions. It means the code path is alive and ready for the
first controlled training test.

## Repository Locations

Vendored HTC-DC Net code:

```text
data_source/source/ml_models/external/HTC-DC-Net/
```

Project-specific setup files:

```text
data_source/source/ml_models/htc_dc_net_setup/
```

Combined NYC + LA model-ready dataset:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/
```

Local smoke-test Python environment:

```text
data_source/source/ml_models/venv_htc_dc_net/
```

## Unit Of Observation

The HTC-DC Net model is trained on image chips, not individual buildings and
not standalone pixels.

For this first dataset, each observation is one aligned 256 x 256 spatial chip:

```text
image/<chip_id>_IMG.tif
mask/<chip_id>_BLG.tif
ndsm/<chip_id>_AGL.tif
```

The shared `<chip_id>` is the alignment key. Files with the same chip ID
describe the same spatial window.

## Inputs And Target

`_IMG.tif` is the image input. In our current version, it is a 3-band
PlanetScope RGB chip.

`_BLG.tif` is the building mask. Pixels inside merged building footprints are
1 and background pixels are 0.

`_AGL.tif` is the target height raster. It stores the building-only LiDAR nDSM
target in meters. This is our local equivalent of the HTC-DC Net
above-ground-level target.

The model learns:

```text
PlanetScope RGB chip -> building-height nDSM chip
```

The building mask is loaded in the ground-truth dictionary. The current HTC
loss implementation primarily uses positive nDSM pixels to define the valid
height region.

## Dataset Layout Expected By The Dataloader

The upstream dataloader expects this structure:

```text
data_dir/
  image/
    <chip_id>_IMG.tif
  mask/
    <chip_id>_BLG.tif
  ndsm/
    <chip_id>_AGL.tif
  train.txt
  val.txt
  test.txt
  image_stats.pickle
  ndsm_stats.pickle
```

Each split file contains one chip ID per line without the suffix. For example:

```text
los_angeles_20231203_182937_07_2488_chip_000134
```

The dataloader then constructs:

```text
image/los_angeles_20231203_182937_07_2488_chip_000134_IMG.tif
mask/los_angeles_20231203_182937_07_2488_chip_000134_BLG.tif
ndsm/los_angeles_20231203_182937_07_2488_chip_000134_AGL.tif
```

## Dataloader Components

The active dataloader code is in:

```text
data_source/source/ml_models/external/HTC-DC-Net/dataloaders.py
```

Key pieces:

- `get_train_val_dataloaders(cfgs)` reads `train.txt` and `val.txt`, checks
  image/nDSM statistics, and returns PyTorch dataloaders.
- `get_test_dataloaders(cfgs)` reads `test.txt` and returns test dataloaders.
- `GBHDataset` reads chip triplets and returns:

```text
chip_id, image, gt
```

where:

```text
image: Tensor shaped [3, 256, 256]
gt["ndsm"]: Tensor shaped [1, 256, 256]
gt["mask"]: Tensor shaped [1, 256, 256], if use_mask is true
```

The dataloader normalizes the RGB image using `image_stats.pickle`. It clips
negative nDSM values to zero.

## Model Builder

The model builder is:

```text
data_source/source/ml_models/external/HTC-DC-Net/build.py
```

For our config:

```yaml
model: htcdc
backbone: efficientnetb0
optimizer: AdamW
lr: 0.0001
```

`get_model_and_optimizer(cfgs)` creates:

```text
model: UBins
optimizer: AdamW
```

## Main Model Class

The main model class is:

```text
UBins
```

defined in:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
```

`UBins` wraps the HTC-DC Net architecture and exposes the training call:

```python
losses, pred = model(image, gt)
```

During validation/evaluation, the model returns:

```python
losses, pred, metric_params = model(image, gt)
```

## Backbone

The released configuration uses:

```text
efficientnetb0
```

In this implementation, the EfficientNet backbone is loaded from Torch Hub:

```text
rwightman/gen-efficientnet-pytorch
```

The classifier and global pooling layers are replaced with identity layers so
the network can act as a feature extractor rather than an image classifier.

Conceptually:

```text
RGB Planet chip -> EfficientNet feature maps
```

These feature maps feed the decoder.

## Encoder

The `Encoder` class walks through the EfficientNet modules and stores
intermediate feature maps.

Those intermediate maps matter because height estimation is spatial. The model
needs high-level context from deep layers and finer spatial information from
shallower layers.

## Decoder

The decoder class is:

```text
DecoderBN
```

It upsamples EfficientNet features back toward image resolution using skip
connections. The core upsampling block is:

```text
UpSampleBN
```

The decoder produces one or more feature maps depending on `fusion_mode`.

Our current config uses the upstream default:

```text
fusion_mode: last
```

This produces multiple prediction levels that are later combined through
weighted losses.

## Adaptive Bins

The central HTC-DC Net idea is not to directly regress height with a single
plain convolution. Instead, it predicts adaptive height bins and a per-pixel
probability distribution over those bins.

The main class is:

```text
MultiLevelUnetAdaptiveBins
```

For each prediction level, it:

1. Uses a small vision-transformer module to estimate bin widths.
2. Converts bin widths into cumulative bin edges.
3. Computes bin centers.
4. Predicts a per-pixel probability distribution over bins.
5. Produces height as the expected value over bin centers.

In simplified notation:

```text
predicted_height_pixel = sum(probability_bin_k * height_center_bin_k)
```

This is useful for height prediction because the model can adapt its
resolution over the height range instead of using fixed evenly spaced bins.

## Mini Vision Transformer Blocks

The adaptive-bin modules are in:

```text
data_source/source/ml_models/external/HTC-DC-Net/parts/miniViT.py
```

The relevant modules are:

```text
mViT
mViTHTC
```

These modules summarize patch-level information and help estimate adaptive
height-bin distributions.

## Output Dictionary

For the successful smoke test, the model returned:

```text
pred keys: bin, ndsm, ndsm_intermediate
```

Meaning:

- `pred["ndsm"]`: final predicted height raster.
- `pred["ndsm_intermediate"]`: intermediate predicted height rasters from the
  multi-level decoder/adaptive-bin heads.
- `pred["bin"]`: learned bin edges for the adaptive height bins.

The final prediction has the same spatial size as the target nDSM after
interpolation.

## Loss Dictionary

For the successful smoke test, the model returned losses such as:

```text
mae_0
mae_1
mae_2
bin_chamfer_0
bin_chamfer_1
bin_chamfer_2
loss_total_0
loss_total_1
loss_total_2
loss_total
```

Meaning:

- `mae_*`: masked mean absolute height error at each prediction level.
- `bin_chamfer_*`: regularization loss that encourages adaptive bins to cover
  the empirical height distribution.
- `loss_total_*`: level-specific total loss.
- `loss_total`: weighted total loss used for backpropagation.

The current config uses:

```yaml
chamfer_weight: 0.01
```

so the primary signal is still height prediction error, with a smaller
regularization term for the adaptive bins.

## Training Loop

The training entry point is:

```text
data_source/source/ml_models/external/HTC-DC-Net/train.py
```

The loop does:

```python
for chip_id, image, gt in train_dataloader:
    image = data_to_device(image, device)
    gt = data_to_device(gt, device)
    losses, pred = model(image, gt)
    loss_total = losses["loss_total"]
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
```

This is exactly the core path we verified in the smoke test.

## Validation Loop

During validation, the model is set to evaluation mode and called under
`torch.no_grad()`.

The validation path returns:

```text
losses
predictions
metric_params
```

Those metric parameters are accumulated and then summarized by the model's
`evaluate` method.

The training config currently tracks:

```yaml
early_stopping:
  - mae
  - rmse
  - val/loss_total
```

## W&B Logging

The upstream code initializes Weights & Biases in `train.py`.

For local testing, we should run:

```bash
WANDB_MODE=offline
```

This keeps the logging calls alive without requiring online syncing.

This matters because the upstream trainer calls `wandb.init`, `logger.log`,
and `logger.watch(model)`. Offline mode lets us keep the original trainer
structure while avoiding authentication/network friction during smoke tests.

## Current Project Configs

Dataset/config file:

```text
data_source/source/ml_models/htc_dc_net_setup/configs/nyc_la_rgb_v1.yaml
```

Model/training config file:

```text
data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_v1.yaml
```

Key current settings:

```yaml
data_dir: data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1
image_size: 256
use_mask: True
normalize: True
seed: 20260706
device: cpu
model: htcdc
backbone: efficientnetb0
batch_size: 1
num_workers: 0
max_epochs: 1
optimizer: AdamW
lr: 0.0001
```

These settings are intentionally conservative. The purpose is to prove the
pipeline works before scaling up training.

## Local Compatibility Fixes Already Applied

The vendored code needed a few small compatibility edits before it could run
locally:

- Fixed a syntax error in `build.py`.
- Added a missing `use_vis` assignment in `dataloaders.py`.
- Allowed `GBHDataset` to accept extra keyword arguments passed by the
  upstream loader wrapper.
- Made local imports in `htcdc.py` robust to both top-level and package-style
  imports.
- Added a small Chamfer-distance fallback for local smoke tests when
  PyTorch3D is unavailable.
- Added `trust_repo=True` for the Torch Hub EfficientNet call so non-
  interactive smoke tests do not stop at a trust prompt.

## Successful Smoke-Test Result

The latest successful check used:

```text
chip_id: los_angeles_20231203_182937_07_2488_chip_000134
model_class: UBins
optimizer_class: AdamW
pred keys: bin, ndsm, ndsm_intermediate
loss_total: 130.27557373046875
backward_step_ok: True
smoke_ok: True
```

This verifies the implementation path from data loading through model build,
forward pass, loss computation, gradient computation, and optimizer update.

## What Step 4 Should Test

The next step should be a tiny overfit experiment. The purpose is to confirm
that the model can reduce training loss on a very small number of chips.

Recommended Step 4 checks:

1. Run overfit mode for more than one epoch on two chips.
2. Confirm `train/loss_avg` decreases.
3. Save checkpoints under `data_source/data/ml_models/generated/htc_dc_net/runs/`.
4. Inspect predicted nDSM images for the overfit chips.
5. Only after that, attempt a normal train/validation run.

If loss cannot decrease on two chips, we should debug the data scale, mask use,
normalization, loss, and target rasters before full training.
