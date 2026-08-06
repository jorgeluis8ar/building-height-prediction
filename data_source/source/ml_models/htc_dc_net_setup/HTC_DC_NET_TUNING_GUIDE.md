# HTC-DC Net Tuning Guide

This note explains the seven main parts of the current HTC-DC Net setup that
we can tweak to improve model behavior. The immediate purpose is not to tune
blindly. The purpose is to understand what each object controls before running
larger experiments.

The current model is still in infrastructure-testing mode. Before we interpret
accuracy, the model should first prove that it can overfit a very small sample
of chips.

## 1. Number Of Epochs

Current mini-run value:

```text
epochs: 1
```

Where it is controlled:

```text
data_source/source/ml_models/run_htc_mini_training.py --epochs
data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_v1.yaml
```

What it does:

An epoch is one full pass through the training chips. With 10 chips and batch
size 1, one epoch means the model sees only 10 training batches. That is enough
to test that the code runs, but it is not enough to learn useful weights.

How it affects the model:

- Too few epochs: predictions mostly reflect pretrained initialization and
  random early updates.
- More epochs: the model has more chances to adapt to our PlanetScope and
  LiDAR nDSM data.
- Too many epochs on a small sample: the model may memorize those chips, which
  is actually useful for an overfit diagnostic but not a final model.

Why it matters now:

For Step 4, the question is:

```text
Can the HTC-DC Net model reduce loss on 10 chips?
```

If it cannot overfit 10 chips, we should debug the pipeline before using more
data.

Recommended first tests:

```text
epochs: 20
epochs: 50
epochs: 100
```

Expected behavior:

Training loss should fall clearly. If it stays flat, likely issues include
learning rate, target scaling, mask/loss definition, image normalization, or a
model-output scale mismatch.

## 2. Learning Rate

Current value:

```yaml
lr: 0.0001
```

Where it is controlled:

```text
data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_v1.yaml
data_source/source/ml_models/run_htc_mini_training.py --lr
```

What it does:

The learning rate controls how large each optimizer update is after
backpropagation.

Current optimizer:

```yaml
optimizer: AdamW
```

How it affects the model:

- Too low: training is stable but slow; loss may barely move in a short run.
- Too high: training may become unstable; loss may jump, produce extreme
  predictions, or become `NaN`.
- Reasonable range: the model should reduce loss steadily without exploding.

Why it matters for the mini-run:

The current one-epoch run produced predictions, but not trained predictions.
For a tiny overfit experiment, we often need a larger learning rate than we
would use in a full training run.

Recommended first tests:

```text
lr: 0.001
lr: 0.0005
lr: 0.0001
```

What to monitor:

```text
training_history.csv
loss_total by epoch
predicted nDSM visual range
pred_mean_m versus target_mean_m
```

Practical interpretation:

If `lr = 0.001` learns quickly and remains stable, use it for overfit tests.
If it produces unstable predictions, drop to `0.0005` or `0.0001`.

## 3. Mask And Loss Behavior

Current behavior in the HTC implementation:

```python
mask = gt["ndsm"] > 0
```

Location:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
UBins.get_losses()
```

What it does:

The loss is computed mainly on pixels where the LiDAR nDSM target is positive.
This means background pixels and zero-height pixels do not dominate the height
loss.

Current ground-truth dictionary:

```text
gt["ndsm"]  height target, meters
gt["mask"]  building footprint mask
```

Important distinction:

```text
gt["ndsm"] > 0
```

selects pixels with positive target height.

```text
gt["mask"] > 0
```

selects pixels inside building footprints, whether or not the target height is
positive.

How it affects the model:

- Using `gt["ndsm"] > 0` focuses the model on valid positive height labels.
- Using `gt["mask"] > 0` tells the model that every building-footprint pixel
  matters, including any zero-valued target pixels.
- Using `gt["mask"] > 0` and `gt["ndsm"] > 0` together is stricter and likely
  safest for our current data.

Potential issue:

If some valid building pixels have target value 0 because LiDAR processing
failed or because target coverage is incomplete, then `gt["ndsm"] > 0` silently
excludes them. That may be good for training, but we should audit how many
footprint pixels are excluded.

Candidate loss masks:

```python
mask = gt["ndsm"] > 0
mask = gt["mask"] > 0
mask = (gt["mask"] > 0) & (gt["ndsm"] > 0)
```

Recommended first test:

Keep the current loss for the first overfit test. Then compare it against:

```python
mask = (gt["mask"] > 0) & (gt["ndsm"] > 0)
```

Why:

This keeps the training target explicitly inside building footprints and
positive-height pixels.

## 4. Height Target Scaling And Height Range

Current target:

```text
_AGL.tif stores LiDAR nDSM height in meters
```

Current model behavior:

`UBins` reads the maximum height range from:

```text
ndsm_stats.pickle
```

Location:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
UBins.__init__()
```

Relevant code concept:

```python
_, _, _, self.h_max, _ = torch.load(ndsm_stats_file)
```

What it does:

HTC-DC Net predicts adaptive height bins between `h_min` and `h_max`. The model
then predicts a probability distribution over those bins for each pixel.

How it affects the model:

- If `h_max` is very large, many bins cover heights that are rare in the
  training chips.
- If most mini-run chips are low-rise but `h_max` allows very tall buildings,
  initial predictions can be biased high.
- If `h_max` is too low, tall buildings are artificially capped and validation
  will be biased.

Important project constraint:

We should not cap the final project labels too low because NYC and LA contain
very tall buildings. However, for a tiny overfit diagnostic, it can be useful
to test whether a tighter temporary range helps learning.

Possible tweaks:

```text
Use full h_max from ndsm_stats.pickle
Use temporary h_max from mini-dataset maximum
Use percentile-based h_max for diagnostics only
Normalize target heights before model loss
```

Recommended approach:

For final modeling, preserve true meter heights and avoid arbitrary caps.
For diagnostics, add an optional experiment config field such as:

```yaml
override_h_max: 120
```

or:

```yaml
target_scale: 0.01
```

only if the model cannot overfit the tiny sample.

What to monitor:

```text
pred_mean_m
target_mean_m
bias_m
prediction raster min/max
```

If predictions start around 20-40 m for low-rise chips, the adaptive-bin range
or initialization may be too broad for fast small-sample learning.

## 5. Adaptive Bin Settings

Current model component:

```text
MultiLevelUnetAdaptiveBins
```

Location:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
data_source/source/ml_models/external/HTC-DC-Net/parts/miniViT.py
```

Key settings:

```text
num_classes / num_bins
patch_size
chamfer_weight
fusion_mode
```

### Number Of Bins

Default:

```text
num_classes: 256
```

What it does:

This controls how many adaptive height intervals the model predicts.

How it affects the model:

- More bins: more detailed height distribution, but more parameters and harder
  optimization.
- Fewer bins: simpler height distribution, easier debugging, possibly less
  fine-grained predictions.

Candidate tests:

```text
num_classes: 64
num_classes: 128
num_classes: 256
```

For infrastructure tests, `64` or `128` may be easier to optimize.

### Patch Size

Current value:

```yaml
patch_size: 4
```

What it does:

Patch size controls how the mini vision-transformer module groups spatial
features when estimating adaptive bins.

How it affects the model:

- Smaller patch size: more local detail, more computation.
- Larger patch size: more spatial aggregation, less local detail.

Candidate tests:

```text
patch_size: 4
patch_size: 8
```

### Chamfer Weight

Current value:

```yaml
chamfer_weight: 0.01
```

What it does:

The Chamfer loss regularizes adaptive bin centers so they cover the empirical
height distribution.

How it affects the model:

- Too low: adaptive bins may be poorly distributed.
- Too high: the model may prioritize bin placement over pixel height accuracy.

Candidate tests:

```text
chamfer_weight: 0.001
chamfer_weight: 0.01
chamfer_weight: 0.05
```

### Fusion Mode

Current default:

```text
fusion_mode: last
```

What it does:

Fusion mode controls which decoder feature levels produce predictions and
contribute to the loss.

How it affects the model:

- Multi-level losses can improve learning by supervising intermediate decoder
  features.
- Simpler fusion can make debugging easier.

Candidate tests:

```text
fusion_mode: single
fusion_mode: last
```

Recommended first adaptive-bin tweaks:

```text
num_classes: 128
chamfer_weight: 0.01
patch_size: 4
fusion_mode: last
```

Only change one of these at a time after the baseline overfit run.

## 6. Backbone

Current backbone:

```yaml
backbone: efficientnetb0
```

Location:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
UBins.__init__()
```

What it does:

The backbone extracts image features from the PlanetScope RGB chip before the
decoder and adaptive-bin height head estimate the nDSM raster.

Current path:

```text
PlanetScope RGB chip -> EfficientNet-B0 -> decoder -> adaptive bins -> nDSM
```

How it affects the model:

- A stronger backbone can learn richer visual features.
- A larger backbone uses more memory and is more likely to overfit small data.
- A pretrained backbone can help when training data are limited, even though
  the pretraining domain is natural images rather than satellite imagery.

Available options in this vendored code:

```text
efficientnetb0
unet
```

The code is written mainly around EfficientNet-B0. Other EfficientNet sizes
may require additional compatibility checks.

Potential tests:

```text
backbone: efficientnetb0
backbone: unet
```

Interpretation:

- `efficientnetb0` is the main path and should remain the default.
- `unet` can be useful as a debugging baseline because it avoids Torch Hub and
  pretrained EfficientNet assumptions, but it is not the main paper-aligned
  model.

Recommendation:

Do not change the backbone until the `efficientnetb0` model can overfit the
10-chip sample. If it cannot, try `unet` as a diagnostic, not as the first
production model.

## 7. Input Data

Current input:

```text
single PlanetScope RGB scene per chip
```

Current tensor shape:

```text
[3, 256, 256]
```

What it does:

Input data define what information the model can use to infer height. The
model cannot learn signals that are absent from the input imagery.

Current limitation:

The first HTC dataset uses only one season and RGB bands. It does not yet use:

```text
winter + summer together
NIR
red edge
spectral indices
seasonal differences
other explanatory variables
```

How it affects the model:

- RGB captures roof color, shadows, texture, and urban context.
- NIR can help separate vegetation from built surfaces.
- Winter/summer pairs can help distinguish roofs from tree canopy and seasonal
  shadows.
- Multi-season input may improve robustness but requires model changes.

Possible input-data tweaks:

```text
Single-scene RGB, current baseline
Single-scene 4-band RGB+NIR
Two-season RGB, 6 channels
Two-season RGB+NIR, 8 channels
Two-season RGB plus seasonal differences
```

Required model changes for more than 3 channels:

The EfficientNet first convolution expects 3 channels. For 4, 6, 8, or more
channels, we must adapt the first convolution. This is documented in:

```text
data_source/source/ml_models/htc_dc_net_setup/TWO_SEASON_PLANETSCOPE_INPUT_README.md
```

Recommendation:

Do not change input channels until the single-scene RGB infrastructure is
stable. The staged sequence should be:

```text
1. Single-scene RGB overfit test
2. More single-scene RGB chips
3. Single-scene RGB validation run
4. Two-season or RGB+NIR dataset version
5. First-convolution modification
6. Two-season overfit test
```

## Recommended Tuning Order

To avoid changing too many things at once, use this order:

1. Increase epochs on the same 10 chips.
2. Test learning rate.
3. Inspect loss mask behavior.
4. Inspect height range and prediction scale.
5. Tune adaptive-bin settings.
6. Try backbone alternatives only as diagnostics.
7. Expand or redesign input data after the simple RGB pipeline is stable.

## First Experiment Grid

For the next tiny overfit experiments, use the same 10 chips and same seed:

```text
seed: 20260707
chips: 5 NYC + 5 LA
```

Suggested runs:

| Run | Epochs | Learning rate | Other changes |
|---|---:|---:|---|
| A | 20 | 0.0001 | none |
| B | 50 | 0.0001 | none |
| C | 50 | 0.0005 | none |
| D | 50 | 0.001 | none |

Success criterion:

```text
Training loss falls strongly on the same 10 chips.
Predicted nDSM maps visually move toward the target AGL maps.
Predicted mean height becomes closer to target mean height.
```

If these fail, tune mask/loss and height range before using more chips.
