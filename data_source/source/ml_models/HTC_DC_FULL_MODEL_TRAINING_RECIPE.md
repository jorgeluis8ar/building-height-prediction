# Full HTC-DC Building-Height Model Training Recipe

## Purpose

This file is the authoritative specification for the next building-height model run. It is intended to be copied into another Codex chat and used as the implementation and training request.

The new model should follow the complete HTC-DC method used by the original HTC-DC paper and the GlobalBuildingAtlas height-estimation pipeline while retaining the NIR band available in our imagery.

Do not silently substitute parameters. If the current code cannot support a setting below, modify the code and test the modification before starting the full run.

## Final decisions at a glance

| Setting | Required value |
|---|---|
| Input channels | 4 |
| Input channel order | red, green, blue, NIR |
| Building footprint raster | Load `_BLG.tif` separately; never concatenate it with RGB+NIR |
| Image size | 256 x 256 pixels |
| Backbone | EfficientNet-B5 |
| Adaptive bins | 256 |
| Transformer patch size | 4 |
| Head-Tail Cut | Enabled |
| HTC timing | Early HTC, before hybrid regression (`earlier: true`) |
| HTC threshold | 1 m |
| Foreground distribution constraint | Gaussian |
| Background distribution constraint | Uniform |
| Fusion mode | `third` |
| Multi-level loss weights | 0.125, 0.25, 0.5, 1.0 |
| Height loss | Standard positive-height L1 |
| Custom height-bin weighting | Disabled |
| Custom background L1 | Disabled |
| Chamfer loss weight | 0.01 |
| Optimizer | AdamW |
| Learning rate | 0.0001 (1e-4) |
| Batch size | 8 |
| Batch sampling | Ordinary random shuffle |
| City/height-balanced sampler | Disabled |
| Training duration target | Maximum 50 epochs |
| Checkpoint selection | Best validation RMSE |
| Inference window | 256 x 256 |
| Inference stride | 128 pixels |
| Prediction aggregation | Mean of overlapping predictions |
| Uncertainty output | Population variance of overlapping predictions |

## 1. Input data

### Dataset must be selected before any run starts

Do not infer or silently reuse a dataset based on the run name. Before modifying code or starting a smoke test, the implementation chat must ask for or confirm the exact dataset directory with the user. Record the confirmed absolute or repository-relative path in `data_dir`, `data_split_dirs`, `test_data_split_dirs`, and run metadata.

The selected dataset must be inspected before training. Confirm and report:

- dataset path and dataset version/name;
- number of train, validation, and test chips;
- cities, scenes, dates, and spatial resolution represented;
- the exact channel count and order in sample image files;
- that every image has exactly four channels in RGB+NIR order;
- that aligned nDSM targets exist;
- that train/validation/test membership is spatially separated as intended;
- that normalization statistics contain exactly four means and four standard deviations and were computed only from the training split; and
- that aligned `_BLG.tif` building-footprint rasters exist and are loaded separately from the four-channel image.

Training must pause if the dataset has not been explicitly confirmed or fails any of these checks.

### Required model input

Each model input must contain exactly four channels in this order:

```text
red;green;blue;nir
```

The model image input must contain no building-footprint channel. It must receive tensors shaped:

```text
[batch, 4, 256, 256]
```

The current five-channel dataset and loader contain a fifth footprint channel. They must not be used unchanged. Build or select a dataset whose image rasters contain only four RGB+NIR channels, while retaining a separate aligned `_BLG.tif` for every sample. Recompute image normalization statistics from training RGB+NIR images only.

Required normalization file contents:

- four means, one for each of RGB+NIR;
- four standard deviations;
- statistics computed exclusively from training chips;
- no footprint statistics.

Load every `_BLG.tif` as a separate binary tensor in the ground-truth dictionary:

```text
image.shape = [batch, 4, 256, 256]          # RGB+NIR only
gt["ndsm"].shape = [batch, 1, 256, 256]
gt["mask"].shape = [batch, 1, 256, 256]    # _BLG.tif
```

The `_BLG` mask may be used for building-pixel and per-building validation metrics, output masking when explicitly requested, and raster-to-building postprocessing. It must not be appended to the image tensor, used as `htc_source: bf`, or used by the disabled custom background-L1 term.

The learned HTC foreground/background gate must be supervised from the nDSM threshold during training:

```text
foreground = nDSM > 1 m
background = nDSM <= 1 m
```

Use:

```yaml
use_mask: true
test_use_mask: true
```

## 2. Backbone and first convolution

Use an ImageNet-pretrained EfficientNet-B5 encoder:

```yaml
backbone: efficientnetb5
in_channels: 4
```

Adapt the pretrained three-channel stem to four channels. Initialize the additional NIR weights from the pretrained RGB weights using the repository's repeated/rescaled initialization or another explicitly documented RGB-to-NIR initialization. Do not randomly reinitialize the entire stem.

Confirm at runtime:

```text
model backbone = tf_efficientnet_b5_ap
first convolution input channels = 4
```

## 3. Complete HTC-AdaBins configuration

The model must use 256 image-adaptive height bins and transformer patch size 4:

```yaml
num_classes: 256
patch_size: 4
```

Enable the full Head-Tail Cut:

```yaml
head_tail_cut: true
earlier: true
```

The cut threshold is 1 m:

```yaml
htc_thres: 1.0
```

If the vendored training implementation does not expose `htc_thres`, verify from code that its foreground/background target is still based on the original HTC-DC rule:

```text
foreground: nDSM > 1 m
background: nDSM <= 1 m
```

The model must learn its own foreground/background gate. Do not use a footprint-driven gate such as `htc_source: bf`. If the implementation exposes `htc_source`, set:

```yaml
htc_source: pred
```

## 4. Distribution constraints

Enable the distribution-based constraints used in the full HTC-DC model:

```yaml
prob_loss: gaussian
prob_loss_bg: uniform
```

Required behavior:

- foreground bin probabilities are constrained to a Gaussian reference distribution centered on the ground-truth height;
- background bin probabilities are constrained to a uniform reference distribution as defined by HTC-DC;
- the constraint is applied through KL divergence;
- the foreground/background selection comes from the HTC gate;
- the final continuous height remains the probability-weighted mean of adaptive bin centers.

Confirm during a training smoke test that the loss dictionary contains nonzero distribution-loss terms for foreground and background. Merely placing these keys in YAML is insufficient if the code path is not executed.

## 5. Multi-level supervision

Use the full repository/paper multi-level setting:

```yaml
fusion_mode: third
```

This must produce four supervised decoder outputs with loss weights:

```text
0.125, 0.25, 0.5, 1.0
```

Confirm that four values appear in each of the following model outputs/loss families:

- intermediate nDSM predictions;
- adaptive bin edges;
- HTC probabilities;
- Chamfer losses;
- HTC cross-entropy losses; and
- foreground/background distribution losses.

## 6. Training losses

### Height loss

Use the released implementation's standard positive-height L1:

```yaml
height_loss_weighting: none
```

For each supervised scale:

```text
positive_mask = target_nDSM > 0
L_height = sum(abs(prediction - target) * positive_mask) / sum(positive_mask)
```

Disable the custom height-bin weighting. Do not apply the previous weights for `<3`, `3-6`, `6-10`, `10-25`, `25-50`, or `>=50 m`.

### Bin-edge loss

Retain the adaptive-bin Chamfer loss:

```yaml
chamfer_weight: 0.01
```

### Head-Tail Cut loss

Include binary cross-entropy supervision for the learned HTC gate using the 1 m nDSM threshold.

### Distribution losses

Include the Gaussian foreground and uniform background probability constraints described above.

### Background handling

Disable the custom background L1:

```yaml
background_loss_weight: 0.0
```

Background is to be handled through the full HTC separation and the uniform background distribution constraint, matching the HTC-DC design. Do not add an extra background penalty for this run.

## 7. Optimization

Use:

```yaml
optimizer: AdamW
lr: 0.0001
batch_size: 8
max_epochs: 50
```

The learning rate `1e-4` is the value documented by the original HTC-DC implementation/paper. Retain explicit AdamW weight decay only if recorded clearly; use `0.01` if matching the current project's historical AdamW behavior:

```yaml
weight_decay: 0.01
```

Do not use a height- or city-balanced batch sampler. Use a standard shuffled training loader:

```python
DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    ...,
)
```

Validation and test loaders must use `shuffle=False`.

Spatially separated train/validation/test split files should be retained. Random shuffle applies only within the training split; it must not mix split membership.

## 8. Augmentation

The requested parameter changes do not require removing the current augmentation. The recommended setting is:

```yaml
augmentation_profile: spatial_spectral
```

Apply spatial transforms synchronously to RGB, NIR, and nDSM. Apply spectral gain augmentation only to RGB+NIR, never to nDSM.

If strict replication of GlobalBuildingAtlas is preferred, record augmentation as an experimental difference because the atlas paper does not document its training augmentation.

## 9. Validation and model selection

At minimum, record:

- pixel RMSE and MAE over all valid pixels;
- positive-height/building-pixel RMSE and MAE;
- background RMSE and MAE;
- footprint-masked pixel RMSE and MAE;
- per-building RMSE, MAE, bias, and slope from connected `_BLG` components;
- metrics separately for NYC and LA; and
- metrics by target-height range.

Save the checkpoint with the best validation RMSE. Also report footprint-masked and per-building metrics, but do not use the test split to choose a checkpoint. Record clearly whether checkpoint selection uses all-valid-pixel RMSE or a building-focused RMSE.

Early stopping may be used only after a documented minimum number of epochs. Record the effective validation interval, patience, minimum delta, best epoch, and stop reason in run metadata.

## 10. Sliding-window inference and uncertainty

### Plain-language explanation

The model accepts only a 256 x 256 image at one time, but a city mosaic is much larger. Sliding-window inference cuts the large mosaic into 256 x 256 windows and moves the window by 128 pixels each time. Because the movement is only half the window width, neighboring windows overlap by 50%.

An interior pixel can therefore be seen in as many as four windows:

```text
top-left context      top-right context
bottom-left context   bottom-right context
```

The model predicts that same pixel once from each surrounding context. For example, its four predictions might be:

```text
18 m, 20 m, 19 m, 23 m
```

The saved height is their mean, `20 m`. The saved variance describes how much those four contextual predictions disagree. Close predictions produce low variance; widely different predictions produce high variance.

This overlap has two purposes:

1. averaging reduces seams and edge artifacts between 256 x 256 windows; and
2. variance identifies pixels whose predicted height is sensitive to where the inference window is placed.

This is performed during inference only. It does not change the training batch size and does not mean that four different satellite scenes are used. It is the same mosaic pixel evaluated through overlapping crops.

Do not infer each large raster with isolated, non-overlapping chips. Use:

```yaml
inference_window_size: 256
inference_stride: 128
```

For every window prediction, accumulate per pixel:

```text
sum_prediction += prediction
sum_squared_prediction += prediction ** 2
prediction_count += 1
```

After processing all windows:

```text
mean_height = sum_prediction / prediction_count
mean_squared = sum_squared_prediction / prediction_count
height_variance = mean_squared - mean_height ** 2
height_variance = max(height_variance, 0)
```

Save two georeferenced Float32 rasters with the same CRS, transform, width, height, and nodata mask as the input mosaic:

```text
*_height_mean.tif
*_height_variance.tif
```

With a 256-pixel window and 128-pixel stride, a regular interior pixel receives up to four contextual predictions. Border pixels may receive fewer. Store `prediction_count` during testing and verify that regular interior counts do not exceed four.

This variance measures disagreement caused by overlapping spatial contexts. It is not a calibrated confidence interval and should not be described as aleatoric or total predictive uncertainty.

For optional raster-to-building output, use the separate footprint raster after mean/variance inference. To reproduce GlobalBuildingAtlas aggregation, assign each building the maximum mean-height pixel inside its footprint and the variance from that same pixel.

## 11. Required YAML configuration

Use this as the target merged `config_used.yaml`. Paths and run name may change, but model and optimization values must not change without documenting an ablation.

```yaml
# Data
data_dir: PATH_TO_4_CHANNEL_RGB_NIR_DATASET
data_split_dirs: PATH_TO_4_CHANNEL_RGB_NIR_DATASET
test_data_split_dirs:
  - PATH_TO_4_CHANNEL_RGB_NIR_DATASET
image_size: 256
in_channels: 4
channel_order:
  - red
  - green
  - blue
  - nir
normalize: true
use_mask: true                  # separate _BLG ground truth; not an image channel
test_use_mask: true
augmentation_profile: spatial_spectral

# Model
model: htcdc
backbone: efficientnetb5
patch_size: 4
num_classes: 256
fusion_mode: third
head_tail_cut: true
earlier: true
htc_thres: 1.0
htc_source: pred
prob_loss: gaussian
prob_loss_bg: uniform

# Losses
height_loss_weighting: none
background_loss_weight: 0.0
chamfer_weight: 0.01

# Optimization
optimizer: AdamW
lr: 0.0001
weight_decay: 0.01
batch_size: 8
max_epochs: 50

# Loading
train_shuffle: true
balanced_batches: false

# Inference
inference_window_size: 256
inference_stride: 128
save_overlap_mean: true
save_overlap_variance: true
```

Some keys above are specification keys and may not yet be read by the existing code. Update the code so that each is enforced, or validate the equivalent hard-coded behavior. Do not assume that an unused YAML key changes execution.

## 12. Required implementation changes to the current project

The current spatial workflow cannot be reused unchanged. Make these changes before training:

1. Ask the user to confirm the exact four-channel dataset directory before starting.
2. Inspect and report the selected dataset, channel order, split counts, scenes, resolution, and training-only normalization statistics.
3. Remove the footprint from the image raster but retain aligned `_BLG.tif` files as separate ground-truth masks.
4. Change the spatial loader's hard-coded five-stat check to require four means/stds.
5. Remove the checks that `image[4]` equals the footprint mask; instead verify that the separately loaded `gt["mask"]` matches `_BLG.tif`.
6. Adapt EfficientNet-B5's stem to four input channels.
7. Replace `CityHeightBatchSampler` with a standard shuffled batch loader.
8. Set and verify `head_tail_cut: true` and `earlier: true`.
9. Set and verify Gaussian foreground and uniform background probability losses.
10. Set `fusion_mode: third` and verify four supervised levels.
11. Disable height-bin-weighted L1 and custom background L1.
12. Add stride-128 overlapping-window mean/variance inference.
13. Save a complete resolved `config_used.yaml`, run metadata, environment information, split files, and source-code commit/diff with the run.

## 13. Preflight acceptance tests

Do not start the 50-epoch run until the dataset has been explicitly confirmed and a one-batch smoke test confirms all of the following:

- input tensor shape is `[8, 4, 256, 256]`;
- the image's four channels are RGB+NIR and contain no footprint values;
- `_BLG.tif` is loaded separately as `gt["mask"]` with shape `[8, 1, 256, 256]`;
- EfficientNet-B5 is instantiated;
- its first convolution accepts four channels;
- the model has 256 adaptive bins;
- `head_tail_cut` is `True`;
- `earlier` is `True`;
- four decoder levels are supervised;
- HTC cross-entropy appears and is finite;
- Gaussian foreground distribution loss appears and is finite;
- uniform background distribution loss appears and is finite;
- custom height-bin weighting is off;
- custom background L1 contributes exactly zero;
- gradients are finite after backward propagation;
- an optimizer step changes trainable parameters; and
- a 256/128 sliding-window test produces at most four predictions per regular interior pixel and writes nonnegative variance.

## 14. Concise instruction for the implementation chat

> Before doing anything else, ask me to confirm the exact dataset directory. Inspect the confirmed dataset and report its split counts, cities/scenes, resolution, four-channel RGB+NIR order, aligned nDSM and `_BLG.tif` targets, and training-only four-band normalization statistics. Modify the current HTC training project to train a full HTC-DC model using exactly four-channel RGB+NIR image inputs. Retain `_BLG.tif` as a separate ground-truth mask for building-aware metrics and optional postprocessing, but never concatenate it with the image or use it as the HTC gate source. Use EfficientNet-B5, 256 adaptive bins, transformer patch size 4, `fusion_mode: third`, early Head-Tail Cut at 1 m, Gaussian foreground and uniform background distribution constraints, standard positive-height L1, Chamfer weight 0.01, no custom height weighting, and no custom background L1. Train with AdamW at 1e-4, weight decay 0.01, batch size 8, ordinary random shuffling, and a maximum of 50 epochs. Retain spatial train/validation/test separation. Add 256-window/128-stride inference that saves the overlap mean and population variance for every pixel. Run and pass all preflight acceptance tests before launching the full training job.
