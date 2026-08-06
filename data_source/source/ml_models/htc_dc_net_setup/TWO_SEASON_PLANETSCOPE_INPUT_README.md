# Two-Season PlanetScope Inputs For HTC-DC Net

This note explains how we can extend the current HTC-DC Net setup from one
PlanetScope scene per chip to two PlanetScope scenes per chip, usually winter
and summer.

This is not implemented yet. The immediate goal remains to train a simple
single-scene model first and confirm that the infrastructure works.

## Current Setup

The current NYC + LA HTC-DC Net dataset uses one PlanetScope scene per chip.

Each training observation is:

```text
image/<chip_id>_IMG.tif   3-band RGB PlanetScope chip
mask/<chip_id>_BLG.tif    binary building mask
ndsm/<chip_id>_AGL.tif    LiDAR-derived height target in meters
```

The model input tensor is:

```text
[3, 256, 256]
```

Those three channels are:

```text
red
green
blue
```

or, depending on the saved chip order, RGB as stored in the generated `_IMG`
files.

For the current combined dataset:

```text
NYC: 20200122_154449_92_1061
LA:  20231203_182937_07_2488
```

The current model therefore learns:

```text
single PlanetScope RGB chip -> LiDAR nDSM height chip
```

## Future Two-Season Setup

The two-season version would combine winter and summer imagery for the same
spatial chip.

Each observation would become:

```text
winter PlanetScope RGB
summer PlanetScope RGB
building mask
LiDAR nDSM target
```

The most direct model input tensor would be:

```text
[6, 256, 256]
```

with channels:

```text
winter_red
winter_green
winter_blue
summer_red
summer_green
summer_blue
```

If we later use 4-band PlanetScope inputs, the equivalent would be:

```text
[8, 256, 256]
```

with:

```text
winter_blue
winter_green
winter_red
winter_nir
summer_blue
summer_green
summer_red
summer_nir
```

For 8-band PlanetScope scenes, the two-season input could become:

```text
[16, 256, 256]
```

but this would only be appropriate if both cities have comparable 8-band
imagery for the chosen dates.

## Why Two Seasons Might Help

Winter and summer imagery can contain complementary information:

- Tree canopy is different across seasons, which can help separate roofs from
  vegetation.
- Shadows differ by sun angle and acquisition date.
- Roof materials may be more visible in leaf-off winter scenes.
- Seasonal differences can help identify vegetation-covered or tree-obscured
  buildings.
- A model can learn both stable roof reflectance and seasonal context.

In this project, this matters because building-height labels come from LiDAR,
but the prediction inputs come from optical imagery. If one season hides roofs
or produces strong shadows, the other season may recover useful signal.

## Required Data Changes

We would need a new dataset version, not a modification of the current
`nyc_la_rgb_v1` dataset.

Suggested folder name:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_two_season_v1/
```

The safest structure is to keep one model image file per chip, but make that
image file multi-channel:

```text
image/<chip_id>_IMG.tif   6-band winter+summer chip
mask/<chip_id>_BLG.tif
ndsm/<chip_id>_AGL.tif
```

This preserves the upstream HTC dataloader naming convention while changing
the number of channels inside `_IMG.tif`.

The chip IDs should be based on a shared spatial grid, not on only one scene.
For example:

```text
new_york_city_two_season_chip_000001
los_angeles_two_season_chip_000001
```

Each two-season chip must guarantee:

```text
winter image window == summer image window == mask window == AGL window
```

## Required Preprocessing Steps

To build the two-season dataset, we should:

1. Choose one winter and one summer PlanetScope scene per city.
2. Reproject/resample both scenes to the same CRS, transform, pixel size,
   width, height, and bounds.
3. Align both scenes to the same LiDAR nDSM raster grid.
4. Use one shared chip grid per city.
5. For each chip window, read winter RGB and summer RGB.
6. Stack those arrays into a 6-band `_IMG.tif`.
7. Write the matching `_BLG.tif` building mask chip.
8. Write the matching `_AGL.tif` LiDAR target chip.
9. Recompute `image_stats.pickle` using six channels instead of three.
10. Reuse or recompute `ndsm_stats.pickle`.
11. Create new `train.txt`, `val.txt`, `test.txt`, and `chips_manifest.csv`.

The most important alignment rule is that no chip should pair a winter window
from one place with a summer, mask, or AGL window from a slightly different
place.

## Required Dataloader Changes

The current upstream dataloader assumes image statistics have three channels:

```python
self.mean, self.std = torch.load(image_stats_file)
```

and then normalizes the loaded image:

```python
tfs.functional.normalize(img, self.mean, self.std)
```

This can work with six channels if:

```text
_IMG.tif has 6 channels
image_stats.pickle contains 6 means
image_stats.pickle contains 6 standard deviations
```

However, we should explicitly verify that `skimage.io.imread` reads the
multi-band GeoTIFF as:

```text
[height, width, channels]
```

for six bands. If it does not, we should switch the dataloader image reader to
`rasterio`, which handles multi-band GeoTIFFs more predictably.

Recommended dataloader improvement:

```text
Add config field: input_channels: 6
Validate loaded image channel count equals input_channels.
Fail loudly if image_stats length does not equal input_channels.
```

## Required Model Changes

The current released HTC-DC Net config uses EfficientNet-B0. EfficientNet-B0
expects three input channels by default.

For six-channel input, we must modify the first convolutional layer of the
backbone.

Conceptually:

```text
old first conv: input_channels = 3
new first conv: input_channels = 6
```

The rest of the network can remain mostly unchanged because after the first
layer, EfficientNet produces the same feature dimensions.

## How To Modify EfficientNet First Layer

The current model is built in:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
```

The relevant section loads the backbone:

```python
basemodel = torch.hub.load(
    'rwightman/gen-efficientnet-pytorch',
    basemodel_name,
    pretrained=True,
    trust_repo=True,
)
```

After loading the pretrained model, we would replace the first convolution.

The exact attribute name should be confirmed from the loaded model, but for
this EfficientNet implementation it is likely:

```text
basemodel.conv_stem
```

The modification would follow this pattern:

```python
def adapt_first_conv_to_input_channels(model, input_channels):
    old_conv = model.conv_stem
    if input_channels == old_conv.in_channels:
        return model

    new_conv = torch.nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        if input_channels > old_conv.in_channels:
            new_conv.weight[:, :old_conv.in_channels, :, :] = old_conv.weight
            extra = input_channels - old_conv.in_channels
            repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(
                1, extra, 1, 1
            )
            new_conv.weight[:, old_conv.in_channels:, :, :] = repeated
        else:
            new_conv.weight.copy_(old_conv.weight[:, :input_channels, :, :])

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv_stem = new_conv
    return model
```

Then, inside `UBins.__init__`, after loading `basemodel`, call:

```python
input_channels = cfgs.get("input_channels", 3)
basemodel = adapt_first_conv_to_input_channels(basemodel, input_channels)
```

and set in the config:

```yaml
input_channels: 6
```

## Alternative Model Designs

There are three reasonable ways to use two seasons.

### Option A: Early Fusion

Stack winter and summer channels into one tensor:

```text
[6, 256, 256]
```

Then modify the first EfficientNet convolution from 3 channels to 6 channels.

Advantages:

- Simplest architecture change.
- Keeps one encoder, one decoder, and one prediction head.
- Fastest path from our current implementation.

Disadvantages:

- The model must learn seasonal comparisons implicitly.
- Pretrained weights are only partly reusable in the first layer.

This is the recommended first two-season implementation.

### Option B: Derived Seasonal Difference Channels

Use RGB plus seasonal differences:

```text
winter_rgb
summer_rgb
summer_minus_winter_rgb
```

This gives:

```text
[9, 256, 256]
```

Advantages:

- Makes seasonal change explicit.
- May help identify vegetation and shadow differences.

Disadvantages:

- More channels.
- More preprocessing decisions.
- Still requires modifying the first convolution.

This should be a robustness version after Option A.

### Option C: Two Encoders

Use one encoder for winter and one encoder for summer, then fuse the feature
maps before the decoder.

Advantages:

- More expressive.
- Lets the model learn season-specific features before fusion.

Disadvantages:

- Much larger code change.
- More parameters.
- Higher overfitting risk with our current small chip count.

This is not recommended until we have a larger training set.

## Config Changes For Early Fusion

Suggested new config files:

```text
data_source/source/ml_models/htc_dc_net_setup/configs/nyc_la_rgb_two_season_v1.yaml
data_source/source/ml_models/htc_dc_net_setup/configs/htcdc_nyc_la_rgb_two_season_v1.yaml
```

Dataset config additions:

```yaml
data_dir: data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_two_season_v1
input_channels: 6
image_size: 256
use_mask: True
normalize: True
```

Model config can remain mostly the same:

```yaml
model: htcdc
backbone: efficientnetb0
batch_size: 1
optimizer: AdamW
lr: 0.0001
```

## Image Statistics

The six-channel `image_stats.pickle` should contain:

```text
mean = [
  winter_red_mean,
  winter_green_mean,
  winter_blue_mean,
  summer_red_mean,
  summer_green_mean,
  summer_blue_mean
]

std = [
  winter_red_std,
  winter_green_std,
  winter_blue_std,
  summer_red_std,
  summer_green_std,
  summer_blue_std
]
```

The dataloader should verify:

```text
len(mean) == input_channels
len(std) == input_channels
```

## Risks To Check

Before training a two-season model, we need to check:

- Both seasonal rasters are precisely aligned.
- Both rasters have the same cloud/shadow masking standard.
- The two dates are not so far apart that large building changes make the
  imagery inconsistent with the LiDAR label.
- The six-channel chip files are read correctly by the dataloader.
- The first EfficientNet convolution is actually modified and receives
  six-channel tensors.
- The training/validation split remains spatially or chip-level consistent.

## Recommended Implementation Sequence

After the single-scene model infrastructure is proven, implement the
two-season version in this order:

1. Build a new two-season chip dataset without changing model code.
2. Verify a few `_IMG`, `_BLG`, and `_AGL` chips visually.
3. Verify six-channel image statistics.
4. Add `input_channels` validation to the dataloader.
5. Add the first-convolution adaptation helper to `htcdc.py`.
6. Smoke test one batch:

```text
image shape: [1, 6, 256, 256]
gt["ndsm"] shape: [1, 1, 256, 256]
model(image, gt) succeeds
loss_total.backward() succeeds
optimizer.step() succeeds
```

7. Run the same tiny overfit test we plan for the single-scene model.

The pass/fail criterion should be the same as Step 4: the model must be able
to reduce training loss on a very small number of chips before we run a larger
training job.
