# Adding a Water Index to HTC-DC Net

## Why Add NDWI?

The current HTC-DC Net input uses PlanetScope image bands directly. In some chips,
especially near rivers or harbors, the raw model can predict large heights over
water. These water predictions do not enter building-level diagnostics when we
mask by building footprints, but they can contaminate raw mosaics and make visual
QA harder to interpret.

Adding a water index gives the model an explicit feature that separates water
from built surfaces. The goal is not to predict water height, but to help the
network learn that water/non-building areas should have near-zero height.

## Candidate Index

The standard normalized difference water index is:

```text
NDWI = (Green - NIR) / (Green + NIR)
```

For PlanetScope SuperDove-style 8-band imagery, this can usually be computed
with the green and near-infrared bands. For 4-band PlanetScope imagery, the same
logic applies if green and NIR are present.

Interpretation:

```text
higher NDWI: more water-like
lower NDWI: less water-like, often vegetation, bare ground, or built surface
```

## How to Add It to Our Data

For the two-season model, we currently stack winter and summer PlanetScope chips.
The clean extension is:

```text
winter image bands
summer image bands
winter NDWI
summer NDWI
```

If the current model uses 6 channels, adding one NDWI per season would make this
an 8-channel input. If we later use all 8 PlanetScope bands per season, adding
NDWI per season would make the input 18 channels.

## Required Code Changes

1. Update the dataset builder to compute NDWI per chip.
2. Append the NDWI rasters as extra channels in `_IMG.tif`.
3. Recompute `image_stats.pickle` and `stats/image_stats.pickle`.
4. Set `--in-channels` to the new channel count when training.
5. Keep the target `_AGL.tif` and building mask `_BLG.tif` unchanged.

## Model Changes

HTC-DC Net already supports variable input channels through the `in_channels`
configuration we added earlier. The important requirement is that the first
convolution receives the same number of channels as the image chips.

Example:

```text
6-channel current model: --in-channels 6
6 channels + 2 NDWI channels: --in-channels 8
```

## Recommended Order

First use building-masked exports and the weak background penalty. If raw water
predictions remain high, create an NDWI dataset version:

```text
nyc_la_8ch_ndwi_v1
```

Then train the same low-rise bin-weighted model on that dataset and compare:

```text
validation building RMSE
validation building R2
raw water prediction maxima
masked prediction mosaics
visual QA panels near rivers/harbors
```

