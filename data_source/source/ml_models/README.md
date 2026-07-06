# ML Models

This folder logs modeling decisions, planned experiments, and reproducible
machine-learning scripts for predicting building height from PlanetScope
imagery.

Generated model-ready tables, predictions, metrics, and trained model artifacts
should be written under:

```text
data_source/data/ml_models/generated/
```

Raw external inputs should not be stored here. The first modeling inputs come
from earlier generated project outputs:

```text
data_source/data/planet_imagery/source/<city_slug>/
data_source/data/height_labels/generated/<city_slug>/lidar_building_heights_merged_all.gpkg
data_source/data/height_labels/generated/<city_slug>/planet_aligned_lidar_rasters/
```

## Current Prediction Target

The first ML target is `height_mean_m`, which is band 2 in the
Planet-aligned LiDAR height rasters:

```text
height_mean_m = beta_0 + beta_1 PlanetScope_winter + beta_2 PlanetScope_summer + error
```

Operationally, this means predicting building-level LiDAR mean roof height
using winter and summer PlanetScope reflectance information.

## Initial Modeling Decision

Use a model ladder rather than choosing one model immediately:

1. Mean-height baseline.
2. Linear regression or ridge regression.
3. Random forest.
4. XGBoost or another gradient-boosted tree model.

The linear model makes the proposed equation transparent. Random forest is a
strong nonlinear benchmark that is simple to tune. XGBoost is the preferred
first production candidate because gradient-boosted trees usually perform well
on structured tabular features and provide useful regularization controls such
as tree depth, learning rate, row subsampling, column subsampling, and L1/L2
penalties.

The first validation design should report city-specific and pooled metrics:

```text
MAE
RMSE
bias
R squared
MAE/RMSE by height bin
```

Use spatial or grouped validation rather than naive random pixel splits.

## Why Not Train Naively On Every Pixel?

The Planet-aligned LiDAR raster has one height value burned into every pixel
covered by a building footprint. Those pixels are useful, but they are not
independent observations.

The main problems are:

1. Repeated labels within the same building. A large building may contribute
   hundreds or thousands of pixels with the same height label, while a small
   building contributes only a few pixels. A naive pixel-level model would let
   large buildings dominate the loss function.
2. Spatial autocorrelation. Neighboring pixels share similar materials,
   shadows, roof structure, atmospheric conditions, and processing artifacts.
   Randomly splitting pixels can put nearly identical nearby pixels in both
   training and test sets, making accuracy look better than it will be in new
   areas.
3. Label leakage across building boundaries. If pixels from the same building
   appear in both train and test sets, the model is partly tested on the same
   object it saw during training.
4. Misaligned scientific unit. The project goal is to predict height for every
   building footprint, not to predict height for independent anonymous pixels.

The safer first unit of analysis is therefore the building. Pixel information
can still be used, but it should be summarized within each building footprint
or sampled with a grouped split that keeps the same building out of both train
and test sets.

## Why Start With Building-Level Features?

Step 2 in the proposed plan is to create one feature row per merged building
footprint. This is useful because it keeps the target and the prediction unit
aligned:

```text
one building footprint -> one LiDAR height label -> one model prediction
```

For each building, we can summarize the winter and summer PlanetScope pixels
inside the footprint:

```text
mean reflectance
median reflectance
standard deviation
p25 and p75 reflectance
spectral indices
seasonal differences
```

This has three advantages:

1. It avoids overweighting large buildings merely because they contain more
   pixels.
2. It gives a compact tabular dataset that is easy to audit, plot, and model
   with linear regression, random forest, and XGBoost.
3. It makes validation more honest because entire buildings, or entire spatial
   groups of buildings, can be held out together.

Pixel-level or image-chip models can come later. The building-level table is
the cleanest first test of how much height signal exists in the available
PlanetScope bands.

## Candidate PlanetScope Features

The downloaded PlanetScope scenes currently differ by city/date:

| City/date group | Bands found in local TIFF metadata |
|---|---|
| LA newer 8-band scenes | `coastal_blue`, `blue`, `green_i`, `green`, `yellow`, `red`, `rededge`, `nir` |
| NYC 2020 4-band scenes | `blue`, `green`, `red`, `nir` |

For pooled NYC/LA models, the first common feature set should use only:

```text
blue
green
red
nir
```

LA-only robustness models can additionally test:

```text
coastal_blue
green_i
yellow
rededge
```

## Candidate Spectral Indices

Indices should be computed separately for winter and summer, then optionally as
seasonal differences:

```text
index_summer_minus_winter
```

### NDVI

Formula:

```text
NDVI = (NIR - red) / (NIR + red)
```

Meaning: vegetation greenness. High NDVI usually indicates vegetation; low or
near-zero values often indicate built surfaces, bare ground, or water. In this
project, NDVI can help separate roofs from trees and parks around buildings.

Available for both 4-band and 8-band PlanetScope scenes.

### GNDVI

Formula:

```text
GNDVI = (NIR - green) / (NIR + green)
```

Meaning: a vegetation index using green instead of red. It can be sensitive to
vegetation vigor and may behave differently from NDVI in urban scenes.

Available for both 4-band and 8-band PlanetScope scenes.

### NDRE

Formula:

```text
NDRE = (NIR - rededge) / (NIR + rededge)
```

Meaning: red-edge vegetation index. It is often useful for vegetation condition
when a red-edge band exists.

Available only for 8-band PlanetScope scenes, so it is currently LA-only for
the LiDAR-aligned scenes unless comparable 8-band NYC scenes are selected.

### NDBI

Canonical formula:

```text
NDBI = (SWIR - NIR) / (SWIR + NIR)
```

Meaning: built-up index designed to highlight impervious or urban built
surfaces.

Important project constraint: true NDBI requires a SWIR band. The downloaded
PlanetScope scenes do not include SWIR, so we cannot compute canonical NDBI
from PlanetScope alone. Any "NDBI-like" feature should be named honestly as a
built-up proxy, not as true NDBI.

### Visible/NIR Built-Surface Proxies

Because PlanetScope does not include SWIR, we can test simple proxy ratios such
as:

```text
NIR_red_ratio = NIR / red
visible_brightness = mean(blue, green, red)
red_minus_nir_normalized = (red - NIR) / (red + NIR)
```

Meaning: these are not standard built-up indices, but they may capture roof
material, vegetation absence, brightness, or seasonal spectral changes related
to dense built environments.

These should be treated as empirical predictors and validated carefully.

## First Implementation Plan

1. Create a building-level modeling table for LA and NYC.
2. For each merged footprint, join the LiDAR target `height_mean_m`.
3. Extract winter and summer PlanetScope zonal summaries for common bands:
   blue, green, red, and NIR.
4. Add NDVI and GNDVI for winter and summer.
5. Add seasonal differences: summer minus winter for each band and index.
6. Create spatial validation groups, such as grid cells over the AOI.
7. Train the model ladder: mean baseline, ridge, random forest, XGBoost.
8. Report MAE, RMSE, bias, R squared, and errors by height bin.
9. Decide whether to continue with building-level tabular features or move to
   image chips / CNN-style models.

## References To Read

- Internal note on the PlanetScope height-prediction workflow used by
  GlobalBuildingAtlas:
  `data_source/source/ml_models/GLOBAL_BUILDING_ATLAS_HEIGHT_MODEL.md`
- Application note for adapting HTC-DC Net to our NYC/LA PlanetScope and
  LiDAR-derived rasters:
  `data_source/source/ml_models/HTC_DC_NET_APPLICATION_README.md`
- NASA Earth Observatory, "Measuring Vegetation (NDVI and EVI)":
  https://earthobservatory.nasa.gov/features/MeasuringVegetation/
- USGS Landsat Normalized Difference Vegetation Index:
  https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index
- USGS Landsat Enhanced Vegetation Index:
  https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index
- Zha, Gao, and Ni (2003), "Use of normalized difference built-up index in
  automatically mapping urban areas from TM imagery":
  https://doi.org/10.1080/01431160304987
- Gao (1996), "NDWI: A normalized difference water index for remote sensing of
  vegetation liquid water from space":
  https://doi.org/10.1016/S0034-4257(96)00067-3
- Huete et al. (2002), "Overview of the radiometric and biophysical performance
  of the MODIS vegetation indices":
  https://doi.org/10.1016/S0034-4257(02)00096-2
