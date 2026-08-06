# ML Models

## Selected Primary Model

The selected model for continued work is:

```text
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded
```

It uses one off-nadir PlanetScope RGB+NIR scene per city, four input channels,
HTC-DC Net with EfficientNet-B0, and target-height bin weighting. It was chosen
because it has the strongest pooled validation RMSE, validation R2, and NYC
validation RMSE among the directly comparable completed models.

See `SELECTED_HTC_DC_MODEL.md` before training or inference. That file is the
canonical source for the model name, parameters, checkpoint, paths, rerun
command, and cross-computer transfer checklist. The generated dataset and
checkpoint are Git-ignored and are not delivered by cloning GitHub.

For a parameter-by-parameter explanation of the completed weighted Full
HTC-DC RGB+NIR model, including expected RMSE effects and tuning guidance, see
`HTC_DC_MODEL_PARAMETER_GUIDE.md`.

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

## Off-Nadir 4-Channel Variants

On 2026-07-21, two independent 4-channel off-nadir HTC-DC Net variants were
created and trained with the same parameters as the off-nadir RGB baseline:

```text
epochs = 50
learning_rate = 0.00003
batch_size = 8
optimizer = AdamW
backbone = EfficientNet-B0
height_loss_weighting = bins
height_bin_edges = 3,6,10,25,50
height_bin_weights = 4,3,2,1,3,8
background_loss_weight = 0.05
collapse_guard = on
prediction_exports = every 10 epochs
seed = 20260721
```

The two variants are intentionally independent:

| Variant | Dataset | Input channels | Run folder |
|---|---|---:|---|
| RGB + footprint mask | `nyc_la_off_nadir_rgb_mask_v1` | 4 | `nyc76_la95_offnadir_rgbmask_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded` |
| RGB + NIR | `nyc_la_off_nadir_rgb_nir_v1` | 4 | `nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded` |

The RGB + footprint-mask dataset appends the binary building mask as the
fourth image input channel. The RGB + NIR dataset appends the PlanetScope NIR
band as the fourth input channel after reprojecting it to each RGB chip grid.
Both datasets preserve the same train/validation/test split as
`nyc_la_off_nadir_rgb_v1`.

Building-level validation diagnostics at epoch 50:

| Variant | Validation RMSE (m) | Validation R2 | Validation bias (m) |
|---|---:|---:|---:|
| RGB + footprint mask | 7.87 | 0.532 | -0.33 |
| RGB + NIR | 7.70 | 0.555 | -0.76 |

The comparison table is saved at:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/offnadir_4ch_variant_comparison_epoch_050.csv
```

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

## HTC-DC Net RGB Dataset v1

The first combined HTC-DC Net dataset is:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/
```

It combines only the true NYC and LA chip datasets:

| Source city | Chips |
|---|---:|
| New York City | 103 |
| Los Angeles | 142 |
| Total | 245 |

It excludes the NYC/New Jersey Sandy LiDAR diagnostic variant. The dataset
contains:

```text
image/   *_IMG.tif
mask/    *_BLG.tif
ndsm/    *_AGL.tif
train.txt
val.txt
test.txt
all.txt
chips_manifest.csv
stats/image_stats.pickle
stats/ndsm_stats.pickle
image_stats.pickle
ndsm_stats.pickle
```

The split seed is `20260706`, with 171 training chips, 37 validation chips, and
37 test chips. Build or refresh it with:

```bash
data_source/source/height_labels/venv_height_labels/bin/python \
  data_source/source/ml_models/combine_htc_datasets.py \
  --overwrite

data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/htc_dc_net_setup/prepare_htc_dataset_stats.py
```

The current combined dataset was rebuilt after applying the LiDAR target rule
that finite nonpositive AGL pixels inside the building mask are set to `2.4`
meters. Post-rebuild verification found zero valid pixels with `AGL <= 0`
inside the building mask for both NYC and LA chips.

## HTC-DC Net Environment v1

The HTC-DC Net implementation is vendored under:

```text
data_source/source/ml_models/external/HTC-DC-Net/
```

Project-specific config and setup notes are under:

```text
data_source/source/ml_models/htc_dc_net_setup/
```

The current smoke-test environment is:

```text
data_source/source/ml_models/venv_htc_dc_net/
```

The released `efficientnetb0` HTC-DC Net config now imports, builds, loads the
combined NYC+LA dataset, and completes one training-style forward/backward
optimizer step. Use `WANDB_MODE=offline` for local runs and set
`TORCH_HOME=/private/tmp/torch_htc_cache` if the EfficientNet Torch Hub cache
needs to be reused.

## HTC-DC Net Mini Training Run

The first mini training/estimation runner is:

```text
data_source/source/ml_models/run_htc_mini_training.py
```

It selects a reproducible subset of training chips, creates a temporary
mini-dataset, trains the real HTC-DC Net `UBins` model, and exports predicted
nDSM rasters plus chip-level metrics.

Predicted nDSM rasters are written as georeferenced GeoTIFFs. Each prediction
copies the CRS, affine transform, width, height, and resolution from its
matching `_AGL.tif` target chip, so LA predictions remain in `EPSG:32611` and
NYC predictions remain in `EPSG:32618`.

The first completed run used:

```text
NYC chips: 5
LA chips: 5
Epochs: 1
Seed: 20260707
Run name: nyc5_la5_seed20260707_epoch1
```

The run was regenerated after the 2.4 m zero-height target update. Current
one-epoch mini-run diagnostics:

```text
Mean training loss: 77.667369
LA mean MAE: 25.3578 m
NYC mean MAE: 24.7513 m
Mini-run target pixels with valid AGL <= 0 inside building mask: 0
```

The first 50-epoch overfit-style mini run used the same 5 NYC and 5 LA chips,
seed `20260707`, and `lr = 0.001`. It saved checkpoints and predictions every
10 epochs:

```text
Run name: nyc5_la5_seed20260707_epoch50_lr001
Output: data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch50_lr001/
Epochs: 50
Learning rate: 0.001
Prediction/checkpoint interval: 10 epochs
```

Key results:

```text
Epoch 1 mean loss: 57.0792
Best mean loss: 33.2129 at epoch 42
Epoch 50 mean loss: 33.3211
Epoch 50 LA mean MAE: 9.5385 m
Epoch 50 NYC mean MAE: 27.8521 m
```

The run writes:

```text
training_history.csv
training_epoch_loss.csv
training_loss.png
model_epoch_010.pth ... model_epoch_050.pth
predictions_epoch_010/ ... predictions_epoch_050/
predictions_summary_epoch_010.csv ... predictions_summary_epoch_050.csv
model_last.pth
predictions/
predictions_summary.csv
```

The first 40-chip mini run used 20 NYC and 20 LA random training chips with
the same seed and hyperparameters:

```text
Run name: nyc20_la20_seed20260707_epoch50_lr001
Output: data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc20_la20_seed20260707_epoch50_lr001/
NYC chips: 20
LA chips: 20
Epochs: 50
Learning rate: 0.001
Prediction/checkpoint interval: 10 epochs
```

Key results:

```text
Epoch 1 mean loss: 31.5639
Epoch 50 mean loss: 24.3134
Best mean loss: 24.3134 at epoch 50
Epoch 50 LA mean MAE: 4.5515 m
Epoch 50 NYC mean MAE: 22.9391 m
```

Final prediction verification:

```text
Prediction rasters: 40
LA predictions: 20 in EPSG:32611
NYC predictions: 20 in EPSG:32618
Prediction CRS/transform/size matched target chips: yes
```

Important diagnostic finding:

```text
The two 50-epoch mini-runs produced constant prediction rasters.
The 1-epoch smoke run produced spatially varying predictions, so the raster
export path works; the longer mini-runs collapsed during training.
```

The mini-training runner now includes collapse guardrails. Every prediction
summary CSV records:

```text
pred_min_m
pred_max_m
pred_std_m
pred_unique_values
pred_raster_min_m
pred_raster_max_m
pred_raster_std_m
pred_raster_unique_values
target_min_m
target_max_m
target_std_m
target_unique_values
collapse_std_threshold_m
collapse_flag
```

The runner also writes:

```text
prediction_collapse_checks.csv
prediction_collapse_final_check.csv
```

Recommended guarded calibration command before any larger run:

```bash
WANDB_MODE=offline TORCH_HOME=/private/tmp/torch_htc_cache \
  data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --nyc-chips 20 \
  --la-chips 20 \
  --epochs 50 \
  --lr 0.0001 \
  --seed 20260707 \
  --save-predictions-every 5 \
  --collapse-std-threshold 0.05 \
  --collapse-min-share 0.8 \
  --collapse-patience 1 \
  --stop-on-collapse \
  --run-name nyc20_la20_seed20260707_epoch50_lr0001_guarded
```

The one-epoch baseline outputs are stored locally under:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch1/
```

Important files:

```text
model_last.pth
training_history.csv
predictions_summary.csv
predictions/*_ndsm_pred.tif
mini_dataset/
```

Rerun command:

```bash
WANDB_MODE=offline TORCH_HOME=/private/tmp/torch_htc_cache \
  data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --nyc-chips 5 \
  --la-chips 5 \
  --epochs 1 \
  --seed 20260707 \
  --run-name nyc5_la5_seed20260707_epoch1 \
  --overwrite
```

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

## HTC-DC Net Multi-Scene Dataset Builder

The multi-scene dataset builder is:

```text
data_source/source/ml_models/build_htc_dataset_multiscene.py
```

It starts from the existing `nyc_la_rgb_v1` chips and stacks RGB channels from
additional PlanetScope scenes onto each chip's existing grid.  The first target
version is:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1/
```

This version uses four scenes per city:

```text
4 scenes x 3 RGB bands = 12 input channels
```

The scene list comes from:

```text
data_source/data/planet_imagery/generated/intermediate_sun_elevation_scene_review.csv
```

Channel order is fixed by city and recorded in `scene_channel_plan.csv`,
`chips_manifest.csv`, and the GeoTIFF band descriptions:

```text
1. base RGB chip scene used by nyc_la_rgb_v1
2. the other existing LiDAR-aligned scene used by nyc_la_6ch_v1
3. first new intermediate scene
4. second new intermediate scene
```

The builder should be run only after the new Planet orders have reached
`success` and have been downloaded:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/build_htc_dataset_multiscene.py \
  --scene-count 4 \
  --output-dir data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1 \
  --overwrite
```

The training runner now validates image channels before training.  The dataset
band count and `image_stats.pickle` mean/std lengths must match
`--in-channels`, so valid invocations include:

```bash
# RGB baseline
--dataset-dir data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1 --in-channels 3

# Two-scene baseline
--dataset-dir data_source/data/ml_models/generated/htc_dc_net/nyc_la_6ch_v1 --in-channels 6

# Four-scene sun-elevation experiment
--dataset-dir data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1 --in-channels 12
```

## HTC-DC Net Cross-Validation

The first HTC-DC Net parameter search uses 3-fold cross-validation on the
`nyc_la_12ch_v1` dataset. The objective metric is mean chip-level RMSE on
validation chips.

Use `build_htc_cv_folds.py` to create fold-specific HTC dataset folders under:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1/cv_folds/
```

The fold builder uses the original `train.txt` plus `val.txt` as the CV pool,
keeps `test.txt` untouched, and stratifies folds by city and LiDAR AGL height
distribution. The image, mask, and nDSM folders are symlinked back to the parent
dataset to avoid copying large GeoTIFF chips.

Use `run_htc_cross_validation.py` to run the staged historical 12-channel grid.
The grid keeps that experiment's 12-channel baseline settings fixed except for
learning rate, background loss weight, and low-rise/high-rise bin weights. It
does not define the currently selected primary model. Outputs are written
under:

```text
data_source/data/ml_models/generated/htc_dc_net/cross_validation/nyc_la_12ch_v1/
```

Expected summary outputs:

```text
cv_results_by_fold.csv
cv_results_by_config.csv
cv_ranked_configs.csv
cv_metric_trends.png
best_config.yaml
```

Smoke-test command:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_cross_validation.py \
  --configs baseline_current \
  --folds fold_01 \
  --epochs 1 \
  --save-every 1 \
  --overwrite
```

Full staged run command:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_cross_validation.py \
  --configs all \
  --folds all \
  --overwrite
```

## Off-Nadir Model Scatter Comparison

Use `plot_offnadir_model_scatter_comparison.py` to compare the final
building-level predictions from the RGB, RGB plus footprint mask, and RGB plus
NIR models on the training, validation, and test samples. The script uses the
same axes, 1:1 reference line, and regression diagnostics in every panel. It
also produces a 0-50 m view so that the dense low-rise and mid-rise portion of
the distribution remains visible.

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/plot_offnadir_model_scatter_comparison.py
```

Outputs are written to:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/
  offnadir_model_scatter_comparison_epoch_050/
```

The regression metrics use all eligible building components. The figures may
display a reproducible sample of points to limit overplotting; this does not
change the reported RMSE, MAE, bias, slope, or R2.

## Full HTC-DC Paper Recipe

The full four-channel paper-recipe workflow is implemented by:

```text
prepare_htc_full_recipe_dataset.py
preflight_htc_full_recipe.py
run_htc_full_recipe_training.py
predict_htc_sliding_window_city.py
plot_htc_full_recipe_training.py
```

It uses RGB+NIR image inputs, a separately loaded footprint mask,
EfficientNet-B5, 256 adaptive bins, four supervised decoder levels, early HTC,
Gaussian foreground and uniform background constraints, standard
positive-height L1, and 256/128 overlapping inference. The preflight must pass
before training and verifies training-only normalization, alignment, finite
losses/gradients, an actual optimizer update, and overlap mean/variance logic.

The completed run is:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/
  nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_seed20260723_epoch50/
```

Epoch 40 was selected by city-balanced validation building RMSE. See the run's
`README.md` for parameters, diagnostics, and limitations.

The same workflow now supports `--height-loss-weighting bins`,
`--height-bin-edges`, and `--height-bin-weights`. The completed weighted Full
B5 experiment is:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/
  nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_binweighted_seed20260723_epoch50/
```

It uses edges `3,6,10,25,50` m and weights `4,3,2,1,3,8`. Epoch 15 was selected
at 11.6229 m city-balanced validation building RMSE, versus 11.9659 m for the
otherwise comparable unweighted run. The run README records all split and city
diagnostics and the remaining high-rise compression.

## LiDAR Height Split Distribution

Use `plot_lidar_height_split_distribution.py` to audit whether the training,
validation, and test chips contain comparable LiDAR-derived building-height
distributions. The diagnostic uses the same building-component median heights
as the building-level scatter workflow and reports results overall and by city.

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/plot_lidar_height_split_distribution.py
```

Outputs include normalized histograms, empirical cumulative distributions,
height-bin shares, split summary statistics, and train-to-validation/test
distribution distances. This audit should be run whenever a new HTC dataset
split is created.
