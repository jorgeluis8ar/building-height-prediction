# HTC-DC Net Post-Training Diagnostics

This file defines the standard diagnostics package to run after every HTC-DC
Net training run. The goal is to make model comparisons consistent across
experiments.

## Primary Wrapper

Use:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_post_training_diagnostics.py \
  --run-dir data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/<run_name> \
  --dataset-dir data_source/data/ml_models/generated/htc_dc_net/<dataset_name> \
  --epoch <epoch_number> \
  --device cpu
```

The wrapper writes outputs under:

```text
<run_dir>/evaluation/
```

By default, it runs diagnostics for:

```text
train
validation
test
all
```

It expects the checkpoint:

```text
<run_dir>/model_epoch_<epoch_number>.pth
```

Use `--checkpoint` if the model file has a different name.

## Required Diagnostics

### 1. Prediction Exports

For each split, export unmasked chip-level predictions:

```text
<split>_predictions_epoch_XXX_unmasked/
<split>_predictions_summary_epoch_XXX_unmasked.csv
<split>_prediction_collapse_check_epoch_XXX_unmasked.csv
```

Unmasked predictions are useful because they let us inspect model behavior
outside building footprints. This is important for detecting artifacts such as
high predicted values over water, roads, or background areas.

### 2. City Mosaics

For the `all` split, create both city-level mosaics:

```text
mosaics_all_epoch_XXX_unmasked_3m/
mosaics_all_epoch_XXX_masked_3m/
```

The unmasked mosaic shows the full model surface over the AOI. The masked
mosaic keeps only pixels inside the building footprint mask and is the main
raster for building-height interpretation.

### 3. Building-Level Scatter Plots

For each split, summarize raster pixels to connected building-mask components
and compare:

```text
x-axis: LiDAR-derived building height
y-axis: predicted building height
```

Outputs:

```text
building_component_predictions.csv
building_scatter_metrics.csv
building_height_scatter_all.png
building_height_scatter_los_angeles.png
building_height_scatter_new_york_city.png
building_height_scatter_three_panel_<split>_epoch_XXX.png
```

The scatter plots must report:

```text
best-fit line
RMSE
R squared
number of building components
```

These plots are the main diagnostic for whether the model is predicting height
variation or collapsing toward an average value.

### 4. Height-Bin RMSE Bars

For each split, bin buildings by LiDAR-derived height:

```text
0-10 m
10-20 m
20-30 m
30-40 m
>40 m
```

Output:

```text
height_bin_rmse_<split>_epoch_XXX.png
height_bin_error_summary_<split>_epoch_XXX.csv
```

This shows where the model performs well or poorly across the height
distribution. It is especially useful for checking low-rise and high-rise
performance.

### 5. Height-Bin Bias Bars

Output:

```text
height_bin_bias_<split>_epoch_XXX.png
```

Bias is:

```text
predicted height - LiDAR height
```

Positive values mean overprediction. Negative values mean underprediction.

### 6. Residual Boxplots

Output:

```text
height_bin_residual_boxplot_<split>_epoch_XXX.png
```

The residual definition must appear in the plot:

```text
Residual = Predicted Height - LiDAR height
```

This plot shows the distribution of errors inside each height bin, not only
the average RMSE. It helps identify whether a bin has symmetric noise,
systematic bias, or a few large outliers.

### 7. Full-City Three-Panel Plots

For each city, create:

```text
PlanetScope scene
Prediction (building mask)
Target AGL
```

Output:

```text
full_city_three_panel_epoch_XXX/<city>_three_panel_planetscope_prediction_target_epoch_XXX.png
```

The prediction and target panels must use the same color scale and the same
height units in meters.

## Model-Selection Metric

The main model-selection metric remains:

```text
validation-chip RMSE
```

Building-level scatter and height-bin diagnostics are secondary but important
for interpretation. In particular, a model with a slightly better average RMSE
may still be less useful if it fails badly for low-rise or high-rise
buildings.

## Standard Example

For the off-nadir 3-channel epoch-50 model:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_post_training_diagnostics.py \
  --run-dir data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc76_la95_offnadir_3ch_lowrise_binweighted_bg005_seed20260720_epoch50_guarded \
  --dataset-dir data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_v1 \
  --epoch 50 \
  --device cpu
```

## Optional Flags

Use existing predictions without exporting again:

```bash
--skip-export
```

Skip city mosaics:

```bash
--skip-mosaics
```

Skip full-city three-panel plots:

```bash
--skip-full-city-panels
```

Run only one split:

```bash
--splits val
```

Change height bins:

```bash
--height-bin-edges 0,10,20,30,40,50,100
```

