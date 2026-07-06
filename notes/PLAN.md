# Six-Step Implementation Plan For HTC-DC Net

## Summary

We will implement HTC-DC Net as a staged workflow: first verify our generated image/mask/AGL chips, then create one clean combined NYC+LA dataset, then adapt the HTC-DC Net configuration, run a smoke test, train the first model, and finally evaluate plus produce full-scene height predictions. The first model will use **single-season 3-band RGB PlanetScope input** and **building-only LiDAR nDSM `_AGL.tif` targets**.

## 1. Visual QA The HTC Inputs

- Create a QA script/notebook that samples chips from NYC and LA and plots three aligned panels: `_IMG.tif`, `_BLG.tif`, `_AGL.tif`.
- Confirm that buildings in `_BLG.tif` overlap positive heights in `_AGL.tif`.
- Flag chips with empty masks, implausible AGL values, bad alignment, nodata artifacts, or obvious Planet/LiDAR mismatch.
- Use the New Jersey LiDAR variant only as a diagnostic, not in the first training dataset.

Success criteria: at least 20 random chips per city visually pass alignment and target sanity checks.

## 2. Build One Combined HTC Dataset

- Create a combined dataset under `data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/`.
- Copy or link NYC and LA chip triplets into unified folders:
  - `image/`
  - `mask/`
  - `ndsm/`
- Use only:
  - NYC New York LiDAR chips: 103 chips
  - LA chips: 142 chips
- Exclude `new_york_city_new_jersey_lidar` from v1 training.
- Recompute combined `train.txt`, `val.txt`, `test.txt`, `all.txt`, `chips_manifest.csv`, and `image_stats.pickle`.
- Add `ndsm_stats.pickle` if the HTC dataloader requires it.

Success criteria: every basename in the split files has exactly one `_IMG.tif`, one `_BLG.tif`, and one `_AGL.tif`.

## 3. Prepare The HTC-DC Net Code Environment

- Clone or place HTC-DC Net code in a documented external/code folder, not inside raw data.
- Create a reproducible environment file for the model run.
- Start with the released 3-channel RGB configuration.
- Set dataset paths to the combined NYC+LA dataset.
- Disable or stub Weights & Biases if it blocks local smoke tests.
- Confirm whether the dataloader expects `image_stats.pickle`, `ndsm_stats.pickle`, or config-defined normalization values.

Success criteria: Python can import the HTC model, dataloader, and config without crashing.

## 4. Run A Tiny Smoke Test

- Run the HTC dataloader on 2-4 chips.
- Check tensor shapes:
  - image: 3 channels
  - mask: 1 channel or expected mask shape
  - nDSM/AGL: 1 target channel
- Run one forward pass.
- Run one backward pass on a tiny batch.
- Save one predicted chip for inspection.

Success criteria: one train iteration completes and produces finite loss and finite predicted heights.

## 5. Train The First HTC-DC Net Model

- Train the first model on the combined NYC+LA RGB dataset.
- Use the public HTC defaults unless the smoke test reveals incompatibilities:
  - EfficientNet-B0 backbone
  - AdamW
  - learning rate `0.0001`
  - chip size `256`
  - masked positive-height loss
- Save checkpoints and config under `data_source/data/ml_models/generated/htc_dc_net/runs/`.
- Track training/validation MAE, RMSE, loss, and best checkpoint.

Success criteria: training completes at least one stable run and saves `checkpoint_best_rmse.pth.tar` or equivalent.

## 6. Evaluate And Run Full-Scene Prediction

- Evaluate on the held-out test chips.
- Report:
  - MAE
  - RMSE
  - bias
  - masked building-pixel MAE/RMSE
  - building-component MAE/RMSE if supported
- Run sliding-window inference on the full NYC and LA Planet scenes.
- Save predicted height rasters and uncertainty/variance rasters under `data_source/data/ml_models/generated/htc_dc_net/predictions/`.
- Compare predicted full-scene rasters against LiDAR `_AGL.tif` targets and summarize errors by city and height bin.

Success criteria: full-scene prediction rasters are georeferenced, aligned to the Planet scene grids, and produce interpretable error metrics.

## Assumptions

- First model uses RGB only, not NIR or 8-band LA-specific inputs.
- First model includes NYC and LA only; New Jersey LiDAR variant is excluded from training.
- First target is building-only nDSM from `_AGL.tif`, equivalent to `ndsm_buildings_only_m`.
- Current chip count is small, so this is a feasibility baseline, not the final production model.
- If HTC-DC Net’s public dataloader has strict pickle/config expectations, we adapt our dataset packaging rather than changing the target definition.
