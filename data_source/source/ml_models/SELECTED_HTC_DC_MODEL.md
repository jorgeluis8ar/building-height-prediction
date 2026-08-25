# Selected HTC-DC Net Model

## Selection Status

The primary model selected for continued development, evaluation, and
cross-computer reproduction is:

```text
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded
```

Do not substitute the 12-channel model or either EfficientNet-B5 full-recipe
experiment when reproducing the selected result. Those runs remain documented
as historical experiments.

## Why This Model Was Selected

The model was selected using held-out validation performance, not training
error or architectural complexity. Among the directly comparable off-nadir
models, it has the lowest pooled validation RMSE, highest validation R2, and
best NYC validation RMSE.

| Model | Validation RMSE m | Test RMSE m | Validation R2 | NYC validation RMSE m |
|---|---:|---:|---:|---:|
| **Selected RGB+NIR B0** | **7.699** | 8.091 | **0.555** | **18.183** |
| RGB+mask B0 | 7.871 | **7.981** | 0.532 | 18.590 |
| Full B5, height-weighted | 8.413 | 8.597 | 0.470 | 19.708 |
| Full B5, unweighted | 8.711 | 8.126 | 0.437 | 21.158 |
| 12-channel B0 | 11.920 | 10.250 | lower | worse overall |

The RGB+mask B0 model remains the preferred robustness comparison because its
test RMSE is 0.11 m lower and its final collapse check is cleaner. It is not
the primary model because model selection was based on validation results and
the RGB+NIR model performs better on NYC, the more difficult city.

## Exact Configuration

| Parameter | Selected value |
|---|---|
| Dataset | `nyc_la_off_nadir_rgb_nir_v1` |
| Training chips | 76 NYC + 95 LA |
| Input channels | 4 |
| Channel order | Red, green, blue, NIR |
| PlanetScope inputs | One off-nadir RGB+NIR scene per city |
| Model | HTC-DC Net |
| Backbone | EfficientNet-B0 |
| Epochs | 50 |
| Learning rate | 0.00003 |
| Batch size | 8 |
| Optimizer | AdamW |
| Chamfer weight | 0.01 |
| Height-loss weighting | Target-height bins |
| Height-bin edges | `3,6,10,25,50` m |
| Height-bin weights | `4,3,2,1,3,8` |
| Background-loss weight | 0.05 |
| Patience | 20 |
| Prediction exports | Every 10 epochs |
| Checkpoint exports | Every 10 epochs |
| Collapse threshold | Prediction standard deviation below 0.05 m |
| Collapse trigger share | 0.8 |
| Collapse patience | 1 check |
| Stop on collapse | Enabled |
| Sampling | Random, without replacement within each city |
| Seed | `20260721` |
| Selected checkpoint | `model_epoch_050.pth` |

## Canonical Paths On The Original Computer

Dataset:

```text
data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1/
```

Run directory:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded/
```

Checkpoint:

```text
data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded/
model_epoch_050.pth
```

Checkpoint transfer verification:

```text
size_bytes = 209586693
sha256 = f6f43953905a5abf209bc8501cf8cc5af070256989f37b6f637878bc9c946331
```

After copying the checkpoint to another computer, calculate its SHA-256 hash
and require an exact match before using it for inference or diagnostics.

Recorded configuration and metadata:

```text
config_used.yaml
run_metadata.json
```

Primary diagnostics:

```text
evaluation/post_training_diagnostics_summary_epoch_050.csv
evaluation/val_building_scatter_epoch_050/
evaluation/test_building_scatter_epoch_050/
evaluation/val_height_bin_error_epoch_050/
evaluation/test_height_bin_error_epoch_050/
```

## Important GitHub Limitation

The repository `.gitignore` excludes all of `data_source/data/`. Therefore,
GitHub contains the reproducible source code and documentation but does not
contain:

- the model-ready RGB+NIR dataset;
- PlanetScope imagery;
- LiDAR nDSM chips;
- `model_epoch_050.pth`;
- prediction rasters, mosaics, or diagnostic CSV files.

A Git clone on another computer is not enough to run inference from the saved
model. Transfer the selected checkpoint and required generated data separately,
or rebuild the dataset and retrain the model using the commands below. Do not
force large or licensed data into Git without first reviewing licensing and
repository-size constraints.

## Cross-Computer Reproduction

### Option A: Use The Existing Trained Model

Transfer these items outside GitHub while preserving their relative paths:

1. `model_epoch_050.pth`.
2. `config_used.yaml` and `run_metadata.json`.
3. `nyc_la_off_nadir_rgb_nir_v1/stats/` and its split/manifest files.
4. The model input imagery or complete model-ready dataset needed for the
   intended inference task.
5. Any PlanetScope source imagery required to rebuild city mosaics.

After transfer, verify that the checkpoint, dataset, channel count, channel
order, image statistics, CRS, transform, and pixel resolution match this file.

### Portable Inference Bundle

`create_selected_model_inference_bundle.py` packages the selected epoch-50
checkpoint, exact RGB+NIR normalization statistics, model configuration,
manifest, Windows CPU requirements, and GeoTIFF predictor under:

```text
data_source/data/ml_models/generated/htc_dc_net/selected_model_inference_bundle_v1/
```

The bundle is generated data and is not committed to Git because the checkpoint
is approximately 200 MB. Transfer the complete bundle between computers using
Dropbox or another approved large-file channel. The destination computer still
needs the repository's tracked `external/HTC-DC-Net` model source.

### Option B: Rebuild And Retrain

First create the off-nadir RGB base dataset and then build the RGB+NIR variant:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/build_htc_dataset_single_scene_variants.py \
  --base-dataset-dir \
    data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_v1 \
  --output-dir \
    data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1 \
  --variant rgb_nir
```

Run the selected training configuration:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --dataset-dir \
    data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1 \
  --in-channels 4 \
  --nyc-chips 76 \
  --la-chips 95 \
  --epochs 50 \
  --seed 20260721 \
  --lr 0.00003 \
  --batch-size 8 \
  --patience 20 \
  --save-predictions-every 10 \
  --save-checkpoints-every 10 \
  --height-loss-weighting bins \
  --height-bin-edges 3,6,10,25,50 \
  --height-bin-weights 4,3,2,1,3,8 \
  --background-loss-weight 0.05 \
  --collapse-std-threshold 0.05 \
  --collapse-min-share 0.8 \
  --collapse-patience 1 \
  --stop-on-collapse \
  --run-name \
    nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded
```

Do not pass `--overwrite` unless replacing an existing run is intentional.

## Verification Checklist

Before accepting a reproduction, confirm:

- dataset has 244 total chips: 171 train, 36 validation, and 37 test;
- training set has exactly 76 NYC and 95 LA chips;
- every image has four bands ordered RGB+NIR;
- image, building mask, and nDSM grids align exactly;
- model uses EfficientNet-B0 and four input channels;
- height-bin edges and weights match the table above;
- seed is `20260721`;
- collapse guard is enabled;
- evaluation uses building-component median heights;
- validation RMSE is reported before consulting test performance.

Small metric differences may occur across hardware and software versions even
with the same seed. Any difference in dataset membership, normalization,
channel order, or scene inputs constitutes a different experiment.
