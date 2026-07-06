# GlobalBuildingAtlas Height-Prediction Workflow

This note summarizes the height-prediction code in
[`zhu-xlab/GlobalBuildingAtlas`](https://github.com/zhu-xlab/GlobalBuildingAtlas)
and its height-estimation submodule
[`zhu-xlab/HTC-DC-Net`](https://github.com/zhu-xlab/HTC-DC-Net). The goal is
to record what their PlanetScope-based building-height workflow does and what
parts are useful for this project.

Inspection date: 2026-07-02.

## High-Level Finding

GlobalBuildingAtlas uses a **raster/image model**, not a tabular Random Forest
or XGBoost model. The model predicts an **nDSM/height map** from optical
satellite imagery. Training uses image chips with matching height rasters and,
optionally, building masks. Inference runs the trained model over large
PlanetScope mosaics with a sliding window and writes predicted height rasters.

The parent repository separates height work into two parts:

1. `im2bh`: monocular height estimation with HTC-DC Net.
2. `infer_height`: global inference and uncertainty quantification over Planet
   imagery.

## Model Used

The height model is **HTC-DC Net**, implemented in the `HTC-DC-Net` repository.
It is a PyTorch deep-learning model for monocular height estimation from
single-view remote-sensing images.

The default configuration uses:

```text
model: htcdc
backbone: efficientnetb0
optimizer: AdamW
learning rate: 0.0001
max epochs: 500
batch size: 32
early stopping metrics: mae, rmse, val/loss_total
```

Architecturally, the code combines:

1. An EfficientNet-B0 encoder, loaded through `torch.hub`.
2. A decoder that upsamples image features.
3. Adaptive height bins, inspired by AdaBins-style monocular depth estimation.
4. Optional head-tail-cut logic for separating near-zero/background pixels from
   height-bearing foreground pixels.
5. A loss combining masked MAE and a Chamfer loss on predicted height bins.

The repo also contains baseline U-Net/AdaBins-style inference paths in
`GlobalBuildingAtlas/infer_height/baselines/`, but the main training submodule
points to `model: htcdc`.

## Programming Language And Libraries

The workflow is Python.

Important libraries found in the code and environment:

```text
PyTorch
torchvision
PyTorch3D
timm / torch.hub EfficientNet
GDAL
rasterio
OpenCV / cv2
NumPy
scikit-image
PyYAML
Weights & Biases / wandb
tqdm
```

The `HTC-DC-Net` README lists recommended versions including PyTorch 1.7.1,
PyTorch3D 0.4.0, `fvcore`, `timm`, `scikit-image`, and `wandb`.

## Data Structure

The training repo expects a raster-chip dataset organized as:

```text
data_dir/
  image/   optical satellite image chips
  mask/    building footprint masks, optional but used for building metrics
  ndsm/    ground-truth normalized DSM / height maps
```

Each scene/chip has the same basename and three file suffixes:

```text
scene_001_IMG.tif  optical image
scene_001_BLG.tif  building mask
scene_001_AGL.tif  nDSM / above-ground-level height raster
```

Train/validation/test split files list the basenames:

```text
data_split_dir/
  train.txt
  val.txt
  test.txt
```

The data loader reads the optical image and target nDSM rasters as arrays. It
also reads the building mask when requested. The default chip size used by the
model path is 256 x 256 pixels, and the config includes `patch_size: 4` for the
transformer/adaptive-bin module.

## Unit Of Observation

There are two units:

1. **Training unit:** raster image chip.
2. **Prediction unit:** raster pixel, later summarized to buildings for some
   evaluation metrics.

The model learns from aligned image chips and height-map chips:

```text
image chip -> predicted nDSM chip
```

This is different from our current first plan of:

```text
building footprint -> PlanetScope zonal features -> one height label
```

However, their evaluation code explicitly bridges back to buildings when a
building mask is available. During testing, it identifies connected components
in the building mask, computes the median predicted and ground-truth height for
each component, and reports per-building metrics.

## Imagery Inputs

The inference utility reads a Planet TIFF with GDAL and takes only the first
three bands:

```python
img = planet_infer_readTiff(filename)[:3]
```

If `rgb=False`, it reverses those channels before normalization. The code then
normalizes imagery using stored training-set means and standard deviations from:

```text
data/gbh/image_stats.pickle
```

This means the released inference path appears to use a 3-channel optical input
for the deployed Planet inference script, not all available PlanetScope bands.
For our project, this is important: GlobalBuildingAtlas demonstrates that a
height raster can be predicted from Planet imagery, but it does not imply we
must restrict ourselves to RGB. Since our downloaded LA scenes include 8 bands
and NYC 2020 scenes include 4 bands, our first tabular model can still use the
common `blue`, `green`, `red`, and `nir` bands.

## Training Setup

Training is launched with:

```bash
python train.py --config configs/configs1.yaml --exp_config configs/htcdc.yaml
```

The script:

1. Loads a data/logging config and a model/training config.
2. Creates a timestamped experiment directory under the checkpoint folder.
3. Saves the merged config as `config.yaml`.
4. Initializes Weights & Biases logging.
5. Fixes the random seed.
6. Builds train and validation data loaders from `train.txt` and `val.txt`.
7. Builds the HTC-DC model and optimizer.
8. Trains for up to 500 epochs.
9. Evaluates on validation data each epoch.
10. Saves checkpoints, including:

```text
checkpoint_last.pth.tar
checkpoint_best_mae.pth.tar
checkpoint_best_rmse.pth.tar
checkpoint_best_loss_total.pth.tar
```

Early stopping is configured with patience 20 and monitors MAE, RMSE, and
validation loss.

## Loss Function

The core loss is computed only over positive target-height pixels:

```text
mask = gt_ndsm > 0
masked MAE(pred_ndsm, gt_ndsm)
```

HTC-DC also predicts adaptive height-bin edges and adds a Chamfer loss between
the predicted bin centers and target height values. The default Chamfer weight
in the config is:

```text
chamfer_weight: 0.01
```

The final loss is a weighted sum across multi-level predictions. For the
default fusion mode, the weights are:

```text
0.25, 0.5, 1.0
```

## Validation And Testing

Validation during training reports:

```text
MAE
RMSE
height-bin MAE/RMSE variants
validation loss
```

Testing is launched with:

```bash
python test.py --config /path/to/saved/config test_checkpoint_file checkpoint_best_rmse.pth.tar
```

The test config can include multiple test split directories, for example:

```text
data/MEL+
data/LOS+
data/SAO+
data/MUC+
data/GUA+
data/split1+
data/split2+
```

When test masks are available, the code reports both raster-level and
building-aware metrics:

```text
mae
rmse
mae_mask
rmse_mask
mae_non_mask
rmse_non_mask
mae_building
rmse_building
mae_per_building
rmse_per_building
per_building_bin_rmse
```

The most relevant design lesson for us is that even a raster model benefits
from building-level evaluation. The final output can be a raster, while
diagnostics can still be reported per building.

## Prediction / Inference Setup

The `infer_height` folder performs GPU inference over Planet mosaics.

The inference script:

1. Loads `config.yaml` from a trained model folder.
2. Instantiates the model.
3. Loads `checkpoint_best_rmse.pth.tar`.
4. Opens the Planet mosaic TIFF.
5. Normalizes the image using saved mean/std statistics.
6. Runs sliding-window inference with:

```text
window size: 256 x 256
stride: 64
batch size for windows: 64
```

7. Adds overlapping predictions into an accumulator.
8. Divides by the number of predictions per pixel to average overlaps.
9. Computes a variance estimate from overlapping predictions:

```text
variance = mean(prediction^2) - mean(prediction)^2
```

10. Writes two GeoTIFFs:

```text
*_ss.tif   predicted height surface
*_var.tif  prediction variance / uncertainty proxy
```

The output GeoTIFF copies the input Planet mosaic geotransform and projection.

## What This Means For Our Project

GlobalBuildingAtlas supports a two-stage strategy for us:

1. **First stage: building-level tabular ML.**
   Create a clean `city x merged_building_id` feature table from Planet bands,
   train ridge/random forest/XGBoost, evaluate spatially, and burn predictions
   back into a Planet-aligned raster.

2. **Second stage: raster/image-chip model.**
   If the building-level model is not strong enough, move toward a GBA-style
   image-chip model:

   ```text
   Planet image chip -> height_mean_m raster chip
   ```

   Our existing `planet_aligned_lidar_rasters` already gives us the target
   height rasters needed for this. We would still need building masks and
   train/val/test chip split files.

The main caution is that GBA's deployed inference path uses only 3 image
channels, while our planned tabular approach can exploit all common
PlanetScope bands and seasonal information. Therefore, GBA is most useful as a
template for raster-chip modeling and raster inference, not as a reason to skip
the building-level model.

## Recommended Adaptation For NYC/LA

For this project, do not jump directly to HTC-DC Net. The better path is:

1. Build the building-level Planet feature table.
2. Train and validate ridge/random forest/XGBoost with spatial splits.
3. Rasterize predicted building heights back to the Planet grid.
4. In parallel, prepare the ingredients for a future chip model:

```text
image/<chip_id>_IMG.tif
mask/<chip_id>_BLG.tif
ndsm/<chip_id>_AGL.tif
train.txt / val.txt / test.txt
```

5. Use GBA's raster-chip workflow as the blueprint only if the tabular
building-level model underperforms or if we decide the final contribution
requires a true pixel/raster neural model.

## Source Files Inspected

Parent repo:

- `README.md`
- `infer_height/README.md`
- `infer_height/main.py`
- `infer_height/utils.py`

Height-model submodule:

- `README.md`
- `configs/configs1.yaml`
- `configs/htcdc.yaml`
- `dataloaders.py`
- `train.py`
- `test.py`
- `build.py`
- `htcdc.py`
- `basenets.py`
- `utils.py`

## Links

- GlobalBuildingAtlas repository:
  https://github.com/zhu-xlab/GlobalBuildingAtlas
- HTC-DC-Net repository:
  https://github.com/zhu-xlab/HTC-DC-Net
- GlobalBuildingAtlas paper:
  https://essd.copernicus.org/articles/17/6647/2025/
- HTC-DC Net paper linked from the submodule:
  https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10294289
