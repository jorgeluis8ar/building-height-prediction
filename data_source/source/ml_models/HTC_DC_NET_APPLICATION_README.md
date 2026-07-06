# HTC-DC Net Application Notes For This Project

This note explains how the GlobalBuildingAtlas height model works, whether its
code can be applied to our NYC/LA PlanetScope and LiDAR-derived labels, and
what we still need before running it.

Repositories inspected:

- GlobalBuildingAtlas: https://github.com/zhu-xlab/GlobalBuildingAtlas
- HTC-DC-Net: https://github.com/zhu-xlab/HTC-DC-Net

Local inspection copies:

```text
/private/tmp/GlobalBuildingAtlas_inspect
/private/tmp/HTC-DC-Net_inspect
```

The parent repo was cloned locally. Its `im2bh` submodule uses an SSH GitHub
URL, so it could not be fetched through the parent clone in this environment.
The same public submodule was cloned separately over HTTPS as
`/private/tmp/HTC-DC-Net_inspect`.

## Direct Answers

### 1. LiDAR Point Clouds To nDSM

I did **not** find code in GlobalBuildingAtlas or HTC-DC-Net that converts raw
LiDAR point clouds, LAS, or LAZ files into normalized DSM rasters.

The HTC-DC-Net training code assumes the nDSM already exists as a raster named:

```text
<scene_id>_AGL.tif
```

The expected training folder is:

```text
data_dir/
  image/   optical image chips
  mask/    building mask chips
  ndsm/    ground-truth normalized DSM / AGL height chips
```

Therefore, their public code starts **after** the LiDAR-to-nDSM step. It reads
prepared nDSM rasters and uses them as supervised labels. For our project, our
equivalent label source is:

```text
data_source/data/height_labels/generated/<city_slug>/planet_aligned_lidar_rasters/
```

Specifically, if we want to train against `height_mean_m`, we would use band 2
from the Planet-aligned LiDAR rasters as our target height raster.

### 2. Polygon Simplification Algorithm

The active polygonization path in GlobalBuildingAtlas is under:

```text
GlobalBuildingAtlas/im2bf/GBA_Poly/
```

The README says binary building masks are processed through building
regularization, polygonization, and simplification with:

```bash
python tools/test.py configs/gba_poly/inference_polygonization.py --format-only
```

The active inference config uses:

```text
PolygonizerV10
PolyRegularizerV5
```

The core path is:

```text
configs/gba_poly/inference_polygonization.py
rsipoly/models/segmentors/polygonizer_v10.py
rsipoly/models/segmentors/poly_regularizer_v5.py
rsipoly/utils/tanmlh_polygon_utils.py
```

What happens:

1. A binary building mask is converted to a normalized image.
2. `PolyRegularizerV5` regularizes the mask using a ConvNeXt + UPerNet-style
   semantic segmentation network.
3. The regularized mask is converted into polygon JSONs using
   `rasterio.features.shapes`.
4. The polygon rings are simplified with:

```python
polygon_utils.simplify_poly_jsons(
    poly_jsons,
    lam=5,
    max_step_size=128,
    interval=1,
    num_min_bins=16,
    format='json',
)
```

The simplification function is a custom dynamic-programming ring simplifier in
`tanmlh_polygon_utils.py`. The relevant functions are:

```text
simplify_poly_jsons()
sample_rings_from_json()
simplify_rings_dp()
batch_decode_ring_dp()
```

There are older helper functions that call `skimage.measure.approximate_polygon`
for simpler Douglas-Peucker-style simplification, but the active
`PolygonizerV10` inference path uses the custom DP simplifier above.

### 3. Can HTC-DC Net Be Applied To Our Data?

Yes, in principle. But not directly from our current files. We need to create a
GBA/HTC-DC-compatible chip dataset first.

We currently have:

```text
PlanetScope rasters
Planet-aligned LiDAR target rasters
merged footprint GeoPackages
```

HTC-DC-Net expects:

```text
image/<chip_id>_IMG.tif
mask/<chip_id>_BLG.tif
ndsm/<chip_id>_AGL.tif
train.txt
val.txt
test.txt
image_stats.pickle
ndsm_stats.pickle
```

So the missing bridge is a script that creates aligned image, mask, and target
height chips from our current Planet and LiDAR rasters.

## What HTC-DC Net Is

HTC-DC Net is a PyTorch deep-learning model for monocular height estimation.
In this context, "monocular" means it predicts a height raster from a single
optical image view, not stereo imagery or LiDAR at inference time.

The supervised learning problem is:

```text
optical satellite image chip -> nDSM / above-ground height chip
```

The model is trained from raster chips, not from building-level rows.

## Unit Of Observation

The training observation is an image chip:

```text
one image chip
one target nDSM/height chip
optional building mask chip
```

The prediction unit is a pixel. The model outputs a continuous raster where
each pixel has a predicted height.

However, the evaluation code can also compute building-aware metrics if a
building mask is available. In testing, the code identifies connected
components in the building mask and computes median predicted and ground-truth
height for each connected building component.

So HTC-DC Net is:

```text
training unit: raster chip
prediction unit: raster pixel
optional reporting unit: building component
```

## Input Data Format

The HTC-DC-Net README specifies:

```text
data_dir/
  image/
  mask/
  ndsm/
```

Each chip must share the same basename:

```text
scene_001_IMG.tif
scene_001_BLG.tif
scene_001_AGL.tif
```

Meaning:

| File | Meaning |
|---|---|
| `_IMG.tif` | Optical satellite image chip |
| `_BLG.tif` | Building footprint mask chip |
| `_AGL.tif` | Above-ground-level nDSM / height target chip |

The split files contain chip basenames without suffixes:

```text
train.txt
val.txt
test.txt
```

Example:

```text
nyc_20200122_chip_000001
nyc_20200122_chip_000002
la_20231203_chip_000001
```

## Model Architecture

The main config uses:

```text
model: htcdc
backbone: efficientnetb0
optimizer: AdamW
learning rate: 0.0001
batch size: 32
max epochs: 500
patience: 20
patch_size: 4
```

The model combines:

1. An EfficientNet-B0 encoder loaded through `torch.hub`.
2. A decoder that upsamples feature maps back toward image resolution.
3. Adaptive height bins, similar in spirit to AdaBins monocular depth models.
4. Multi-level outputs, where intermediate and final predictions contribute to
   the loss.
5. Optional head-tail-cut logic for distinguishing near-zero/background pixels
   from positive-height pixels.

The final prediction is:

```text
predicted nDSM / height raster
```

## Loss Function

The model trains only on positive target-height pixels:

```python
mask = gt["ndsm"] > 0
```

The main prediction loss is masked MAE:

```text
MAE(predicted_height, true_height) over positive-height pixels
```

It also uses a Chamfer loss on adaptive height-bin centers:

```text
loss_total = masked_MAE + chamfer_weight * bin_chamfer + optional auxiliary losses
```

The default config sets:

```text
chamfer_weight: 0.01
```

## Training

Training command from the repo:

```bash
python train.py --config configs/configs1.yaml --exp_config configs/htcdc.yaml
```

Training does the following:

1. Reads config files.
2. Creates a timestamped checkpoint directory.
3. Saves the merged config as `config.yaml`.
4. Initializes Weights & Biases logging.
5. Computes or reads image and nDSM statistics.
6. Loads image/height/mask chips from `train.txt` and `val.txt`.
7. Trains the model.
8. Validates each epoch.
9. Saves checkpoints.

Important output checkpoints:

```text
checkpoint_last.pth.tar
checkpoint_best_mae.pth.tar
checkpoint_best_rmse.pth.tar
checkpoint_best_loss_total.pth.tar
```

## Validation And Testing

Validation monitors:

```text
MAE
RMSE
validation loss
height-bin MAE/RMSE variants
```

Testing can use:

```bash
python test.py --config /path/to/saved/config test_checkpoint_file checkpoint_best_rmse.pth.tar
```

If a building mask exists, the test code reports:

```text
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

For our project, `mae_per_building` and `rmse_per_building` are especially
important because our labels come from building footprints and our scientific
object is building height.

## Inference

GlobalBuildingAtlas inference is in:

```text
GlobalBuildingAtlas/infer_height/
```

The inference script:

1. Loads `config.yaml`.
2. Builds the model.
3. Loads `checkpoint_best_rmse.pth.tar`.
4. Reads a Planet mosaic TIFF.
5. Normalizes the image with stored training means and standard deviations.
6. Runs sliding-window inference.
7. Averages overlapping window predictions.
8. Writes a predicted height GeoTIFF.
9. Writes a variance GeoTIFF as an uncertainty proxy.

Default sliding-window settings in the code:

```text
window size: 256 x 256
stride: 64
window batch size: 64
```

Outputs:

```text
*_ss.tif   predicted height surface
*_var.tif  prediction variance from overlapping windows
```

The output GeoTIFF copies the input Planet raster's geotransform and
projection.

## Important Imagery Constraint

The public `infer_height/utils.py` inference path reads only the first three
bands:

```python
img = planet_infer_readTiff(filename)[:3]
```

Then, if `rgb=False`, it reverses channel order before normalization.

This means the released inference path is configured for a 3-channel image
input. It does not directly use all 4 or 8 PlanetScope bands.

For our data, this matters:

| City/scenes | Available local Planet bands |
|---|---|
| NYC 2020 LiDAR-aligned scenes | 4 bands: blue, green, red, nir |
| LA 2023 LiDAR-aligned scenes | 8 bands: coastal_blue, blue, green_i, green, yellow, red, rededge, nir |

To use HTC-DC Net exactly as released, we should first create 3-band image
chips. To use NIR or all available PlanetScope bands, we would need to modify
the model input layer, normalization statistics, and inference code.

## What We Need To Make It Run On NYC/LA

We need to create a chip-preparation workflow:

1. Select one winter and one summer Planet scene per city.
2. Decide whether HTC-DC Net will use:

```text
single-season 3-band RGB chips
single-season 4-band chips
two-season stacked chips
```

The easiest faithful first run is single-season 3-band chips.

3. Create `_IMG.tif` chips from Planet rasters.
4. Create `_AGL.tif` chips from band 2, `height_mean_m`, of our
   Planet-aligned LiDAR rasters.
5. Create `_BLG.tif` building mask chips from merged footprints.
6. Set non-building target pixels to 0 or nodata consistently with the
   HTC-DC loss, which uses `gt["ndsm"] > 0` as the valid-height mask.
7. Generate `train.txt`, `val.txt`, and `test.txt` using spatial splits.
8. Compute `image_stats.pickle`.
9. Compute `ndsm_stats.pickle`.
10. Adjust config paths.
11. Train with `train.py`.
12. Test with `test.py`.
13. Run sliding-window inference over a full Planet scene.

## Recommended Project Adaptation

I recommend adapting HTC-DC Net in two phases.

### Phase 1: Faithful Minimal Reproduction

Use the model as close to the public code as possible:

```text
input: 3-band Planet image chip
target: height_mean_m chip
mask: building footprint mask chip
```

This gets us a working raster-height neural baseline fastest.

### Phase 2: Project-Specific Extension

After the minimal run works, extend the model/input to test:

```text
4-band RGB+NIR input
8-band LA-only input
winter/summer stacked input
building-mask-aware loss
height_p90_m or height_p95_m targets
```

This is where the model would become more tailored to our research design.

## Differences From The Building-Level ML Plan

The earlier Random Forest / XGBoost approach uses:

```text
one building -> summarized Planet features -> one height prediction
```

HTC-DC Net uses:

```text
one image chip -> height raster chip
```

The final output goal is a raster, so HTC-DC Net is more aligned with the final
output format. But it is also more complex and requires a GPU-ready chip
dataset. The building-level model remains useful as a simpler baseline and as
a diagnostic check on whether Planet imagery contains enough height signal.

## Files Most Relevant For Us

GlobalBuildingAtlas:

```text
README.md
infer_height/README.md
infer_height/main.py
infer_height/utils.py
im2bf/README.md
im2bf/GBA_Poly/configs/gba_poly/inference_polygonization.py
im2bf/GBA_Poly/rsipoly/models/segmentors/polygonizer_v10.py
im2bf/GBA_Poly/rsipoly/models/segmentors/poly_regularizer_v5.py
im2bf/GBA_Poly/rsipoly/utils/tanmlh_polygon_utils.py
```

HTC-DC-Net:

```text
README.md
configs/configs1.yaml
configs/htcdc.yaml
dataloaders.py
train.py
test.py
build.py
htcdc.py
basenets.py
utils.py
```

## Bottom Line

The public GBA/HTC-DC code can guide our raster model, but it is not plug and
play yet. The next concrete task is not model training itself. It is creating
a GBA-style NYC/LA chip dataset from our already aligned Planet rasters,
LiDAR-derived height rasters, and merged building footprints.
