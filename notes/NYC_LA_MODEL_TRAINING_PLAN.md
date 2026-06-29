# NYC and Los Angeles Building-Height Model Training Plan

Last updated: 2026-06-24

## Purpose

This note translates the current project status, the two reviewed papers, and
the inspected public repositories into a practical plan for training a first
building-height model for New York City and Los Angeles.

The near-term goal is:

```text
predict height_m for every building footprint in New York City and Los Angeles
using satellite imagery features, with later extensions to other covariates
```

The project should keep the main unit of observation as:

```text
city x building
```

Raster products should be treated as model inputs, labels, masks, or
intermediate products, not as the canonical research dataset.

## Evidence From Reviewed Work

### UT-GLOBUS

Sources reviewed:

- Paper: `s41597-024-03719-w.pdf`
- Repository: `https://github.com/texuslabut/UT-GLOBUS`

Relevant findings:

- Data structure: building-level vector output. The repository writes a
  city-level GeoPackage with building polygons and height attributes.
- Model: Random Forest regression. The code imports
  `sklearn.ensemble.RandomForestRegressor` and ships a trained
  `UT-GLOBUS.joblib` model.
- Predictors: spaceborne nDSM from ALOS, ICESat-2, GEDI, WSF3D adjustment,
  population density, footprint area, and footprint perimeter.
- Training labels: LiDAR-derived nDSM building heights assigned to footprint
  polygons.
- Training split: roughly 268,000 buildings from six U.S. cities, with 80%
  randomly selected for fitting and 20% held out for validation; tuning used
  3-fold cross-validation.
- Validation metrics: RMSE, mean bias error, R-squared, and spatial
  correlation for gridded urban canopy products.
- Important limitation for this project: UT-GLOBUS is designed mainly for
  urban canopy and weather-model inputs, not precise building-level height
  prediction. It uses coarse global predictors and reports weaker performance
  for tall buildings.

Takeaway:

UT-GLOBUS supports using building polygons as the canonical unit and shows a
simple tabular baseline path, but its predictor set is not the right first
choice if this project wants a satellite-imagery-first model.

### Microsoft TEMPO

Sources reviewed:

- Paper: `2511.12104v1-2.pdf`
- Repository: `https://github.com/microsoft/buildings`

Relevant findings:

- Data structure: raster-first. Planet quads are paired with 512 by 512 target
  rasters. Public outputs are Cloud-Optimized GeoTIFFs with two bands:
  density and height.
- Model: neural network. The released training configuration uses a modified
  U-Net with an ImageNet-pretrained EfficientNet-B6 backbone.
- Inputs: PlanetScope RGB basemap imagery plus an Overture building-density
  prior as a fourth channel.
- Labels: Google Open Buildings 2.5D weak labels, processed into density and
  height rasters aligned to Planet quads.
- Homogenization: crop to Planet quad footprint, merge overlaps, reproject to
  EPSG:3857, resample to a 512 by 512 grid, normalize bands, and save aligned
  COGs.
- Sampling: image quads are sampled with weights proportional to building
  density plus a small constant, so built-up areas are selected more often but
  empty areas are not ignored.
- Loss and training: Huber loss with hard sigmoid bounded outputs, two heads
  for density and height, AdamW optimizer, 512-pixel imagery patches, and
  64-pixel target patches after 8x downsampling.
- Validation metrics: MAE for positive-reference pixels, F1/precision/recall
  after thresholding, macro-F1 over height bins, accuracy, R-squared, and
  temporal consistency metrics.

Takeaway:

TEMPO is useful as a reference for raster alignment, image-chip training,
masking, and neural-network design, but its output is too coarse for this
project's desired per-building footprint predictions.

## Recommended Project Design

Use a hybrid design:

1. Keep the master training and prediction table as building-level vectors.
2. Derive LiDAR-based building labels from point clouds and footprints.
3. Extract raster imagery chips and raster masks around each building.
4. Train a tabular baseline first.
5. Train an image-plus-mask model only after labels and baseline diagnostics are
   stable.

This preserves the project's research unit while borrowing the strongest
engineering pieces from TEMPO.

## Target Building-Level Schema

The first model-ready table should include:

```text
building_id
city
geometry
footprint_area_m2
height_m
height_definition
height_source
confidence_tier
usable_for_training
usable_for_validation
planet_scene_id
planet_asset_type
planet_acquired
sun_azimuth
sun_elevation
image_chip_path
footprint_mask_path
neighbor_mask_path
split_id
```

The model should never use official NYC `HEIGHT_ROO` or LA `HEIGHT` to create
LiDAR labels. Those fields should be used only for validation and diagnostics.

## Immediate Technical Order

### Step 1: Finish independent height labels

Create:

```text
data_source/source/height_labels/derive_lidar_building_heights.py
```

Start with the diagnostic sample recommended in
`data_source/source/height_labels/LIDAR_BUILDING_HEIGHT_PROCESS.md`:

```text
about 500 buildings from New York City
about 500 buildings from Los Angeles
```

Outputs should be written under:

```text
data_source/data/height_labels/generated/<city_slug>/
```

Required diagnostic outputs:

```text
lidar_tile_inventory.csv
building_height_diagnostics_sample.csv
height_definition_comparison.csv
quality_tier_summary.csv
```

Minimum contents of `building_height_diagnostics_sample.csv`:

```text
building_id
city
height_p50_m
height_p75_m
height_p90_m
height_p95_m
height_max_clean_m
ground_elevation_m
roof_pixel_count
roof_coverage_fraction
ground_point_count
ground_ring_coverage
quality_tier
official_height_m
official_height_source
official_height_units_confirmed
temporal_mismatch_flag
```

Primary first label definition:

```text
height_definition = lidar_ndsm_roof_p90_minus_local_ground
height_m = height_p90_m
```

Do not finalize this until p50, p75, p90, p95, and max-clean are compared
against the independent official fields.

### Step 2: Build the first model training index

Create a training-index task after the diagnostic labels are stable:

```text
data_source/source/build_training_index/
data_source/data/training_index/generated/
```

The first training index should join:

- clipped footprints;
- selected height labels;
- selected Planet scene metadata;
- Planet image paths;
- quality tiers;
- train/validation/test split assignments.

The index should exclude rejected labels and include a switch for whether
tier C labels are included.

### Step 3: Create building-centered imagery chips

Create a separate task:

```text
data_source/source/build_training_masks/
data_source/data/training_masks/generated/<city_slug>/
```

For each building, create:

- a fixed-size image chip centered on the footprint;
- a binary footprint mask;
- an optional neighboring-footprints mask;
- a context ring mask for shadows and surrounding texture.

Recommended first chip sizes:

```text
64 m x 64 m for small and medium buildings
128 m x 128 m for large or tall-candidate buildings
```

Start with a single standardized size if model implementation speed matters.

Recommended imagery inputs for the first pass:

- Planet RGB bands;
- near-infrared if using 8-band SuperDove assets;
- red edge if available and consistent across both scenes;
- per-scene sun azimuth and sun elevation as tabular features;
- UDM/cloud flags for exclusion or quality flags.

Use the same band order and normalization for NYC and LA.

### Step 4: Train a tabular baseline

Create:

```text
data_source/source/train_height_models/train_tabular_baseline.py
```

Recommended first model:

```text
LightGBM or XGBoost
```

If dependency simplicity is more important than speed, use scikit-learn
HistGradientBoostingRegressor first.

Feature groups:

- footprint geometry: area, perimeter, compactness, elongation, orientation;
- spectral summary inside footprint: band mean, standard deviation,
  percentiles;
- context summary in ring buffer: band means, texture, built density;
- shadow indicators: dark-pixel share, shadow direction relative to sun
  azimuth, contrast between footprint and shadow-side context;
- scene metadata: acquisition date, season, sun elevation, sun azimuth;
- city indicator for the within-city baseline, then remove or stress-test it
  for cross-city transfer.

Baseline outputs:

```text
data_source/data/trained_models/generated/tabular_baseline/
data_source/data/validation/generated/tabular_baseline/
```

### Step 5: Train a building-chip image model

Only start after the tabular baseline has produced credible diagnostics.

Recommended first architecture:

- CNN encoder over image chip plus footprint mask;
- small MLP over tabular features;
- concatenated latent representation;
- regression head for height;
- optional quantile heads for p10, p50, p90 uncertainty.

Do not begin with a full TEMPO-style global U-Net unless the target changes to
a gridded product. The project target is per-building height, so the first
image model should predict one height per building.

### Step 6: Evaluate with spatial splits

Avoid random building splits as the primary evidence. Use:

- within-NYC spatial holdout;
- within-LA spatial holdout;
- train NYC, test LA;
- train LA, test NYC;
- train pooled NYC+LA, test spatial holdout from both cities.

Report:

```text
MAE
RMSE
median_absolute_error
bias
R_squared
coverage by confidence tier
error by height bin
error by footprint area bin
error by neighborhood/spatial fold
```

Height bins:

```text
0-10 m
10-30 m
30-55 m
55-100 m
100+ m
```

## Upcoming Weeks

### Week 1: LiDAR diagnostic labels

Tasks:

- Implement the diagnostic LiDAR label script for NYC and LA.
- Inventory only manifest-approved LAZ files.
- Sample about 500 diverse buildings per city.
- Compute p50, p75, p90, p95, and max-clean roof-height candidates.
- Compare candidates against NYC `HEIGHT_ROO` and LA `HEIGHT`.
- Produce quality-tier summaries.

Exit criteria:

- The sample output exists for both cities.
- Official height units are confirmed or explicitly flagged unresolved.
- One primary height definition is recommended with evidence.

### Week 2: Full label run for NYC and LA

Tasks:

- Scale the accepted LiDAR label method to all 5 km AOI buildings for NYC and
  LA.
- Save building-level label GeoPackages and lightweight CSV or Parquet tables.
- Produce maps or summaries of rejected buildings and quality tiers.

Exit criteria:

- Every footprint has either a usable label, a lower-tier label, or a documented
  reject reason.
- Training-eligible buildings are clearly marked.

### Week 3: Planet imagery and chip/mask pipeline

Tasks:

- Confirm Planet orders/downloads for the selected NYC and LA scenes.
- Build the first chip/mask script.
- Generate chips for a small labeled sample first, then all usable labels.
- Store scene metadata and chip paths in a training index.

Exit criteria:

- A model-ready index links each usable building to a label, footprint, image
  chip, mask, and split.
- A small sample of chips is visually audited for alignment.

### Week 4: Tabular baseline

Tasks:

- Extract spectral and geometry features.
- Train the first gradient-boosted regression baseline.
- Evaluate spatial holdouts and city-transfer tests.
- Inspect residuals by height bin, footprint size, quality tier, and city.

Exit criteria:

- Baseline metrics exist for NYC, LA, pooled, and cross-city tests.
- Residual diagnostics identify the main failure modes.

### Week 5: Image model prototype

Tasks:

- Train a small building-chip CNN model using image bands plus footprint mask.
- Compare against the tabular baseline on the same splits.
- Add tabular features to the image model if the image-only model underperforms.

Exit criteria:

- The image model beats or complements the tabular baseline on at least one
  meaningful validation setting, or its failure mode is documented.

### Week 6: Model selection and inference preparation

Tasks:

- Select the best first production model.
- Calibrate prediction intervals or quantile heads if available.
- Run inference for every NYC and LA footprint in the 5 km AOIs.
- Prepare validation tables and maps.

Exit criteria:

- Every building has a prediction or a documented no-prediction reason.
- Validation outputs are reproducible from the training index and model
  artifacts.

## Model Sequence

Recommended order:

1. LiDAR label diagnostic.
2. Full LiDAR label generation.
3. Tabular baseline from footprint and imagery summaries.
4. Building-chip CNN.
5. Multimodal CNN plus tabular model.
6. Quantile or ensemble model for uncertainty.

Do not train the neural network before the label diagnostic and tabular
baseline are complete. The baseline will reveal label problems, unit problems,
and spatial leakage much faster than the image model.

## Main Risks

- Official height fields may be in feet but must be confirmed from metadata.
- NYC LiDAR is 2013-2014 while footprints may include later construction.
- LA LiDAR is 2023-2024 while footprint vintages may be older.
- Planet scene date mismatch can make shadows and roof signals inconsistent
  with labels.
- Random splits will overstate performance.
- Tall buildings will likely have larger errors; report them separately.
- Trees and roof equipment can contaminate footprint-masked LiDAR heights.

## Next Concrete Action

Start with:

```text
data_source/source/height_labels/derive_lidar_building_heights.py
```

and implement only the diagnostic sample first. The model pipeline should not
advance to chip generation until the label definition and quality tiers are
credible for both New York City and Los Angeles.
