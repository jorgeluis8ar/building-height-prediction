# Project Progress

Last updated: 2026-08-19

## Overall Plan

1. Initialize the repository from `README.md` and `claude.md`.
2. Document city AOIs, candidate CBD coordinates, and data-source coverage.
3. Acquire or document source datasets for footprints, labels, imagery,
   elevation, and benchmark products.
4. Harmonize source-specific footprint and height labels into the common
   building-level schema.
5. Build training masks, imagery indexes, and model-ready feature tables.
6. Train baseline and multimodal height-prediction models.
7. Run inference, validate spatially, benchmark external products, and produce
   research outputs.

## Complete

- Added safe Windows-compatible global training order/download scripts. The
  order workflow creates one deterministic AOI-clipped plan per training city,
  grouping 8-band and 4-band bundles into 711 requests for 6,350 available
  scenes, and requires explicit bounded confirmation before contacting Planet.
  The downloader accepts only successful orders, blocks partial fulfillment,
  checks free disk space, verifies every delivered Planet manifest file, and
  checkpoints each city. Both reject validation/testing rows, recover safely
  after interruptions, and write honest dated logs. Only local dry-run and
  simulated completeness tests have been executed; no Planet order was created
  and no imagery was downloaded.
- Added `create_planet_global_city_splits.py` for a stable city-level split of
  all 1,779 cities represented in the global selected-scenes file. Seed
  `419453` produces exactly 711 training, 711 validation, and 357 testing
  cities using SHA-256 ordering, with every scene inheriting its city's group.
  The script writes full and compact scene manifests plus a training order
  input, performs no Planet API actions, and explicitly flags the 26 cities
  with fewer than nine selected scenes.
- Added `analyze_planet_global_scene_selection.py` for an API-free analysis of
  the completed global scene selection. It creates a world map of all WUP AOI
  centroids and selected-scene centroids, country city/scene counts, numeric
  and categorical metadata summaries, an acquisition-year summary, and
  detailed AOI/scene-footprint maps for Aba, Tokyo, and Buenos Aires. The
  analysis explicitly retains sample cities with no selected scenes and writes
  dated, fail-loud logs under the generated analysis folder.
- Added `select_planet_global_city_scenes.py` for deterministic selection of
  up to nine PlanetScope scenes per WUP global city. The workflow distinguishes
  Northern and Southern Hemisphere solstice seasons, maximizes distinct years
  before flagged repeated-year fallbacks, balances four scene-centroid
  directions and a five/four seasonal target, applies explicit cloud/coverage
  tiers, maximizes sun-elevation diversity, prefers high view angles, and
  requires RGB+NIR surface reflectance with 8-band preference and 4-band
  fallback. Asset-list requests are cached and metadata-only; the script
  cannot activate, order, or download imagery.
- Offline regression-tested the global selector with the 186-row Aba, Nigeria
  metadata file. The test selected nine unique standard-quality solstice
  scenes, preserved seven distinct years before two repeated-year fallbacks,
  and honestly flagged infeasible season/direction targets and the intentionally
  skipped offline asset verification. Production runs require live asset
  verification through the saved Planet OAuth session.

- Defined the global city sample from the WUP 2018 workbook using the strict
  rule `POP2018 > 300` thousand. Added
  `extract_wup2018_cities_over_300k.py` and generated
  `wup2018_cities_over_300k_2018.csv` with 1,862 unique urban agglomerations.
  The smallest selected population is 300,097 and the largest is 37,468,302.
- Extended `create_city_buffers.py` with `--global-wup-cities`. Generated a
  separate combined 5km AOI and 1,862 city-specific AOIs without overwriting
  the completed 29-city AOI pipeline. Global slugs include the unique WUP
  urban code to prevent same-name city collisions.
- Added `search_planet_global_city_scenes.py` for resumable, metadata-only
  Planet PSScene discovery across the global AOIs. It uses bounded city
  batches and city/year checkpoints, applies permission, cloud-cover, and AOI
  coverage filters, retains complete raw item metadata and footprint
  coordinates, and writes honest dated logs. Offline syntax, CLI, inventory,
  and AOI validations passed. The Planet API search itself has not been run.

- Read `README.md` for the project scope, data components, methods, validation
  strategy, and expected research outputs.
- Read `claude.md` for the required folder structure, naming conventions,
  logging rules, failure behavior, progress tracking, Git rules, and data
  handling rules.
- Removed previous folders, data scaffolding, scripts, and old progress notes.
- Preserved the root `README.md`, root `claude.md`, and hidden `.git/`
  repository metadata.
- Created the current base folder structure declared in `claude.md`.
- Created `.gitignore` to prevent datasets and common cache files from being
  committed.
- Generated WUP 2025 city centroid CSV files for city AOI work:
  - `data_source/data/city_aois/generated/wup2025_all_city_centroids.csv`
    contains all 16,828 cities from `WUP2025-F21-DEGURBA-Cities_Pop.xlsx`.
  - `data_source/data/city_aois/generated/wup2025_city_centroids.csv`
    contains the 32 cities listed in `city_selection.csv`.
  - `data_source/data/city_aois/generated/wup2025_city_centroids_unmatched.csv`
    is currently header-only, meaning every selected city matched a WUP row.
- Debugged `data_source/source/city_aois/create_city_buffers.py` so it can
  create 5km GeoJSON buffer polygons from `cities_sample.csv`.
- Rewrote `data_source/data/city_aois/generated/cities_sample.csv` using the
  27 cities listed under `Current Cities` in `README.md`, matched to
  `WUP2025-F21-DEGURBA-Cities_Pop.xlsx` coordinates.
- Updated `claude.md` to include city-specific 5km AOI GeoJSON outputs under
  `data_source/data/city_aois/generated/city_buffers_5km_by_city/`.
- Updated and reran `data_source/source/city_aois/create_city_buffers.py`.
  The script now writes:
  - one combined 27-feature file:
    `data_source/data/city_aois/generated/city_buffers_5km.geojson`
  - 27 single-city files:
    `data_source/data/city_aois/generated/city_buffers_5km_by_city/*_5km.geojson`
- Added `data_source/source/city_aois/README.md` to document city AOI script
  changes and run behavior.
- Updated `data_source/source/city_aois/create_city_buffers.py` to use
  `WUP2018-F22-Cities_Over_300K_Annual_V7.xls` as the city-center/CBD
  coordinate source. The script now rebuilds `cities_sample.csv` from
  `README.md` current cities and WUP 2018 coordinates before regenerating
  the combined and city-specific 5km AOI GeoJSON files.
- Updated the `README.md` Current Cities section using the LiDAR Readiness
  Screening in `data_catalog.md`. The active sample now includes 29 cities
  with ready, partial, promising, or otherwise strong open LiDAR access.
- Reran `data_source/source/city_aois/create_city_buffers.py` for the 29-city
  LiDAR-ready sample using WUP 2018 CBD coordinates. The generated city sample,
  combined 5km GeoJSON, and city-specific 5km GeoJSON folder now all contain
  exactly 29 current cities.
- Created `data_source/source/city_aois/venv_city_aois/` for reproducible
  execution of `create_city_buffers.py`, pinned `xlrd==2.0.2` in
  `data_source/source/city_aois/requirements.txt`, and updated the script to
  automatically relaunch itself inside that local virtual environment.
- Created city-specific `source/<city_slug>/` and `generated/<city_slug>/`
  folders for the 29 current cities under these data domains:
  `building_footprints`, `planet_imagery`, `height_labels`, `elevation`,
  `predictions`, `validation`, and `benchmark_products`.
- Updated `claude.md` to document the domain-first, city-second folder pattern
  and the current city slugs.
- Updated `data_source/source/planet_imagery/search_planet_city_scenes.py` to
  use the README current-city list, city-specific 5km AOI GeoJSON buffers, and
  a minimum 95% AOI coverage filter for Planet PSScene metadata results.
- Created `data_source/source/planet_imagery/venv_planet_imagery/`, added
  `data_source/source/planet_imagery/requirements.txt`, and updated the Planet
  search script to automatically relaunch inside that local virtual
  environment.
- Added a building-footprint source-date manifest for Planet scene selection:
  `data_source/source/planet_imagery/building_footprint_source_dates.csv`.
- Added `data_source/source/planet_imagery/select_planet_city_scenes.py` and
  selected two reviewed candidate scenes per current city from
  `cities_scenes_results_planet.csv`, writing
  `data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv`.
- Split Planet imagery acquisition into an asynchronous order/download workflow:
  - `data_source/source/planet_imagery/order_selected_planet_scenes.py`
    creates AOI-clipped Planet orders and writes
    `data_source/data/planet_imagery/generated/planet_orders_manifest.csv`.
  - `data_source/source/planet_imagery/download_ordered_planet_scenes.py`
    reads the manifest and downloads only completed orders.
  - `data_source/source/planet_imagery/download_selected_planet_scenes.py`
    is now a deprecated guard that exits with instructions.
- Corrected repository path handling across project scripts. Planet order and
  download paths are now resolved from the detected
  `building-height-prediction` repository root and paths outside that root are
  rejected. The city AOI Python script and Planet analysis R script are also
  anchored to their detected repository root.
- Moved the downloaded Boston Planet test order from the mistakenly created
  `SUMMER 2026 RA/data_source/` tree into
  `data_source/data/planet_imagery/source/boston/` in this repository and
  updated the order manifest paths.
- Created `data_source/source/building_footprints/clip_building_footprints.py`
  to clip each current city's raw building footprints to its 5km AOI.
- Created `data_source/source/building_footprints/venv_building_footprints/`,
  added `data_source/source/building_footprints/requirements.txt`, and updated
  the building-footprint script to automatically relaunch inside that local
  virtual environment.
- Ran the building-footprint clipping script for all 29 current cities. Every
  city now has a non-empty GeoPackage output in:
  `data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg`.
- Added `data_source/source/height_labels/download_usgs_3dep_lidar.py` for
  reproducible USGS 3DEP project discovery and AOI-filtered LAZ downloads.
- Created `data_source/source/height_labels/venv_height_labels/`, pinned its
  dependencies in `requirements.txt`, and documented the workflow in the
  folder-level `README.md`.
- Queried the official National Map products API and 3DEP Elevation Index for
  Boston, Chicago, Los Angeles, New York City, San Francisco, and Seattle.
- Selected reviewed USGS work units and wrote project/tile inventories to
  `data_source/data/height_labels/generated/`.
- Verified 789 unique AOI-intersecting LAZ tiles with a total estimated size of
  85.09 GiB. No LiDAR files were downloaded during discovery.
- Downloaded the 68 selected New York City and 99 selected Los Angeles USGS
  3DEP LAZ tiles. All 167 manifest files pass expected-byte and SHA-256
  verification, totaling 9.70 GiB.
- Strengthened the USGS downloader to recover full-size,
  checksum-matching `.part` files left when cloud synchronization interrupts
  the final atomic rename.
- Added
  `data_source/source/height_labels/LIDAR_BUILDING_HEIGHT_PROCESS.md`, which
  defines the proposed LiDAR roof-to-local-ground height estimator, DTM/DSM
  processing stages, point-class filters, quality tiers, temporal checks,
  validation design, and final building-level schema.
- Inspected representative New York City and Los Angeles LAZ files. Ground is
  class 2, while most urban surface returns are class 1; the sample tiles do
  not provide building class 6, so footprint-masked roof extraction is
  required.
- Confirmed that the clipped footprint files contain independent official
  height fields (`HEIGHT_ROO` for New York City and `HEIGHT` for Los Angeles)
  that can validate, but must not determine, the LiDAR-derived labels.
- Reviewed the UT-GLOBUS and Microsoft TEMPO papers and repositories for
  building-height modeling design choices. Created
  `notes/NYC_LA_MODEL_TRAINING_PLAN.md` with a concrete NYC/LA training plan
  that keeps `city x building` as the canonical unit, uses raster imagery chips
  and masks as model inputs, starts with LiDAR label diagnostics, then proceeds
  to a tabular baseline and image-chip model.
- Added `data_source/source/height_labels/derive_lidar_building_heights.py`
  to compute NYC/LA LiDAR-derived building-height diagnostics from
  manifest-approved USGS 3DEP LAZ tiles and clipped building footprints.
  The script writes p50, p75, p90, p95, max-clean height candidates, local
  ground elevation, official-height comparisons, quality tiers, and
  training/validation usability flags.
- Smoke-tested the LiDAR height diagnostic script with three buildings in NYC
  and three buildings in LA using `--skip-sha256`. Both smoke tests completed
  cleanly and wrote the expected diagnostic outputs.
- Ran the default 500-building-per-city NYC/LA LiDAR diagnostic with official
  footprint height units confirmed as feet and converted to meters. Los Angeles
  produced 444 training-usable labels out of 500 sampled buildings; New York
  City produced 464 training-usable labels out of 500 sampled buildings.
- Added height-label diagnostic documentation, including a step-by-step
  explanation of `derive_lidar_building_heights.py`, diagnostic metric
  definitions, and a LiDAR point-classification guide.
- Created `data_source/source/height_labels/city_crs_reference.csv` with AOI
  CRS, recommended local metric CRS, and confirmed U.S. LiDAR CRS metadata.
- Created `data_source/source/height_labels/lidar_sampling_geometry_tikz.tex`
  as a TikZ schematic for the roof/ground sampling geometry.
- Generated diagnostic summary-statistics tables and scatter plots under
  `data_source/data/height_labels/generated/diagnostic_analysis/`.
- Updated `derive_lidar_building_heights.py` to support repeated simple random
  diagnostic samples with replacement using `--sample-runs`. The script writes
  one temporary CSV per run and merges those files into each city's final
  `building_height_diagnostics_sample.csv`.
- Ran 15 independent 500-building with-replacement diagnostic samples per city
  with official height fields confirmed as feet and converted to meters. Los
  Angeles produced 7,500 sampled rows from 7,165 unique buildings and 6,702
  training-usable rows; New York City produced 7,500 sampled rows from 6,907
  unique buildings and 6,989 training-usable rows.
- Rewrote `data_source/source/height_labels/LiDAR_diagnostics_verification.R`
  for metric selection and generated summary statistics, city/metric scatter
  plots, and RMSE-by-height-bin box plots under
  `data_source/data/height_labels/generated/diagnostic_analysis/metric_selection/`.
- Updated `derive_lidar_building_heights.py` to enforce a 2.4 m minimum for
  positive LiDAR-derived heights and to remove the prior upper-height rejection
  rule. Reran the 15 x 500 with-replacement diagnostics with reproducible seed
  `20260625` (run seeds `20260625` through `20260639`). Los Angeles produced
  7,500 sampled rows from 7,172 unique buildings and 6,808 training-usable rows;
  New York City produced 7,500 sampled rows from 6,929 unique buildings and
  6,999 training-usable rows. Regenerated the metric-selection plots and tables
  from this seeded diagnostic run.
- Updated `LiDAR_diagnostics_verification.R` so nonpositive LiDAR-derived
  heights are set to missing before metric-selection summaries, scatter plots,
  and RMSE calculations. Added
  `lidar_nonpositive_height_cleaning_audit.csv` to document the excluded rows.
- Finalized the LiDAR label schema in `derive_lidar_building_heights.py`:
  `height_label_m` is the primary p90 label, `height_p95_m` and
  `height_max_m` are robustness labels, and `local_ground_m` records the local
  ground estimate. Ran the all-building pipeline with official height units in
  feet converted to meters. Los Angeles produced 79,645 footprint rows with
  71,889 training-usable labels; New York City produced 46,744 footprint rows
  with 43,486 training-usable labels. Full outputs are stored as
  `building_height_labels_all.csv`, `height_definition_comparison_all.csv`,
  `quality_tier_summary_all.csv`, and `lidar_building_heights_all.gpkg` under
  each city folder in `data_source/data/height_labels/generated/`.
- Added `data_source/source/building_footprints/merge_contiguous_footprints.py`
  to diagnose and create candidate merged footprint layers for buildings split
  across multiple adjacent polygons. Ran the NYC/LA diagnostic with
  shared-boundary merging and a 0.5 m official-height similarity gate. Los
  Angeles changed from 79,645 original polygons to 74,795 merged polygons; New
  York City changed from 46,744 to 32,578. The script writes merged layers,
  original-to-merged crosswalks, component diagnostics, and review-sample
  GeoPackages under each city's building-footprint generated folder.
- Updated the merged footprint outputs to carry official source height and
  ground/elevation summaries in meters, including mean, median, min, max, and
  area-weighted values. NYC `HEIGHT_ROO`/`GROUND_ELE` and LA `HEIGHT`/`ELEV`
  are converted from feet to meters before aggregation.
- Updated `data_source/source/building_footprints/clip_building_footprints.py`
  so 5km building-footprint generation selects any source polygon intersecting
  the 5km AOI and preserves the full original polygon geometry instead of
  clipping geometry to the AOI boundary. Reran the script for all 29 current
  cities. The generated city GeoPackages keep the existing
  `<city_slug>_building_footprints_5km.gpkg` names for downstream
  compatibility and include
  `aoi_selection_rule = intersects_5km_aoi_preserve_full_geometry`.
- Reran `data_source/source/building_footprints/merge_contiguous_footprints.py`
  for Los Angeles and New York City using the regenerated whole-geometry 5km
  footprint files. Los Angeles now changes from 79,645 original polygons to
  74,795 merged polygons across 3,610 merged components; New York City remains
  46,744 to 32,578 across 6,821 merged components.
- Added `data_source/source/planet_imagery/select_lidar_aligned_planet_scenes.py`
  to select NYC/LA Planet scenes around the USGS LiDAR capture windows. The
  LiDAR windows are 2023-01-08 to 2024-01-07 for Los Angeles and 2013-08-06 to
  2014-04-21 for New York City. The selected LA scenes are inside the LiDAR
  window; the selected NYC scenes are the nearest strict clean post-LiDAR
  winter/summer scenes in the current Planet scene list because the inventory
  begins after the 2013-2014 LiDAR capture. Outputs are
  `lidar_capture_summary_for_planet_selection.csv` and
  `selected_lidar_aligned_planet_scenes.csv`.
- Ran a targeted Planet PSScene metadata backfill for 2010-01-01 through
  2016-01-01 across all 29 current cities using
  `search_planet_city_scenes.py`. The query wrote
  `cities_scenes_results_planet_2010_2015_backfill.csv` with six qualifying
  rows, none for New York City. Added and ran
  `merge_planet_scene_backfill.py`; all six backfill rows were already present
  in `cities_scenes_results_planet.csv`, so the main table remains 17,033
  rows. Reran the LiDAR-aligned selector; NYC remains the 2020 winter/summer
  pair because no closer 2010-2015 NYC PSScene rows satisfy the current search
  constraints.
- Created Planet orders for the four LiDAR-aligned NYC/LA scenes using
  `selected_lidar_aligned_planet_scenes.csv`; no downloads were run. New York
  City orders use `ortho_analytic_4b_sr` / `analytic_sr_udm2` because the 2020
  scenes do not expose the selected 8-band SR asset, while Los Angeles orders
  use `ortho_analytic_8b_sr` / `analytic_8b_sr_udm2`. The order IDs are:
  NYC winter `1da57091-4c6e-4b3b-bde1-2be6aace54bc`, NYC summer
  `bfd6d704-820c-4457-a089-f835d36f8383`, LA winter
  `e7156ff1-873e-4a9c-9537-0507b6c97e07`, and LA summer
  `c074f656-c35f-40f1-86e0-5c1c87bd3de3`. Updated
  `order_selected_planet_scenes.py` so future order runs preserve the
  manifest's download bookkeeping columns.
- Updated `download_ordered_planet_scenes.py` with a repeatable `--scene-id`
  filter and used it to download only the four LiDAR-aligned NYC/LA Planet
  orders. Existing older NYC/LA scene folders were preserved. Each new order
  downloaded five files and is marked `downloaded` in
  `planet_orders_manifest.csv`.
- Updated `derive_lidar_building_heights.py` to support
  `--footprint-source merged` and to write `height_mean_m` and
  `height_median_m` alongside the existing percentile and max LiDAR height
  metrics. Reran the full all-building LiDAR pipeline on the refreshed merged
  Los Angeles and New York City footprint layers. Los Angeles produced 74,795
  merged footprint rows, with 68,793 training-usable labels; New York City
  produced 32,578 merged footprint rows, with 31,356 training-usable labels.
  Outputs are stored as `building_height_labels_merged_all.csv`,
  `height_definition_comparison_merged_all.csv`,
  `quality_tier_summary_merged_all.csv`, and
  `lidar_building_heights_merged_all.gpkg` under each city folder in
  `data_source/data/height_labels/generated/`.
- Added `data_source/source/height_labels/rasterize_lidar_heights_to_planet_grid.py`
  to rasterize the merged LiDAR height GeoPackages onto the exact downloaded
  PlanetScope grids. The script uses each Planet TIFF as the template, copies
  its CRS, affine transform, 3 m pixel size, width, height, and bounds, and
  writes multiband height rasters under
  `data_source/data/height_labels/generated/<city_slug>/planet_aligned_lidar_rasters/`.
  Ran the workflow for all downloaded Los Angeles and New York City Planet
  scenes: four LA rasters in `EPSG:32611` at 3340 x 3325 pixels and four NYC
  rasters in `EPSG:32618` at 3342 x 3329 pixels. Independent audit confirmed
  CRS, transform, dimensions, bounds, and resolution exactly match the Planet
  templates. The run also wrote
  `data_source/data/height_labels/generated/planet_aligned_lidar_raster_summary.csv`.
- Created the `ml_models` task folders under `data_source/source/` and
  `data_source/data/`. Added `data_source/source/ml_models/README.md` as the
  running ML decision log. The first logged plan predicts `height_mean_m` from
  winter and summer PlanetScope features, starts with building-level zonal
  features, compares a mean baseline, ridge, random forest, and XGBoost, and
  documents why naive pixel-level training can overstate accuracy.
- Reviewed `zhu-xlab/GlobalBuildingAtlas` and its `HTC-DC-Net` height-model
  submodule. Added
  `data_source/source/ml_models/GLOBAL_BUILDING_ATLAS_HEIGHT_MODEL.md`, which
  summarizes their PlanetScope height-prediction workflow: PyTorch HTC-DC Net,
  raster-chip training with image/mask/nDSM TIFF triplets, sliding-window
  Planet inference, raster and building-aware validation metrics, and lessons
  for adapting the workflow to NYC/LA.
- Added `data_source/source/ml_models/HTC_DC_NET_APPLICATION_README.md` after
  deeper inspection of the GlobalBuildingAtlas and HTC-DC-Net code. The note
  records that the public repos do not include raw LiDAR point-cloud-to-nDSM
  generation code, identifies the active polygonization/simplification path
  (`PolygonizerV10`, `PolyRegularizerV5`, and the custom dynamic-programming
  `simplify_poly_jsons` ring simplifier), and lists the GBA-style image/mask
  /height chip dataset we need before HTC-DC Net can run on NYC/LA.
- Added `data_source/source/height_labels/build_lidar_ndsm_raster.py` and
  generated the first NYC Planet-aligned LiDAR nDSM raster at
  `data_source/data/height_labels/generated/new_york_city/lidar_ndsm/new_york_city_lidar_ndsm_planet_aligned.tif`.
  The workflow uses only `NY_New_York_CMGP_SANDY_LiDAR_15`, explicitly
  excludes `NJ_New_Jersey_SANDY_LiDAR_15`, aligns to the NYC Planet scene
  `20200122_154449_92_1061`, and writes DSM, observed DTM, filled DTM, nDSM,
  building mask, and building-only nDSM bands.
- Generalized `build_lidar_ndsm_raster.py` to create HTC-DC-Net-style
  full-scene and chip datasets. The script now writes `_IMG.tif` RGB Planet
  rasters, `_BLG.tif` building masks, `_AGL.tif` building-only nDSM targets,
  256 x 256 image/mask/AGL chips, `train.txt`, `val.txt`, `test.txt`,
  `all.txt`, `chips_manifest.csv`, and `stats/image_stats.pickle`. Ran it for
  three variants: NYC New York LiDAR (`103` chips), NYC/New Jersey Sandy LiDAR
  variant (`17` chips), and Los Angeles (`142` chips).
- Added `data_source/source/ml_models/combine_htc_datasets.py` and created the
  combined HTC-DC Net RGB v1 dataset at
  `data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/`. The
  combined dataset includes only true NYC and LA chips, excludes the
  NYC/New Jersey Sandy LiDAR diagnostic variant, and writes unified `image/`,
  `mask/`, `ndsm/`, split files, `chips_manifest.csv`, `image_stats.pickle`,
  and `ndsm_stats.pickle`. Current counts are 245 chips total: 171 train, 37
  validation, and 37 test.
- Vendored the HTC-DC Net code at
  `data_source/source/ml_models/external/HTC-DC-Net/` from commit
  `adae55edc8be589757cec57f839d59a681d93364` and added project-specific setup
  files under `data_source/source/ml_models/htc_dc_net_setup/`. Created the
  local Apple Silicon smoke-test environment
  `data_source/source/ml_models/venv_htc_dc_net/`, wrote HTC-compatible
  root-level dataset statistics, and verified the released `efficientnetb0`
  HTC-DC Net config on the combined NYC+LA dataset. The verification loads the
  dataloader, builds `UBins` with `AdamW`, runs `model(image, gt)`, computes
  losses and predictions, and completes one backward/optimizer step.
- Added `data_source/source/ml_models/run_htc_mini_training.py` and ran the
  first mini HTC-DC Net training/estimation job using 5 New York City and 5
  Los Angeles training chips, one epoch, and seed `20260707`. The run trained
  the real `UBins`/`efficientnetb0` model, saved `model_last.pth`, exported 10
  predicted nDSM rasters, and wrote `training_history.csv` and
  `predictions_summary.csv` under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch1/`.
- Updated the HTC mini-training prediction exporter to write georeferenced
  GeoTIFF predictions by copying each chip's `_AGL.tif` CRS, transform, width,
  height, and resolution. Regenerated the first mini-run predictions and
  verified that all 10 predictions match their target chip geometry: five Los
  Angeles predictions in `EPSG:32611` and five New York City predictions in
  `EPSG:32618`.
- Updated `build_lidar_ndsm_raster.py` so finite nonpositive nDSM pixels inside
  the building mask are assigned a minimum AGL target of `2.4` meters before
  writing full-scene HTC `_AGL.tif` files and chips. Reran the full nDSM/HTC
  chip build for New York City and Los Angeles, rebuilt the combined
  `nyc_la_rgb_v1` HTC dataset, refreshed HTC-compatible root stats, and reran
  the 5 NYC + 5 LA mini model with the same seed and one-epoch parameters. The
  rebuild imputed 268 NYC pixels and 1,829 LA pixels; validation found zero
  valid `AGL <= 0` pixels inside the building mask in the full-scene, combined
  chip, and mini-run targets.
- Extended `run_htc_mini_training.py` to save periodic prediction rasters,
  periodic checkpoints, epoch-level training-loss summaries, and
  `training_loss.png`. Ran the same 5 NYC + 5 LA mini experiment for 50 epochs
  with `lr = 0.001`, seed `20260707`, and prediction/checkpoint exports every
  10 epochs. Outputs are under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch50_lr001/`.
  Mean training loss fell from 57.0792 at epoch 1 to 33.3211 at epoch 50, with
  the best mean loss 33.2129 at epoch 42. Final mini-run mean MAE was 9.5385 m
  for Los Angeles and 27.8521 m for New York City.
- Ran a larger 40-chip HTC-DC Net mini experiment using 20 random New York City
  and 20 random Los Angeles training chips, seed `20260707`, 50 epochs,
  `lr = 0.001`, and prediction/checkpoint exports every 10 epochs. Outputs are
  under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc20_la20_seed20260707_epoch50_lr001/`.
  Mean training loss fell from 31.5639 at epoch 1 to 24.3134 at epoch 50,
  which was the best epoch. Final mean MAE was 4.5515 m for Los Angeles and
  22.9391 m for New York City. The run exported 40 final georeferenced
  prediction rasters: 20 in `EPSG:32611` and 20 in `EPSG:32618`.

## Current Status

- Repository scaffold is reset and ready for the next project task.
- City AOI centroid source files have been regenerated from WUP 2018.
- `README.md` now lists 29 current cities selected around LiDAR readiness.
- `cities_sample.csv` and city AOI 5km buffers are current for the 29-city
  LiDAR-ready sample using WUP 2018 CBD coordinates.
- The city AOI script can be run with
  `python data_source/source/city_aois/create_city_buffers.py`; it will use
  `data_source/source/city_aois/venv_city_aois/` automatically.
- The Planet metadata search script can be run with
  `python3 data_source/source/planet_imagery/search_planet_city_scenes.py`; it
  will use `data_source/source/planet_imagery/venv_planet_imagery/`
  automatically.
- Planet scene selection can be rerun with
  `python3 data_source/source/planet_imagery/select_planet_city_scenes.py`.
- Planet imagery ordering and downloading are now separate. Run
  `order_selected_planet_scenes.py --dry-run` before `--confirm-order`, then
  run `download_ordered_planet_scenes.py --dry-run` before
  `--confirm-download` after orders complete.
- The building-footprint AOI-selection script can be run with
  `python3 data_source/source/building_footprints/clip_building_footprints.py`;
  it will use
  `data_source/source/building_footprints/venv_building_footprints/`
  automatically. The output name still contains `_5km`, but geometries are
  preserved whole when they intersect the 5km AOI.
- Contiguous footprint merge diagnostics can be run with
  `python3 data_source/source/building_footprints/merge_contiguous_footprints.py`;
  merged layers are not canonical until their review samples have been
  inspected.
- The USGS LiDAR script can be run with
  `python3 data_source/source/height_labels/download_usgs_3dep_lidar.py
  --dry-run --estimate-sizes`; it automatically uses
  `data_source/source/height_labels/venv_height_labels/`.
- U.S. 3DEP discovery is complete. New York City and Los Angeles are fully
  downloaded and verified. The remaining four U.S. cities are still pending.
- City-specific data folders now exist locally for each of the 29 current
  cities across the main city-varying data domains.
- Task-specific source code currently exists for city AOIs, Planet imagery,
  building footprints, U.S. height-label acquisition, and NYC/LA LiDAR
  building-height diagnostics.
- Planet-aligned LiDAR height rasters now exist for all downloaded LA and NYC
  PlanetScope scenes. These rasters are ready to use as pixel-aligned label
  layers for imagery-chip extraction and model-training data assembly.
- LiDAR point-cloud nDSM and HTC-DC-Net-style image/mask/AGL chip datasets now
  exist for NYC New York LiDAR, the NYC/New Jersey Sandy LiDAR variant, and Los
  Angeles. These are ready for visual QA before attempting model training.
- The combined NYC+LA HTC-DC Net RGB v1 dataset exists under
  `data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/` and is
  ready for HTC training experiments. The HTC dataloader/environment/model
  smoke test has passed with the released `efficientnetb0` config.
- HTC-DC Net setup instructions, configs, and verification notes are in
  `data_source/source/ml_models/htc_dc_net_setup/`.
- The first HTC-DC Net mini-run estimates are saved locally under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch1/`.
  They have been regenerated against the updated 2.4 m minimum AGL target
  definition.
- The 50-epoch HTC-DC Net mini-run estimates, periodic checkpoints, periodic
  predictions, and loss plot are saved locally under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc5_la5_seed20260707_epoch50_lr001/`.
- The 40-chip, 50-epoch HTC-DC Net mini-run estimates, periodic checkpoints,
  periodic predictions, and loss plot are saved locally under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc20_la20_seed20260707_epoch50_lr001/`.
- Diagnostics showed that the two 50-epoch mini-runs collapsed to constant
  prediction rasters even though the 1-epoch smoke run produced spatially
  varying predictions. The mini-training runner now writes per-chip prediction
  range/std/unique-value diagnostics and can stop early when saved predictions
  collapse.
- ML model decisions and future scripts should now be maintained under
  `data_source/source/ml_models/`; generated ML tables, metrics, predictions,
  and model artifacts should go under `data_source/data/ml_models/generated/`.
- The next modeling plan is documented in
  `notes/NYC_LA_MODEL_TRAINING_PLAN.md`.
- Added
  `data_source/source/planet_imagery/select_intermediate_sun_elevation_scenes.py`
  for the 12-channel HTC-DC Net experiment. The selector chooses two strict
  zero-cloud, full-AOI, standard-quality intermediate PlanetScope scenes per
  city to maximize sun-elevation diversity across the four-scene set. It wrote
  `selected_intermediate_planet_scenes.csv` and
  `intermediate_sun_elevation_scene_review.csv`. The selected new scenes are
  LA `20230713_182102_57_241c`, LA `20231203_171912_53_2445`, NYC
  `20200124_153319_56_1063`, and NYC `20200526_155004_25_1058`.
- Ran the Planet order dry-run for the four intermediate scenes. The dry-run
  passed without creating orders: LA scenes use `ortho_analytic_8b_sr`
  (`analytic_8b_sr_udm2`) and NYC scenes use `ortho_analytic_4b_sr`
  (`analytic_sr_udm2`). After explicit approval, submitted the four live
  Planet orders. After the orders reached `success`, ran the required download
  dry-run and then downloaded all four scenes. The order IDs, source folders,
  and raster-open verification notes are recorded in
  `data_source/data/planet_imagery/generated/intermediate_planet_order_status.md`.
- Added `data_source/source/ml_models/build_htc_dataset_multiscene.py` to
  build a future `nyc_la_12ch_v1` HTC-DC Net dataset after the new Planet
  orders are downloaded. The script stacks four RGB scenes per city, writes
  the usual `image/`, `mask/`, `ndsm/`, split files, manifest, normalization
  stats, and a `scene_channel_plan.csv`.
- Updated `run_htc_mini_training.py` so `--in-channels` is validated against
  the dataset image band count and `image_stats.pickle` lengths before
  training. Existing `nyc_la_rgb_v1` and `nyc_la_6ch_v1` datasets both passed
  channel validation.
- Built the 12-channel NYC+LA HTC-DC Net dataset at
  `data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1/` using four
  PlanetScope RGB scenes per city. The dataset reuses the `nyc_la_6ch_v1`
  train/validation/test split for comparability, producing 171 training chips
  with 76 NYC and 95 LA chips. Alignment verification passed for all 245 chips:
  all image chips have 12 bands, image/mask/AGL rasters share CRS, transform,
  resolution, width, and height, LA chips are in `EPSG:32611`, and NYC chips
  are in `EPSG:32618`.
- Ran the 12-channel HTC-DC Net model with the same settings as the previous
  low-rise bin-weighted 6-channel run, changing only the dataset and
  `--in-channels 12`. The run is saved under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc76_la95_12ch_lowrise_binweighted_bg005_seed20260715_epoch20_guarded/`.
  Final training-chip diagnostics: LA mean MAE 4.17 m, NYC mean MAE 12.05 m,
  overall mean MAE 7.67 m, and 2 collapsed chips out of 171.
- Ran the comparable 50-epoch 12-channel HTC-DC Net model under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc76_la95_12ch_lowrise_binweighted_bg005_seed20260715_epoch50_guarded/`.
  Final training-chip diagnostics improved to LA mean MAE 3.38 m, NYC mean
  MAE 8.83 m, overall mean MAE 5.80 m, and 0 collapsed chips out of 171.
  Building-level scatter diagnostics are saved in the run `evaluation/`
  folder; the overall building-level fit is `y = -0.100 + 0.908x`,
  RMSE 5.69 m, and R2 0.808.
- Added HTC-DC Net cross-validation tooling for the `nyc_la_12ch_v1` dataset.
  `build_htc_cv_folds.py` creates three fold-specific HTC dataset folders
  from the original train+validation pool while leaving the 37-chip test split
  untouched. `run_htc_cross_validation.py` runs the staged parameter grid and
  ranks configurations by mean validation-chip RMSE. A one-config, one-fold,
  one-epoch smoke test completed successfully and wrote outputs under
  `data_source/data/ml_models/generated/htc_dc_net/cross_validation/nyc_la_12ch_v1/`.
- Added `data_source/source/planet_imagery/search_planet_nyc_la_view_angle_scenes.py`
  for a metadata-only Planet Data API query of Los Angeles and New York City
  scenes that includes `view_angle`, `satellite_azimuth`, and `sun_azimuth`.
  Ran it for 2010-01-01 through 2026-07-21. The separate output
  `data_source/data/planet_imagery/generated/nyc_la_scenes_results_planet_with_view_angle.csv`
  contains 1,586 rows: 904 Los Angeles scenes and 682 New York City scenes,
  with non-missing `view_angle` for every row. No Planet orders or downloads
  were created.
- Created `data_source/data/planet_imagery/generated/selected_off_nadir_planet_scenes.csv`
  for two user-selected high-view-angle PlanetScope scenes: LA
  `20251002_190325_64_24d1` and NYC `20241113_160040_50_24e0`. Ran the
  required dry-run successfully, then submitted AOI-clipped Planet orders only
  after user approval. Both orders use `ortho_analytic_8b_sr`
  (`analytic_8b_sr_udm2`) and were queued in `planet_orders_manifest.csv`: LA
  order `ac6d971f-bcd6-4b2d-88aa-5634655acd03`, NYC order
  `d59c99c4-18a7-4008-bbb1-0e39e491bf3f`. No downloads were run.
- Scaffolding is ready for the next planned off-nadir 3-channel HTC-DC Net
  experiment. Created the pending dataset folder
  `data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_v1/`
  with `image/`, `mask/`, `ndsm/`, `stats/`, copied split files, and wrote
  `off_nadir_scene_plan.csv`, `dataset_setup_status.json`, and `README.md`.
  Created the planned run folder
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc76_la95_offnadir_3ch_lowrise_binweighted_bg005_seed20260720_epoch50_guarded/`
  with `planned_config.yaml` and `README.md`. Also added
  `data_source/source/ml_models/HTC_MODEL_RUN_SUMMARY_README.md`, which
  summarized completed HTC model runs and, at that stage, highlighted the
  50-epoch 12-channel low-rise/high-rise bin-weighted model. That historical
  designation was later superseded by the selected off-nadir RGB+NIR B0 model.
- Downloaded the two queued off-nadir PlanetScope orders after the required
  dry-run confirmed both were in `success` state. LA scene
  `20251002_190325_64_24d1` and NYC scene `20241113_160040_50_24e0` each
  downloaded five files, including the clipped 8-band SR TIFF, UDM2, metadata
  JSON, XML, and manifest. Raster-open verification passed: LA is
  `EPSG:32611`, 3340 by 3325 pixels, 8 bands, 3 m resolution; NYC is
  `EPSG:32618`, 3342 by 3329 pixels, 8 bands, 3 m resolution. The off-nadir
  HTC dataset status is now pending LiDAR nDSM/mask/RGB chip creation.
- Built and validated the off-nadir 3-channel HTC dataset at
  `data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_v1/`.
  The city-specific off-nadir LiDAR nDSM/HTC builds completed for LA scene
  `20251002_190325_64_24d1` and NYC scene `20241113_160040_50_24e0`, using
  each Planet scene as the target grid. One NYC edge chip with non-empty
  mask/AGL but an all-zero RGB image was excluded. The final model-ready
  dataset has 244 chips: 142 LA and 102 NYC; train/validation/test are
  171/36/37, with the training split preserving 95 LA and 76 NYC chips. Final
  validation passed: every chip has non-empty image, mask, and AGL data, and
  every chip and full-scene `_IMG`, `_BLG`, and `_AGL` triplet shares CRS,
  transform, dimensions, and 3 m resolution.
- Trained the proposed off-nadir 3-channel HTC-DC Net model
  `nyc76_la95_offnadir_3ch_lowrise_binweighted_bg005_seed20260720_epoch50_guarded`
  on `nyc_la_off_nadir_rgb_v1`. Parameters: 3 input channels, 95 LA + 76 NYC
  training chips, 50 epochs, learning rate 0.00003, batch size 8, AdamW,
  EfficientNet-B0, bin-weighted low-rise/high-rise loss with edges
  `3,6,10,25,50` and weights `4,3,2,1,3,8`, background loss weight 0.05,
  prediction/checkpoint exports every 10 epochs, and collapse guard enabled.
  Training completed all 50 epochs without stop-on-collapse. Final epoch-50
  train-chip diagnostics: LA MAE 3.51 m and RMSE 5.28 m; NYC MAE 9.29 m and
  RMSE 12.88 m; overall MAE 6.08 m and RMSE 8.66 m; 0 collapsed chips in the
  final summary. Outputs are saved under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/nyc76_la95_offnadir_3ch_lowrise_binweighted_bg005_seed20260720_epoch50_guarded/`.
- Generated the full diagnostics package for the off-nadir 3-channel epoch-50
  model. Exported unmasked predictions for train, validation, test, and all
  chips; built all-chip masked and unmasked 3 m city prediction mosaics; and
  saved building-level scatter plots, three-panel scatter diagnostics,
  height-bin RMSE bars, height-bin bias bars, residual boxplots, and full-city
  PlanetScope/prediction/target three-panel plots under the run `evaluation/`
  folder. Building-level validation diagnostics show overall RMSE 8.08 m and
  R2 0.535, with LA RMSE 3.99 m and NYC RMSE 18.29 m.
- Added the standard HTC-DC Net post-training diagnostics workflow. The new
  wrapper `data_source/source/ml_models/run_htc_post_training_diagnostics.py`
  exports split predictions, builds all-chip mosaics, runs building-level
  scatter diagnostics, creates three-panel scatter plots, creates height-bin
  RMSE/bias bars and residual boxplots, and creates full-city
  PlanetScope/prediction/target panels when full-scene rasters are available.
  The companion documentation
  `data_source/source/ml_models/HTC_POST_TRAINING_DIAGNOSTICS_README.md`
  records this as the preferred diagnostics package after each model run. A
  validation-only smoke test passed on the off-nadir epoch-50 model.
- Created and trained two independent 4-channel off-nadir HTC-DC Net variants.
  `data_source/source/ml_models/build_htc_dataset_single_scene_variants.py`
  builds `nyc_la_off_nadir_rgb_mask_v1` (RGB plus building-footprint mask) and
  `nyc_la_off_nadir_rgb_nir_v1` (RGB plus PlanetScope NIR reprojected to the
  RGB chip grid). Both datasets have 244 chips with the same 171/36/37
  train/validation/test split as `nyc_la_off_nadir_rgb_v1`, and validation
  checks passed for 4 input channels, 3 m resolution, CRS/grid alignment, and
  non-empty channels. Trained both with the same off-nadir baseline parameters:
  50 epochs, learning rate 0.00003, batch size 8, AdamW/EfficientNet-B0,
  low-rise/high-rise bin-weighted loss, background loss weight 0.05, seed
  `20260721`, and collapse guard enabled. Final epoch-50 collapse checks were
  below the stop threshold for both models. Standard post-training diagnostics
  were generated for train, validation, test, and all chips. Building-level
  validation diagnostics: RGB+mask RMSE 7.87 m and R2 0.532; RGB+NIR RMSE
  7.70 m and R2 0.555. The comparison table is saved as
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/offnadir_4ch_variant_comparison_epoch_050.csv`.
- Added `data_source/source/ml_models/plot_offnadir_model_scatter_comparison.py`
  and generated common-axis building-level scatter comparisons for the three
  off-nadir models across training, validation, and test samples. Outputs
  include full-distribution and 0-50 m detail figures plus a consolidated
  metric table under
  `data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/offnadir_model_scatter_comparison_epoch_050/`.
- Added `data_source/source/ml_models/plot_lidar_height_split_distribution.py`
  and audited the off-nadir training, validation, and test LiDAR-height
  distributions at the building-component level. Overall distributions are
  similar, but NYC shows a moderate split shift: train-to-validation and
  train-to-test KS distances are 0.083 and 0.096, with mean absolute quantile
  differences of 2.88 m and 4.50 m. The results indicate that height-composition
  imbalance contributes to the generalization gap but does not explain it
  completely. Figures and tables are saved under the RGB+NIR run's
  `evaluation/lidar_height_split_distribution_epoch_050/` folder.
- Implemented and completed the full HTC-DC paper-recipe RGB+NIR run using the
  original 76-NYC/95-LA training split. Preflight found that the original
  `nyc_la_off_nadir_rgb_nir_v1` normalization statistics included all splits,
  so `nyc_la_off_nadir_rgb_nir_full_recipe_v1` was created with identical
  rasters and split membership but training-only four-band statistics. The new
  workflow adds EfficientNet-B5 support, explicit AdamW weight decay, 256
  adaptive bins, four-level `third` fusion, early HTC at 1 m, Gaussian
  foreground and uniform background constraints, standard positive-height L1,
  and 256/128 overlapping inference with mean and population-variance rasters.
  All 244 chip alignments and the batch-of-eight forward/backward preflight
  passed. Training completed 50 epochs; epoch 40 was selected with 11.9659 m
  city-balanced validation building RMSE. Standard pooled building RMSE was
  7.94 m on train, 8.71 m on validation, and 8.13 m on test. NYC remains the
  primary limitation: validation RMSE was 21.16 m versus 2.77 m for LA. The
  full recipe did not outperform the earlier RGB+NIR B0 model on pooled
  validation RMSE (8.71 m versus 7.70 m). Outputs and an honest interpretation
  are recorded in the run `README.md` under
  `nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_seed20260723_epoch50/`.
- Extended the Full HTC-DC training and preflight workflows with optional
  target-height bin weighting while retaining unweighted legacy defaults.
  Exact edges `3,6,10,25,50` m and weights `4,3,2,1,3,8` passed preflight and
  one-epoch smoke tests. The 50-epoch weighted RGB+NIR EfficientNet-B5 run
  completed with no collapsed validation chips. Epoch 15 was selected at
  11.6229 m city-balanced validation building RMSE, improving the comparable
  unweighted Full B5 result of 11.9659 m. Pooled building RMSE was 9.15 m on
  train, 8.41 m on validation, and 8.60 m on test. NYC remains difficult
  (19.71 m validation RMSE), and predictions still compress the high-rise
  tail. Complete diagnostics are in the run `README.md` under
  `nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_binweighted_seed20260723_epoch50/`.

- Selected
  `nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded`
  as the primary HTC-DC Net model for continued work. The model uses one
  off-nadir RGB+NIR PlanetScope scene per city, four channels, EfficientNet-B0,
  50 epochs, and target-height bin weighting. It was selected by held-out
  validation performance: pooled validation RMSE 7.70 m, validation R2 0.555,
  and NYC validation RMSE 18.18 m. Added
  `data_source/source/ml_models/SELECTED_HTC_DC_MODEL.md` as the canonical
  source for its parameters, checkpoint, paths, exact rerun command, and
  cross-computer transfer requirements. The model-ready data and checkpoint
  remain outside Git because `data_source/data/` is ignored.

## Remaining

- Expanded the global open-LiDAR screening inventory with a metadata-only
  17-country audit covering 249 WUP cities. The cross-platform script
  `query_named_country_open_lidar_availability.py` reuses the completed USGS
  results, performs official ArcGIS footprint-union queries for Canada and
  Ireland, validates representative file endpoints where exposed, and checks
  official national portals without downloading LiDAR, DSM, or DTM payloads.
  All 249 city records and all 249 portal/file checks completed successfully.
  Results are 206 `ready_for_download`, 26 `incomplete`, 14
  `manual_portal_check_required`, and 3 `registration_required`. The 26
  incomplete cities comprise 19 U.S. and 7 Canadian AOIs below 99% measured
  coverage; Australia's 11 cities, Glasgow, Edinburgh, and Belfast require
  interactive portal checks; Sweden's 3 cities require current STAC
  authorization/terms acceptance. The rebuilt 1,862-city global inventory now
  contains 206 ready, 26 incomplete, 14 manual-portal, 3 registration-required,
  23 query-required, and 1,590 not-yet-checked records.
- Completed a metadata-only USGS 3DEP audit for all 144 U.S. WUP cities with
  `query_usgs_3dep_global_city_availability.py`. All 144 API calls succeeded
  and returned downloadable LAZ metadata: 125 cities have at least 99% coverage
  of the circular 5 km AOI and are `ready_for_download`; 19 cities are below
  99% and are `incomplete`. No city returned `not_found` or `query_failed`. The
  audit recorded 25,323 AOI-intersecting tile records and download URLs. A
  representative URL for every city responded to a one-byte range request
  with HTTP 206; no LiDAR file bodies were saved. The global inventory builder
  now incorporates these results.
- Updated the requested U.S. readiness rule: cities with at least 99% of their
  5 km AOI covered by returned USGS LPC tile footprints are labeled
  `ready_for_download`; cities below 99% are labeled `incomplete`. The full
  144-city U.S. audit is rerun under this common rule so the six original pilot
  cities and the expanded city pool use the same coverage calculation.
- Added `select_training_cities_with_open_lidar.py`, a local and reproducible
  join between the 206 `ready_for_download` LiDAR cities and the 711-city
  PlanetScope training split. Matching by stable `city_slug` identifies 94
  training cities with ready LiDAR sources. The script writes a detailed city
  table and country summary without querying or downloading remote data.
- Added `add_training_lidar_acquisition_years.py` and enriched all 94 rows in
  the detailed training/open-LiDAR city table with city-specific acquisition
  years, start/end dates, precision, official metadata source, and explanatory
  notes. Exact official work-unit, project, or tile metadata is available for
  72 cities; the remaining rows retain the narrowest documented official city,
  flight-lot, regional, or campaign period instead of inventing a single year.
  The run used metadata indexes only and downloaded no LiDAR or imagery data.
- Added `select_planet_scenes_for_training_lidar_years.py`, a separate and
  resumable selector that targets eight PlanetScope scenes for each defensible
  LiDAR acquisition year among the 94 training/open-LiDAR cities. It retains
  all established scene-quality filters, changes the target to four scenes per
  solstice season and two per cardinal direction within each city-year, and
  writes combined, per-city-year, eligibility, summary, and shortfall CSVs.
  Exact and endpoint-range years are processed from 2016 onward; pre-2016 and
  broad unresolved ranges are reported rather than silently substituted or
  expanded. Syntax, CLI, and the no-network exclusion path passed locally;
  production selection awaits the 94-city Planet metadata held on Windows.
- Added `combine_planet_global_city_scene_metadata.py`, an API-free streaming
  combiner for the 1,862 per-city Planet query files. It validates the complete
  expected file set and every city/scene key, preserves a union of metadata
  columns, adds source-file provenance, and writes the combined CSV atomically.
  A partial one-city test combined all 186 Aba rows successfully; the complete
  run awaits the full per-city metadata directory on Windows.
- Extended the Planet metadata combiner with an explicit expected-city count,
  allowing the same validated workflow to create a separate all-scenes table
  for exactly the 94 training/open-LiDAR cities while retaining the strict
  1,862-city default.
- Extended the LiDAR-year Planet selector to accept the consolidated 94-city
  scene-metadata CSV directly while preserving the original per-city input
  mode. The 231 MB combined file validated at 57,481 unique scene rows across
  exactly 94 cities. An offline Boston test selected eight unique 2021 scenes
  with exact 4/4 solstice-season and 2/2/2/2 cardinal-direction balance; asset
  checks were deliberately skipped for this no-API test and remain required
  for production output.
- Revised the LiDAR selector's final unit from eight scenes per acquisition
  year to eight scenes total per city. The deterministic hierarchy now uses
  documented acquisition-year scenes first, nearest post-LiDAR scenes second,
  and only then flagged pre-LiDAR or non-solstice standard-quality fallbacks.
  The complete offline regression selected exactly 752 unique scenes: eight
  for every one of 94 cities, with an empty true-shortfall table. Of these,
  452 are from documented acquisition years, 292 are post-LiDAR, eight are
  flagged pre-LiDAR fallbacks for Newport, and three are flagged non-solstice
  fallbacks for Miami. Live RGB+NIR asset verification remains required before
  treating the production output as order-compatible.
- Added `audit_planet_raster_grids.py` and inspected all 14 downloaded analytic
  surface-reflectance GeoTIFFs currently available for Los Angeles and New York
  City. All seven Los Angeles rasters share EPSG:32611 and an identical 3 m
  grid; all seven New York City rasters share EPSG:32618 and a second identical
  3 m grid. The cities therefore differ in projection, while every scene grid
  is internally consistent within its city. Detailed header and summary CSVs
  were written for downstream LiDAR rasterization checks.
- Generalized `build_lidar_ndsm_raster.py` for the downloaded global-city
  directory layout. The script now audits all Planet analytic SR rasters for a
  city, selects a strict-majority complete grid, records minority-grid outliers,
  and verifies the nDSM against every majority-grid scene. It supports explicit
  classified LAS/LAZ inputs and existing single-band nDSM GeoTIFFs. Point-cloud
  runs now inventory classifications before rasterization and stop if the
  confirmed ground class (standard class 2 by default) is absent. Existing nDSM
  rasters are reprojected onto the selected Planet grid. No LiDAR payload was
  downloaded or processed as part of this source-code change.
- Generalized `order_planet_training_city_scenes.py` without changing its
  original 711-city defaults. Explicit arguments now allow the authoritative
  94 training/open-LiDAR cities to be joined to their training inventory,
  require exactly eight scenes per city, use distinct deterministic order
  names, and write a separate resumable manifest. An API-free in-memory test
  reconciled exactly 94 city orders and 752 unique city-scene rows. No Planet
  order was submitted and no imagery was downloaded during development.
- Added `run_us_lidar_to_planet_ndsm.py`, a US-only sequential and resumable
  workflow for the 54 USGS-backed training cities. It plans the latest
  qualifying acquisition, audits the strict-majority Planet grid, downloads
  only manifest-listed AOI tiles, confirms the point classification before
  using ground class 2, and writes one validated three-band Planet-aligned
  nDSM. Optional raw-LiDAR cleanup is gated behind an explicit flag and occurs
  only after validation; Planet imagery, failed downloads, manifests, audits,
  logs, and outputs are never deleted. API-free input validation confirmed 54
  cities, 9,997 candidate tile records, and eight selected scenes per city. A
  live metadata-only test found the USGS spatial attribute query unavailable,
  so the script records an explicit official-publication-date fallback instead
  of silently treating it as exact project collection metadata. No LiDAR data
  was downloaded during implementation or testing.
- Extended the US LiDAR orchestrator with configurable parallel tile transfers
  through `--download-workers`. A single coordinator still owns manifests and
  all CPU/memory-intensive raster stages remain sequential. Transfers now
  report per-tile progress, retry transient failures with exponential delays,
  use a 15-minute socket timeout, and preserve/resume `.partial` files through
  HTTP Range requests when supported. The default remains one worker; Windows
  runs may request eight without launching unsafe concurrent script processes.
- Added `download_ms_buildings_us_training_aois.py` to fill the 52-city US
  building-footprint prerequisite for the LiDAR workflow. It derives Microsoft
  level-9 quadkeys locally, pins the public `2026-07-24` release, caches only
  required immutable gzip partitions, streams GeoJSONL records, clips polygons
  exactly to each WUP 5 km AOI, and writes each GeoPackage in the AOI's declared
  CRS at the exact city path consumed by the LiDAR orchestrator. Microsoft
  model-derived height is excluded to prevent target leakage. A complete
  metadata-only regression found all 52 cities, 66 unique partitions, and an
  estimated 2.668 GB compressed transfer. An offline Boston partition test
  verified parsing, exact clipping, CRS retention, height exclusion, atomic
  GeoPackage output, and the required city-specific filename. No production
  footprint partition was downloaded during the 52-city dry run.
- Corrected Microsoft footprint boundary validation so independent CRS
  reprojection cannot reject microscopic clipping slivers. Both new and resumed
  GeoPackages now ignore only geometry within 5 cm of the AOI edge and continue
  to fail on material overflow beyond that positional tolerance.
- Fixed the post-validation footprint-area audit to use the shared CRS-safe
  metric-area helper, eliminating the undefined `metric_crs` failure exposed by
  the Birmingham pilot after its boundary check passed.

- Maintain folder-level README files in `data_source/source/<task>/` after
  commits, as required by `claude.md`.
- Harmonize AOI-selected building-footprint attributes into the common
  building-level schema once height-label processing begins.
- Inspect full-run rejected buildings and extreme residuals before moving to
  image-chip extraction and model training.
- Visually inspect `*_footprint_merge_review_sample.gpkg` for NYC and LA as an
  additional quality check on the merged-footprint layer.
