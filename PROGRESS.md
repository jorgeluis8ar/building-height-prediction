# Project Progress

Last updated: 2026-07-02

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
- ML model decisions and future scripts should now be maintained under
  `data_source/source/ml_models/`; generated ML tables, metrics, predictions,
  and model artifacts should go under `data_source/data/ml_models/generated/`.
- The next modeling plan is documented in
  `notes/NYC_LA_MODEL_TRAINING_PLAN.md`.

## Remaining

- Maintain folder-level README files in `data_source/source/<task>/` after
  commits, as required by `claude.md`.
- Harmonize AOI-selected building-footprint attributes into the common
  building-level schema once height-label processing begins.
- Inspect full-run rejected buildings and extreme residuals before moving to
  image-chip extraction and model training.
- Visually inspect `*_footprint_merge_review_sample.gpkg` for NYC and LA as an
  additional quality check on the merged-footprint layer.
