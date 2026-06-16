# Project Progress

Last updated: 2026-06-16

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

## Current Status

- Repository scaffold is reset and ready for the next project task.
- City AOI centroid source files have been regenerated from WUP 2018.
- `README.md` now lists 29 current cities selected around LiDAR readiness.
- `cities_sample.csv` and city AOI 5km buffers are current for the 29-city
  LiDAR-ready sample using WUP 2018 CBD coordinates.
- The city AOI script can be run with
  `python data_source/source/city_aois/create_city_buffers.py`; it will use
  `data_source/source/city_aois/venv_city_aois/` automatically.
- No task-specific subfolders have been created inside `data_source/source/`,
  because `claude.md` says those listed subfolders are examples and should not
  be created yet.

## Remaining

- Define the first processing task.
- Create the matching `data_source/source/<task>/` subfolder only when that
  task is ready.
- Add task scripts with novice-friendly comments, explicit input checks,
  fatal failure behavior, and honest dated logs.
- Maintain folder-level README files in `data_source/source/<task>/` after
  commits, as required by `claude.md`.
