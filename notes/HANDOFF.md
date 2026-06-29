# Project Handoff

Last updated: 2026-06-24

This file is a handoff note for starting a new Codex/Claude conversation on
the `building-height-prediction` repository. It captures the project rules,
current state, scripts, data status, known warnings, and commit history needed
to continue without relying on the previous chat context.

## Start Here In A New Chat

Ask the new assistant to read these files before doing any work:

```text
README.md
claude.md
PROGRESS.md
notes/HANDOFF.md
data_source/source/city_aois/README.md
data_source/source/building_footprints/README.md
data_source/source/planet_imagery/README.md
data_source/source/height_labels/README.md
```

Recommended opening instruction:

```text
Use the repo at:
/Users/jorgeochoa/Library/CloudStorage/Dropbox-Brown/Jorge Ochoa/Research/building-height-prediction

Before doing anything, read README.md, claude.md, PROGRESS.md, and
notes/HANDOFF.md. Follow the project rules in claude.md exactly.
```

## Repository

- Repository path:
  `/Users/jorgeochoa/Library/CloudStorage/Dropbox-Brown/Jorge Ochoa/Research/building-height-prediction`
- Remote repository:
  `jorgeluis8ar/building-height-prediction`
- Main branch:
  `main`
- Public repository warning:
  never commit API keys, OAuth tokens, virtual environments, raw imagery, raw
  LiDAR, or large source datasets.

## Critical Project Rules

These rules come from `claude.md` and should govern all future work.

- Fail loud. Do not allow hidden partial failures.
- Log honestly. A failed or partial run must never look clean.
- Use relative paths inside code, resolved from the detected repository root.
- Do not hard-code absolute local paths in scripts.
- Do not manually modify raw files in any `data_source/data/<domain>/source/`
  folder.
- Write pipeline outputs needed downstream to
  `data_source/data/<domain>/generated/`.
- Keep task-specific documentation updated in
  `data_source/source/<task>/README.md`.
- Prefer Python for new, separable work unless another language is requested.
- Do not run Git commits unless the user explicitly asks.
- Use local virtual environments plus `requirements.txt`, not committed venvs.

## Current City Sample

The active sample contains 29 cities selected because they have ready, partial,
promising, or otherwise strong open LiDAR access:

```text
Amsterdam
Barcelona
Birmingham
Boston
Buenos Aires
Cape Town
Chicago
Copenhagen
Guadalajara
Helsinki
Hong Kong
London
Los Angeles
Lyon
Madrid
Manchester
Marseille
Montreal
New York City
Oslo
Paris
Rotterdam
San Francisco
Sao Paulo
Seattle
Utrecht
Valencia
Vancouver
Zurich
```

City slugs are lowercase with underscores, for example `new_york_city` and
`los_angeles`.

## Folder Structure

The current project is organized around `data_source/source/<task>/` scripts
and `data_source/data/<domain>/` data. Major active folders are:

```text
data_source/source/city_aois/
data_source/source/building_footprints/
data_source/source/planet_imagery/
data_source/source/height_labels/

data_source/data/city_aois/
data_source/data/building_footprints/
data_source/data/planet_imagery/
data_source/data/height_labels/
data_source/data/elevation/
data_source/data/predictions/
data_source/data/validation/
data_source/data/benchmark_products/
```

City-specific `source/<city_slug>/` and `generated/<city_slug>/` folders exist
under the main city-varying data domains.

## Commit History

Current Git history at the time of this handoff:

```text
6407c9e Add reproducible Planet imagery ordering workflow
9529481 Create 5km building footprint processing workflow
ec9b967 Create 5km city AOIs from UN WUP city centers
9b7c0b1 update after setting up claude agent and docker container
957a921 Add initial project README
```

## Completed Work

### Repository setup

- Reset the project around `README.md` and `claude.md`.
- Created the base folder structure required by `claude.md`.
- Added `.gitignore` rules for data, virtual environments, caches, and local
  secrets.
- Added `PROGRESS.md` as the running project log.

### City AOIs

Script:

```text
data_source/source/city_aois/create_city_buffers.py
```

Key outputs:

```text
data_source/data/city_aois/generated/cities_sample.csv
data_source/data/city_aois/generated/city_buffers_5km.geojson
data_source/data/city_aois/generated/city_buffers_5km_by_city/<city_slug>_5km.geojson
```

Current status:

- The script uses the UN WUP 2018 file
  `WUP2018-F22-Cities_Over_300K_Annual_V7.xls` as the city center/CBD source.
- It rebuilds `cities_sample.csv` from the `README.md` Current Cities section.
- It writes one combined 5km AOI GeoJSON and one city-specific 5km GeoJSON per
  city.
- A local venv exists at
  `data_source/source/city_aois/venv_city_aois/`.
- Dependencies are pinned in
  `data_source/source/city_aois/requirements.txt`.
- The script automatically relaunches inside its venv.

### Building footprints

Script:

```text
data_source/source/building_footprints/clip_building_footprints.py
```

Key output pattern:

```text
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg
```

Current status:

- Raw building-footprint files were added by city under
  `data_source/data/building_footprints/source/<city_slug>/`.
- The clipping script clips each city's raw footprints to its 5km AOI.
- All 29 current cities have non-empty clipped GeoPackage outputs.
- A local venv exists at
  `data_source/source/building_footprints/venv_building_footprints/`.
- Dependencies are pinned in
  `data_source/source/building_footprints/requirements.txt`.
- The script automatically relaunches inside its venv.

Important known footprint-height fields:

- New York City clipped footprints include `HEIGHT_ROO`.
- Los Angeles clipped footprints include `HEIGHT`.
- These are independent official/administrative height fields for validation.
  They should not be used to construct the LiDAR-derived labels.

### Planet imagery

Main scripts:

```text
data_source/source/planet_imagery/search_planet_city_scenes.py
data_source/source/planet_imagery/select_planet_city_scenes.py
data_source/source/planet_imagery/order_selected_planet_scenes.py
data_source/source/planet_imagery/download_ordered_planet_scenes.py
data_source/source/planet_imagery/download_selected_planet_scenes.py
```

Key generated files:

```text
data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv
data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv
data_source/data/planet_imagery/generated/planet_orders_manifest.csv
```

Source imagery output pattern:

```text
data_source/data/planet_imagery/source/<city_slug>/<season>_<scene_id>/
```

Current status:

- Planet search uses the 29 current cities from `README.md`.
- Planet search uses city-specific 5km AOI buffers.
- Planet search requires at least 95% AOI coverage.
- Scene selection chooses two reviewed scenes per current city.
- Most cities use `ortho_analytic_8b_sr`; Oslo is the known exception using
  `ortho_analytic_4b_sr`.
- Ordering and downloading are intentionally split:
  - `order_selected_planet_scenes.py` creates AOI-clipped Planet orders and
    writes/updates the manifest.
  - `download_ordered_planet_scenes.py` checks order status and downloads only
    completed orders.
  - `download_selected_planet_scenes.py` is a deprecated guard only. Do not
    restore all-in-one order/download behavior.
- `PLANET_API.py` is local-only and must remain ignored by Git.
- Scripts resolve paths from the detected repository root and reject paths
  outside the repository.
- A Boston test order was downloaded successfully and moved into the correct
  repository path after an earlier path bug.
- Remaining Planet orders were created after user permission. Check
  `planet_orders_manifest.csv` for exact status and order IDs before any
  download work.

Planet safety rules:

- Always run `--dry-run` before `--confirm-order`.
- Always run `--dry-run` before `--confirm-download`.
- Do not activate, order, or download Planet assets without explicit user
  permission.
- Never print, commit, or copy the Planet API key.

### USGS 3DEP LiDAR

Script:

```text
data_source/source/height_labels/download_usgs_3dep_lidar.py
```

Key outputs:

```text
data_source/data/height_labels/generated/usgs_3dep_projects.csv
data_source/data/height_labels/generated/usgs_3dep_tile_manifest.csv
data_source/data/height_labels/source/<city_slug>/usgs_3dep/<project_name>/
```

Current U.S. project selection:

```text
Boston:        MA_CentralEastern_2021_B21
Chicago:       IL_4_County_QL1_LiDAR_2016_B16
Los Angeles:   CA_LosAngeles_B23
New York City: NY_New_York_CMGP_SANDY_LiDAR_15
New York City: NJ_New_Jersey_SANDY_LiDAR_15
San Francisco: CA_SanFrancisco_B23
Seattle:       WA_KingCounty_2021_B21
```

Discovery status:

- The official National Map products API and 3DEP Elevation Index were queried.
- 789 unique AOI-intersecting LAZ tiles were identified across Boston,
  Chicago, Los Angeles, New York City, San Francisco, and Seattle.
- Estimated compressed download size for all six U.S. cities is 85.09 GiB.
- No downloads should be attempted without explicit user permission.

Download status:

- New York City is downloaded and verified:
  68 selected tiles, 2.24 GiB.
- Los Angeles is downloaded and verified:
  99 selected tiles, 7.45 GiB.
- Together, the selected NYC/LA manifest rows contain 167 verified files,
  totaling about 9.70 GiB.
- Boston, Chicago, San Francisco, and Seattle remain pending.

Known LiDAR warning:

- There are five extra NYC `.copc.laz` variant files present locally that are
  not part of the reviewed 167-tile NYC/LA manifest. Do not use the extra COPC
  files unless they are explicitly added to the manifest and documented.

Representative LiDAR metadata inspected:

- New York City sample:
  - LAS 1.2, point format 1.
  - Project metadata indicates horizontal EPSG:6347, vertical EPSG:5703,
    GEOID12A.
  - Ground is class 2.
  - No building class 6 was found in the inspected sample.
- Los Angeles sample:
  - LAS 1.4, point format 6.
  - CRS parsed as NAD83(2011) / UTM zone 11N plus NAVD88 Geoid18 meters.
  - Ground is class 2.
  - No building class 6 was found in the inspected sample.

The key implication is that roof extraction must be footprint-masked. Do not
assume classified building points exist.

### LiDAR building-height process

Process document:

```text
data_source/source/height_labels/LIDAR_BUILDING_HEIGHT_PROCESS.md
```

Recommended height definition:

```text
building_height_m = robust_roof_elevation_m - local_ground_elevation_m
```

Recommended primary estimator for the first prototype:

```text
height_definition = lidar_ndsm_roof_p90_minus_local_ground
height_m = height_p90_m
```

Recommended process:

1. Inventory source tiles and validate CRS, classes, point counts, and bounds.
2. Harmonize footprints; repair geometries, reproject, create inward roof
   masks, and create local ground rings.
3. Build a DTM from class 2 ground points.
4. Build a DSM/nDSM from footprint-masked non-ground surface points.
5. Compute building-level roof percentiles and coverage metrics.
6. Flag vegetation and mixed-return contamination.
7. Assign quality tiers.
8. Handle temporal mismatch between LiDAR, footprints, and official heights.
9. Validate against `HEIGHT_ROO` in NYC and `HEIGHT` in Los Angeles.
10. Only after a diagnostic sample works, run the full city pipeline.

Strong recommendation:

- Start with about 500 diverse buildings per city for New York City and Los
  Angeles before processing all buildings.
- Compare p50, p75, p90, p95, and max-clean roof-height definitions against
  official height fields.
- Do not call the output perfect "true height"; call it a LiDAR-derived
  building-height label with a documented definition and quality tier.

## Virtual Environments

Current local venvs:

```text
data_source/source/city_aois/venv_city_aois/
data_source/source/building_footprints/venv_building_footprints/
data_source/source/planet_imagery/venv_planet_imagery/
data_source/source/height_labels/venv_height_labels/
```

These venv folders should not be committed. Recreate them from the
corresponding `requirements.txt` files when working on another machine.

For the height-label venv, Python 3.14 required this environment variable when
installing `lazrs==0.7.0`:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

Check the folder-specific README before recreating each venv.

## Current Best Next Step

The next technical step should be a small LiDAR-derived height prototype, not a
full-city run.

Recommended task:

```text
Create a diagnostic height-label script for New York City and Los Angeles that
processes a sample of about 500 buildings per city, computes LiDAR-derived
roof-to-local-ground height candidates, and compares them against official
height fields.
```

Suggested output location:

```text
data_source/data/height_labels/generated/<city_slug>/
```

Suggested diagnostic outputs:

```text
lidar_tile_inventory.csv
building_height_diagnostics_sample.csv
height_definition_comparison.csv
quality_tier_summary.csv
```

Suggested future script name:

```text
data_source/source/height_labels/derive_lidar_building_heights.py
```

Before implementing, inspect
`data_source/source/height_labels/LIDAR_BUILDING_HEIGHT_PROCESS.md`.

## Open Risks And Checks

- Confirm units and metadata for `HEIGHT_ROO` in NYC and `HEIGHT` in Los
  Angeles before interpreting validation error. These are likely in feet, but
  this must be confirmed from source metadata.
- Confirm vertical datums and units for each LiDAR project before mixing
  values across projects.
- Avoid using official height fields to construct LiDAR labels.
- Avoid processing all buildings until the diagnostic sample has been reviewed.
- Keep all Planet API material out of Git.
- Keep raw imagery and raw LiDAR out of Git.
- When moving or downloading files, verify that paths stay under the
  `building-height-prediction` repository, not the sibling `SUMMER 2026 RA`
  folder.

