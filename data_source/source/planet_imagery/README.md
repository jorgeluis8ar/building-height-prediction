# Planet Imagery

This folder contains scripts for searching Planet metadata, selecting reviewed
city scenes, creating AOI-clipped Planet orders, and downloading completed
orders.

## Global Selection Analysis

`analyze_planet_global_scene_selection.py` is a local, API-free analysis of
the completed global scene selection. It overlays all 1,862 WUP city AOI
centroids and the selected Planet scene centroids on a world map, produces
country-level city/scene counts, summarizes numeric and categorical metadata,
and creates detailed AOI/scene-footprint maps for Aba, Tokyo, and Buenos Aires.

Install the updated visualization dependencies and run from Windows CMD:

```bat
python -m pip install -r data_source\source\planet_imagery\requirements.txt
python data_source\source\planet_imagery\analyze_planet_global_scene_selection.py
```

The default outputs are written to:

```text
data_source/data/planet_imagery/generated/global_scene_selection_analysis/
├── global_aoi_and_scene_centroids.png
├── country_city_scene_summary.csv
├── selected_scene_numeric_metadata_summary.csv
├── selected_scene_categorical_metadata_counts.csv
├── selected_scene_acquisition_year_summary.csv
├── detailed_city_maps/
│   ├── aba_21974_selected_scene_map.png
│   ├── tokyo_21671_selected_scene_map.png
│   └── buenos_aires_20058_selected_scene_map.png
└── logs/analyze_planet_global_scene_selection_<UTC timestamp>.log
```

The detailed maps draw the exact scene polygons stored in
`scene_geometry_geojson` and identify their `strip_id`. The Planet metadata
does not contain a separate full-strip polygon, so the script does not claim
to reconstruct geometry beyond the selected scene footprints. Override the
three default cities by repeating `--detail-city <city_slug>` exactly three
times.

## Reproducible Global City Split

`create_planet_global_city_splits.py` assigns every city represented in the
global selected-scenes file to one mutually exclusive group. The production
defaults use seed `419453` and exact counts of 711 training, 711 validation,
and 357 testing cities. A stable SHA-256 score based on `seed:city_slug`
guarantees identical results across computers and Python/Pandas versions.
Every selected scene inherits its city's group, preventing city leakage.

Run locally from Windows CMD; no Planet authorization is required:

```bat
python data_source\source\planet_imagery\create_planet_global_city_splits.py
```

Outputs are written under
`data_source/data/planet_imagery/generated/global_scene_selection_split/`.
`planet_scene_split_manifest.csv` contains all scenes and their assignments;
`training_scene_order_input.csv` contains the full training rows intended for
the first future order stage; and the three `*_scene_ids.csv` files provide
compact per-group ID manifests. The script only writes local CSV/log files—it
cannot activate, order, or download imagery. Cities with fewer than nine real
selected scenes are retained and explicitly flagged rather than being padded
with invented IDs.

## Global Training Orders and Downloads

The global workflow uses city-level, resumable scripts separate from the
legacy 29-city order/download programs:

- `order_planet_training_city_scenes.py` plans exactly 711 clipped city
  orders covering the 6,350 available training scenes. Each request groups
  that city's scenes by 8-band or 4-band surface-reflectance bundle. Its local
  `--dry-run` does not authenticate or contact Planet. Confirmed submission is
  bounded to 25 cities by default, checkpoints after every city, and searches
  for the deterministic order name before creating anything so interrupted
  reruns do not duplicate orders.
- `download_planet_training_city_orders.py` checks completed city orders and
  downloads at most 10 cities per call by default. It accepts only `success`,
  blocks `partial` for manual review, checks free disk space, and marks a city
  downloaded only after all files named by Planet's delivered `manifest.json`
  exist locally.

Both scripts reject validation/testing rows, use relative repository paths,
write dated logs, and require explicit confirmation flags for external writes.
They never order or download merely because the script is launched.

First create and review the complete local plan from Windows CMD:

```bat
python data_source\source\planet_imagery\order_planet_training_city_scenes.py --dry-run
```

Then submit one small test city and inspect the Planet portal/manifest before
larger batches:

```bat
python data_source\source\planet_imagery\order_planet_training_city_scenes.py ^
  --confirm-order --city-offset 0 --city-limit 1
```

After that order reaches `success`, check it without downloading and then
download it explicitly:

```bat
python data_source\source\planet_imagery\download_planet_training_city_orders.py ^
  --dry-run --city-offset 0 --city-limit 1
python data_source\source\planet_imagery\download_planet_training_city_orders.py ^
  --confirm-download --city-offset 0 --city-limit 1
```

The shared manifest is
`data_source/data/planet_imagery/generated/global_training_orders/planet_training_city_orders_manifest.csv`.
Downloaded city folders are stored under
`data_source/data/planet_imagery/source/global_training/<city_slug>/`.
Increase offsets only after reviewing the previous batch. A zero city limit
means all remaining cities and should not be used until the pilot succeeds.

## Global Nine-Scene Selection

`select_planet_global_city_scenes.py` selects up to nine reviewed metadata
rows per WUP global city. It never activates assets, creates orders, or
downloads imagery.

The selector applies these documented rules:

- `quality_category == standard`;
- Northern Hemisphere summer is June-July and winter is December-January;
- Southern Hemisphere summer is December-January and winter is June-July;
- maximize distinct acquisition years before using repeated-year fallbacks;
- target five summer and four winter scenes;
- target two scenes each from the north, south, east, and west scene-centroid
  bearing sectors, with the ninth direction selected by the overall score;
- apply coverage/cloud tiers in this order: 100%/0%, at least 99.5%/0%, at
  least 99.5%/at most 1%, then at least 95% with the lowest available cloud;
- maximize the minimum sun-elevation separation among selected scenes;
- prefer high absolute `view_angle` values; and
- require RGB+NIR surface reflectance, preferring
  `ortho_analytic_8b_sr` and falling back to `ortho_analytic_4b_sr`.

Asset-list calls are metadata-only and are cached in
`planet_scene_asset_availability.csv`. Failed asset calls cause a nonzero exit;
successful checks and completed city outputs remain resumable.

Test one city from Windows CMD after authenticating with `planet auth login`:

```bat
python data_source\source\planet_imagery\select_planet_global_city_scenes.py ^
  --city-slug aba_21974 ^
  --overwrite
```

Process a bounded batch:

```bat
python data_source\source\planet_imagery\select_planet_global_city_scenes.py ^
  --city-offset 0 ^
  --city-limit 25
```

Advance the offset by 25. Existing per-city selection and summary files are
skipped unless `--overwrite` is supplied. The production outputs are:

```text
data_source/data/planet_imagery/generated/global_scene_selection/
├── selected_global_planet_city_scenes.csv
├── global_scene_selection_city_summary.csv
├── global_scene_selection_shortfalls.csv
├── planet_scene_asset_availability.csv
├── by_city/<city_slug>_selected_planet_scenes.csv
└── by_city_summary/<city_slug>_selection_summary.csv
```

Every selected row records its rank, hemisphere, local season, year,
four-sector direction, filter tier, repeated-year status, target-match flags,
sun-diversity gain, asset availability, selected asset type, and serialized
score components. City summaries expose selection shortfalls rather than
silently claiming that infeasible targets were met.

`--skip-asset-check` exists only for offline code testing. It marks outputs as
unverified and those outputs must never be passed to ordering.

## Training LiDAR-year eight-scene selection

`select_planet_scenes_for_training_lidar_years.py` is a separate selector for
the 94 cities that are both in the Planet training split and ready for open
LiDAR. It selects up to eight scenes independently for each defensible LiDAR
acquisition year and never overwrites the original 1,862-city selection.

The scene rules remain standard quality, hemisphere-aware June-July and
December-January solstice seasons, the established AOI/cloud tiers, RGB+NIR
surface reflectance with 8-band preference and 4-band fallback, sun-elevation
diversity, high absolute view angle, and low haze/shadow/snow. Each city-year
targets four summer and four winter scenes and two scenes in each of the north,
south, east, and west scene-centroid sectors.

The temporal rules are intentionally strict. Exact semicolon-separated years
are processed separately from 2016 onward. Two-year flight/project ranges use
both endpoint years. Broad campaign, catalogue, or national ranges are not
expanded into unsupported annual dates. Pre-2016 years are reported and never
replaced with later imagery.

After Planet authentication, test one eligible city from Windows CMD:

```bat
python data_source\source\planet_imagery\select_planet_scenes_for_training_lidar_years.py ^
  --city-slug boston_22939 ^
  --overwrite
```

Run all 94 input cities, including explicit temporal exclusions:

```bat
python data_source\source\planet_imagery\select_planet_scenes_for_training_lidar_years.py ^
  --city-offset 0 ^
  --city-limit 0
```

Outputs are written under:

```text
data_source/data/planet_imagery/generated/training_lidar_year_scene_selection/
├── selected_training_lidar_year_planet_scenes.csv
├── training_lidar_year_eligibility.csv
├── training_lidar_year_scene_selection_summary.csv
├── training_lidar_year_scene_selection_shortfalls.csv
├── planet_scene_asset_availability.csv
├── by_city_year/<city_slug>_<year>_selected_planet_scenes.csv
└── by_city_year_summary/<city_slug>_<year>_selection_summary.csv
```

The script performs selection and asset-list metadata checks only. It does not
activate, order, or download Planet imagery or LiDAR.

## Scripts

Search metadata only:

```bash
python3 data_source/source/planet_imagery/search_planet_city_scenes.py
```

Search metadata for the 1,862-city WUP 2018 global sample in resumable batches:

```bash
python3 data_source/source/planet_imagery/search_planet_global_city_scenes.py \
  --city-offset 0 \
  --city-limit 25
```

Advance `--city-offset` by 25 for each batch, or use repeated `--city-slug`
arguments for targeted runs. A zero `--city-limit` processes every city after
the offset, but small batches are recommended because the full search spans
many cities and annual API windows.

The global search applies the established discovery filters by default:

- Planet permission filter;
- `cloud_cover < 0.30`; and
- calculated 5km AOI coverage of at least 95 percent.

### Combine all global query metadata

`combine_planet_global_city_scene_metadata.py` combines the 1,862 per-city
query files into one CSV without contacting Planet:

```bat
python data_source\source\planet_imagery\combine_planet_global_city_scene_metadata.py
```

The complete output is:

```text
data_source/data/planet_imagery/generated/global_city_scene_metadata/all_global_planet_city_scene_metadata.csv
```

The script validates the inventory count, expected filename for every city,
city slug inside every row, required metadata columns, and uniqueness of
`scene_id` within each city. It builds a union of columns, appends
`source_metadata_file` for provenance, streams rows to avoid holding the full
global table in memory, and replaces the output atomically. By default, any
missing city file is fatal. `--allow-missing-city-files` is available only for
an explicitly partial diagnostic output, which is labeled `PARTIAL` in its
run log.

To combine only the queried scenes for the 94 training/open-LiDAR cities, use
their detailed city list as the inventory and require exactly 94 files:

```bat
python data_source\source\planet_imagery\combine_planet_global_city_scene_metadata.py ^
  --inventory data_source\data\height_labels\generated\training_open_lidar\training_cities_with_open_lidar.csv ^
  --expected-city-count 94 ^
  --output data_source\data\planet_imagery\generated\training_lidar_year_scene_selection\all_94_training_lidar_city_scene_metadata.csv
```

This command includes every queried scene for those cities, not only the final
eight-per-LiDAR-year selections.

It writes one atomic CSV per city plus `search_window_manifest.csv` under
`data_source/data/planet_imagery/generated/global_city_scene_metadata/`.
Successful city/year windows are skipped on rerun. Failed windows are marked
failed and cause a nonzero exit, while completed data remain resumable.

Each scene includes flattened acquisition and quality fields, its full
footprint coordinates, bounding box, centroid, AOI centroid, centroid offset
distance/direction, and AOI coverage. `properties_json`, `item_links_json`,
and `item_json` preserve the complete API response so uncommon or future
Planet metadata fields are not discarded. This is metadata-only: it never
activates, orders, or downloads imagery.

Search NYC/LA metadata only with Planet `view_angle` included in a separate output:

```bash
python3 data_source/source/planet_imagery/search_planet_nyc_la_view_angle_scenes.py \
  --start-date 2010-01-01 \
  --end-date 2026-07-21
```

This writes:

```text
data_source/data/planet_imagery/generated/nyc_la_scenes_results_planet_with_view_angle.csv
```


Targeted 2010-2015 metadata backfill:

```bash
python3 data_source/source/planet_imagery/search_planet_city_scenes.py \
  --start-date 2010-01-01 \
  --end-date 2016-01-01 \
  --output data_source/data/planet_imagery/generated/cities_scenes_results_planet_2010_2015_backfill.csv
python3 data_source/source/planet_imagery/merge_planet_scene_backfill.py
```

Select two reviewed scenes per city:

```bash
python3 data_source/source/planet_imagery/select_planet_city_scenes.py
```

Select NYC/LA scenes aligned to the USGS LiDAR capture windows:

```bash
python3 data_source/source/planet_imagery/select_lidar_aligned_planet_scenes.py
```

Select two additional NYC/LA scenes between the current LiDAR-aligned scene
pair, prioritizing sun-elevation diversity:

```bash
python3 data_source/source/planet_imagery/select_intermediate_sun_elevation_scenes.py
```

Create Planet orders after review:

```bash
python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --dry-run
python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --confirm-order
```

Download completed Planet orders:

```bash
python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --dry-run
python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --confirm-download
```

The old all-in-one script is intentionally deprecated:

```bash
python3 data_source/source/planet_imagery/download_selected_planet_scenes.py
```

It exits with instructions and does not create orders or download data.

## Credentials

Planet authentication is checked in this order:

1. `data_source/source/planet_imagery/PLANET_API.py`
2. `PL_API_KEY` environment variable
3. Saved Planet OAuth profile

`PLANET_API.py` is intentionally ignored by Git. It should contain:

```python
PL_API_KEY = "your_key_here"
```

Never commit this file or paste the key into tracked scripts, notebooks,
README files, logs, or manifests.

## Source-Date Manifest

Scene selection uses:

```text
data_source/source/planet_imagery/building_footprint_source_dates.csv
```

This manifest records the building-footprint reference date used to center the
Planet scene selection, the dataset update date where known, source URLs, and
date-confidence notes.

## Selected Scene Output

The selector writes:

```text
data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv
```

The output contains two rows per city:

- `winter_jan_dec`
- `summer_jun_jul`

Each row keeps the original Planet scene metadata and adds selection fields
such as the footprint reference date, pair gap in days, and whether the strict
zero-cloud/100%-AOI condition was met.

The selector also checks Planet asset availability and writes:

```text
available_asset_types
has_ortho_analytic_8b_sr
selected_asset_type
selected_asset_type_reason
```

`selected_asset_type` is assigned consistently within each city pair. If both
selected scenes have `ortho_analytic_8b_sr`, both rows use it. If either scene
lacks it, both rows use the best shared fallback asset type. At this stage,
Oslo is the only reviewed city pair using `ortho_analytic_4b_sr`.

## LiDAR-Aligned NYC/LA Scene Output

The LiDAR-aligned selector writes:

```text
data_source/data/planet_imagery/generated/lidar_capture_summary_for_planet_selection.csv
data_source/data/planet_imagery/generated/selected_lidar_aligned_planet_scenes.csv
```

This output is separate from `selected_planet_city_scenes.csv`. It is centered
on the USGS 3DEP LiDAR collection windows rather than building-footprint source
dates, and currently contains only Los Angeles and New York City.

Current LiDAR capture windows:

| City | LiDAR collect start | LiDAR collect end | Midpoint |
|---|---:|---:|---:|
| Los Angeles | 2023-01-08 | 2024-01-07 | 2023-07-09 |
| New York City | 2013-08-06 | 2014-04-21 | 2013-12-13 |

Current LiDAR-aligned Planet selections:

| City | Season | Scene ID | Scene date | Relation to LiDAR |
|---|---|---|---:|---|
| Los Angeles | winter_jan_dec | `20231203_182937_07_2488` | 2023-12-03 | inside LiDAR window |
| Los Angeles | summer_jun_jul | `20230705_174134_45_245c` | 2023-07-05 | inside LiDAR window |
| New York City | winter_jan_dec | `20200122_154449_92_1061` | 2020-01-22 | after LiDAR window |
| New York City | summer_jun_jul | `20200614_155201_71_105e` | 2020-06-14 | after LiDAR window |

The NYC selections are not contemporaneous with the LiDAR capture because the
local Planet scene inventory begins after the 2013-2014 USGS Sandy LiDAR
collection. They are the nearest strict zero-cloud, full-AOI, standard-quality
winter/summer scenes available in the current Planet scene list.

A targeted 2010-2015 PSScene metadata backfill was run on 2026-07-01 using the
same cloud and AOI-coverage filters as the main scene search. It found six
qualifying rows across the full 29-city sample, all already present in the main
scene table, and no New York City rows. Therefore the NYC LiDAR-aligned
selection remains the 2020 winter/summer pair above.

## Intermediate Sun-Elevation Scene Output

The intermediate selector writes:

```text
data_source/data/planet_imagery/generated/selected_intermediate_planet_scenes.csv
data_source/data/planet_imagery/generated/intermediate_sun_elevation_scene_review.csv
```

This output is for the 12-channel HTC-DC Net experiment. It starts from the two
LiDAR-aligned scenes already used by the 6-channel model, keeps only strict
zero-cloud, full-AOI, standard-quality candidate scenes between those two
acquisitions, and chooses the pair that maximizes sun-elevation diversity
across the four total scenes for each city.

The selected scenes from the first run are:

| City | Role | Scene ID | Scene date | Sun elevation |
|---|---|---|---:|---:|
| Los Angeles | new intermediate | `20230713_182102_57_241c` | 2023-07-13 | 65.7 |
| Los Angeles | new intermediate | `20231203_171912_53_2445` | 2023-12-03 | 24.1 |
| New York City | new intermediate | `20200124_153319_56_1063` | 2020-01-24 | 26.1 |
| New York City | new intermediate | `20200526_155004_25_1058` | 2020-05-26 | 66.7 |

The order-compatible CSV should be reviewed and dry-run before any orders are
submitted:

```bash
python3 data_source/source/planet_imagery/order_selected_planet_scenes.py \
  --selected-scenes data_source/data/planet_imagery/generated/selected_intermediate_planet_scenes.csv \
  --dry-run
```

Only after explicit approval should the same command be run with
`--confirm-order`. Planet orders may remain queued or running for some time, so
the workflow stops after ordering and resumes later with
`download_ordered_planet_scenes.py --dry-run` once the manifest reports
downloadable order states.

## Order Manifest

The order script writes:

```text
data_source/data/planet_imagery/generated/planet_orders_manifest.csv
```

This manifest is the bridge between ordering and downloading. It records:

- city and season
- scene ID
- selected asset type
- Orders API product bundle
- Planet order name and order ID
- order state and timestamps
- intended output folder
- AOI path
- serialized order request
- download status and downloaded files

The order script does not wait for order completion and never downloads data.
This keeps Planet's asynchronous order processing separate from local file
downloads.

## Download Output

The download script reads `planet_orders_manifest.csv`, checks each order ID,
and downloads only orders with state `success` or `partial`. It skips orders
with state `queued`, `running`, `failed`, or any other non-downloadable state.

Downloaded raw Planet assets are written to:

```text
data_source/data/planet_imagery/source/<city_slug>/<season>_<scene_id>/
```

The script updates the manifest with `download_status`, `downloaded_files`,
`order_state`, and `download_checked_on`.

## Safety Rules

- All script defaults and manifest paths resolve from the
  `building-height-prediction` repository root, regardless of the terminal's
  current working directory.
- Manifest paths remain portable and repository-relative. Scripts resolve them
  from the detected repository root and reject paths outside the repository.
- Run `--dry-run` first for both ordering and downloading.
- Use `--confirm-order` only when you are ready to create Planet orders.
- Use `--confirm-download` only after the portal or manifest shows orders are complete.
- Do not rerun order creation blindly; use the manifest to avoid duplicate orders.
- Do not manually edit downloaded Planet imagery. Treat `source/` as raw data.

## Off-Nadir Candidate Ranking

`rank_planet_off_nadir_candidates.py` performs a metadata-only search for
scenes that cover a fixed city AOI even when the scene footprint is centered
some distance away. It does not order, activate, or download imagery.

For a ten-scene New York City prototype, run:

```bash
python3 data_source/source/planet_imagery/rank_planet_off_nadir_candidates.py \
  --city new_york_city \
  --top-n 10
```

The default filters require zero reported cloud cover, standard quality,
effectively 100% coverage of the 5km AOI, and at least 500m between the AOI
and the scene-footprint boundary. The ranking score gives 60% weight to
Planet's reported `view_angle`, 25% to scene-center offset, 10% to clear
percentage, and 5% to AOI boundary clearance. Scene-center offset is only a
supporting geometric indicator; `view_angle` is the direct off-nadir measure.

Outputs are written to:

```text
data_source/data/planet_imagery/generated/<city>_off_nadir_candidate_pool.csv
data_source/data/planet_imagery/generated/<city>_off_nadir_top10_scenes.csv
```

The output records acquisition time, view and illumination angles, scene/AOI
centroids, center-offset distance and direction, AOI coverage, edge clearance,
quality fields, and the final ranking score. Review this table before creating
any order-compatible selection file.
