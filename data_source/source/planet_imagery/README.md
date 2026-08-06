# Planet Imagery

This folder contains scripts for searching Planet metadata, selecting reviewed
city scenes, creating AOI-clipped Planet orders, and downloading completed
orders.

## Scripts

Search metadata only:

```bash
python3 data_source/source/planet_imagery/search_planet_city_scenes.py
```

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
