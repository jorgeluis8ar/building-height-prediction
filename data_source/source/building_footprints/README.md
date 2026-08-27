# Building Footprint Processing

This folder contains the script that turns raw city building footprints into
project-ready 5km AOI footprint files.

## Scripts

```bash
python3 data_source/source/building_footprints/clip_building_footprints.py
```

`download_ms_buildings_us_training_aois.py` acquires Microsoft Global ML
Building Footprints for the 52 USGS training cities that do not already have a
verified Boston or Seattle footprint layer. It computes level-9 quadkeys
locally, downloads only the required public United States partitions, clips
each footprint exactly to the WUP 5 km AOI, and writes the filename and folder
expected by `run_us_lidar_to_planet_ndsm.py`:

```text
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg
```

Each GeoPackage uses the city AOI's declared CRS. The script fails when an AOI
has no CRS and verifies output CRS equality before marking a city complete.
Boundary validation permits only reprojection slivers within 5 cm of the AOI
edge; geometry extending materially beyond that buffered edge remains a hard
failure. The same validation is applied when resuming from an existing output.
Microsoft's model-derived `height` property is intentionally excluded to avoid
height-label leakage; geometry, confidence, source release, and quadkey
provenance are retained.

The Microsoft release is pinned by a dated link-index URL. The current default
is `2026-07-24`. Raw gzip partitions are cached without modification under:

```text
data_source/data/building_footprints/source/ms_buildings_us/<release_date>/<quadkey>/
```

The complete metadata-only regression found all 52 cities, 66 unique
partitions, and approximately 2.668 GB of compressed source data. It downloaded
no footprint partitions.

On Windows CMD, first plan a single pilot city:

```cmd
data_source\source\building_footprints\venv_building_footprints\Scripts\python.exe data_source\source\building_footprints\download_ms_buildings_us_training_aois.py ^
  --dry-run ^
  --city-slug birmingham_22936
```

Download, clip, and validate that pilot with four partition workers:

```cmd
data_source\source\building_footprints\venv_building_footprints\Scripts\python.exe data_source\source\building_footprints\download_ms_buildings_us_training_aois.py ^
  --confirm-download ^
  --city-slug birmingham_22936 ^
  --download-workers 4 ^
  --minimum-free-gb 100
```

If a previous pilot reached GeoPackage validation but did not finish, rerun it
with `--overwrite`. Valid cached Microsoft partitions are reused, while the
temporary GeoPackage is safely recreated.

After visually inspecting the pilot, plan and run all 52 cities with:

```cmd
data_source\source\building_footprints\venv_building_footprints\Scripts\python.exe data_source\source\building_footprints\download_ms_buildings_us_training_aois.py ^
  --dry-run ^
  --city-limit 0

data_source\source\building_footprints\venv_building_footprints\Scripts\python.exe data_source\source\building_footprints\download_ms_buildings_us_training_aois.py ^
  --confirm-download ^
  --city-limit 0 ^
  --download-workers 8 ^
  --minimum-free-gb 100
```

The city and partition manifests are written to:

```text
data_source/data/building_footprints/generated/ms_buildings_us_training_city_manifest.csv
data_source/data/building_footprints/generated/ms_buildings_us_partition_manifest.csv
```

Do not launch concurrent copies of the script. `--download-workers` provides
safe parallel partition transfers inside one process; city clipping, output
writing, and manifest updates remain coordinated.

After the 5km-intersecting footprint files are generated, contiguous footprint
diagnostics can be generated with:

```bash
python3 data_source/source/building_footprints/merge_contiguous_footprints.py \
  --city los_angeles \
  --city new_york_city \
  --no-source-id-gate \
  --same-height-tolerance-m 0.5
```

The scripts automatically relaunch inside:

```text
data_source/source/building_footprints/venv_building_footprints/
```

## Inputs

- `README.md`
- `data_source/data/building_footprints/source/<city_slug>/`
- `data_source/data/city_aois/generated/city_buffers_5km_by_city/<city_slug>_5km.geojson`

## Outputs

Every city is written in the same GeoPackage format:

```text
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg
```

These files are selected by AOI intersection, not clipped geometrically. If any
part of a source building polygon intersects the 5km city buffer, the whole
original polygon is retained. Each output row includes:

```text
aoi_selection_rule = intersects_5km_aoi_preserve_full_geometry
```

The run also writes:

```text
data_source/data/building_footprints/generated/building_footprints_clip_summary.csv
```

The merge diagnostic writes, for each selected city:

```text
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_merged_5km.gpkg
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_footprint_merge_crosswalk.csv
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_footprint_merge_diagnostics.csv
data_source/data/building_footprints/generated/<city_slug>/<city_slug>_footprint_merge_review_sample.gpkg
```

and the cross-city summary:

```text
data_source/data/building_footprints/generated/building_footprints_merge_summary.csv
```

The merged GeoPackage and diagnostics CSV include official height and
ground/elevation summaries inherited from the source footprints. For NYC,
`HEIGHT_ROO` and `GROUND_ELE` are converted from feet to meters. For LA,
`HEIGHT` and `ELEV` are converted from feet to meters. Each merged component
stores:

```text
official_height_mean_m
official_height_median_m
official_height_min_m
official_height_max_m
official_height_area_weighted_m
official_ground_mean_m
official_ground_median_m
official_ground_min_m
official_ground_max_m
official_ground_area_weighted_m
```

The `*_source` and `*_n` columns identify the original source field and the
number of source polygons contributing non-missing values.

## Contiguous Footprint Merge Diagnostics

The merge script is a diagnostic/consolidation tool for cases where a physical
building is split across multiple adjacent polygons. Outputs stay in the same
city-specific `building_footprints/generated/<city_slug>/` folder because they
are derived footprint layers, not a new raw data domain.

The default script behavior is intentionally conservative: it requires a
meaningful shared boundary and, when source building IDs are useful, matching
source IDs. For NYC and LA, source IDs appear to identify individual polygon
features rather than shared building groups, so the current diagnostic run uses
shared-boundary merging plus an official-height similarity gate of 0.5 meters.

Current NYC/LA diagnostic result:

| City | Original polygons | Merged polygons | Merged components | Largest component |
|---|---:|---:|---:|---:|
| Los Angeles | 79,645 | 74,795 | 3,610 | 20 polygons |
| New York City | 46,744 | 32,578 | 6,821 | 21 polygons |

These merged layers should not become canonical until the
`*_footprint_merge_review_sample.gpkg` files have been visually inspected in
QGIS or another GIS viewer. NYC still has many attached-building merge
candidates, so manual review is especially important before rerunning LiDAR
labels on merged footprints.

## Recreate the Virtual Environment

From the repository root:

```bash
python3 -m venv data_source/source/building_footprints/venv_building_footprints
data_source/source/building_footprints/venv_building_footprints/bin/python -m pip install -r data_source/source/building_footprints/requirements.txt
```

On Windows, use:

```bat
python -m venv data_source\source\building_footprints\venv_building_footprints
data_source\source\building_footprints\venv_building_footprints\Scripts\python.exe -m pip install -r data_source\source\building_footprints\requirements.txt
```

## Notes

- Raw files in `source/` are read-only and are never modified.
- Generated files are overwritten only inside each city's `generated/` folder.
- The script fails loudly if a current city has no readable source data, no AOI,
  or no footprints intersecting the 5km AOI.
- Do not overwrite the original AOI-selected footprint layer with the merged layer.
  Keep the crosswalk so labels and model predictions can be mapped back to the
  original footprint IDs.
