# Height Labels

This folder contains reproducible acquisition and processing code for raw
height-label sources. Raw downloaded data belong under
`data_source/data/height_labels/source/<city_slug>/`; lightweight manifests and
derived outputs belong under `data_source/data/height_labels/generated/`.

The proposed process for converting classified point clouds into defensible
building-level roof-to-ground heights is documented in
`LIDAR_BUILDING_HEIGHT_PROCESS.md`.

## LiDAR-Derived Building Heights

`derive_lidar_building_heights.py` creates building-level height diagnostics
for New York City and Los Angeles from the downloaded USGS 3DEP LAZ tiles and
the generated 5 km clipped building footprints.

The default run is diagnostic-first:

```bash
python3 data_source/source/height_labels/derive_lidar_building_heights.py
```

By default, the script samples 500 buildings per city. This follows the process
recommendation to review a small sample before processing every footprint. To
change the sample size:

```bash
python3 data_source/source/height_labels/derive_lidar_building_heights.py \
  --city new_york_city \
  --sample-size 100
```

To process every footprint after the diagnostic run has been reviewed:

```bash
python3 data_source/source/height_labels/derive_lidar_building_heights.py \
  --all-buildings
```

The script writes:

```text
data_source/data/height_labels/generated/lidar_tile_inventory.csv
data_source/data/height_labels/generated/<city_slug>/building_height_diagnostics_sample.csv
data_source/data/height_labels/generated/<city_slug>/height_definition_comparison.csv
data_source/data/height_labels/generated/<city_slug>/quality_tier_summary.csv
data_source/data/height_labels/generated/<city_slug>/lidar_building_heights.gpkg
```

The primary first height definition is:

```text
height_definition = lidar_ndsm_roof_p90_minus_local_ground
height_m = height_p90_m
```

The output also retains `height_p50_m`, `height_p75_m`, `height_p95_m`, and
`height_max_clean_m` for comparison against independent official footprint
height fields. Official NYC `HEIGHT_ROO` and LA `HEIGHT` are used only for
diagnostics and validation. They are not used to create the LiDAR-derived
height estimates.

The script assumes official height fields are in feet unless another unit is
passed with `--official-height-units`. The output records
`official_height_units_confirmed=False` unless
`--official-units-confirmed` is passed. Confirm the source metadata before
treating official comparison errors as final.

The script rejects clear temporal mismatches, including NYC footprints with
`CONSTRUCTI` after the 2013-2014 LiDAR collection period.

### What `derive_lidar_building_heights.py` Does

The script follows these steps:

1. Relaunches inside the task-specific virtual environment at
   `data_source/source/height_labels/venv_height_labels/` so it uses the
   expected versions of `laspy`, `geopandas`, `shapely`, `pandas`, and related
   packages.
2. Reads command-line options. The most important options are `--city`,
   `--sample-size`, `--all-buildings`, `--official-height-units`,
   `--official-units-confirmed`, and `--skip-sha256`.
3. Creates a dated log file under
   `data_source/data/height_labels/generated/`. A failed run logs the failure
   and exits nonzero.
4. Reads the manifest
   `data_source/data/height_labels/generated/usgs_3dep_tile_manifest.csv`.
   This manifest defines the approved LAZ files and prevents unreviewed local
   files from entering the label pipeline.
5. Verifies each required LAZ file exists and matches the expected byte size.
   Unless `--skip-sha256` is passed, it also verifies the SHA-256 checksum
   recorded in the manifest.
6. Writes `lidar_tile_inventory.csv`, which records basic LAZ metadata such as
   point count, LAS version, point format, bounds, CRS, file size, and checksum
   status.
7. Loads the generated 5 km clipped building footprints for each city:
   `data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg`.
8. Checks that the expected official comparison fields are present. For New
   York City these include `HEIGHT_ROO`, `GROUND_ELE`, and `CONSTRUCTI`. For
   Los Angeles these include `HEIGHT`, `ELEV`, `DATE_`, and `STATUS`.
9. Selects either a diagnostic sample or all buildings. The default diagnostic
   sample is 500 buildings per city. The sample is stratified using official
   height when available, otherwise footprint area, so it contains small,
   medium, large, low-rise, and taller candidates.
10. Reprojects footprints from WGS84 into the city's LiDAR horizontal CRS:
    EPSG:6347 for New York City and EPSG:6340 for Los Angeles.
11. Repairs footprint geometries and creates an inward roof sampling polygon.
    The default inward buffer is 1 meter. If that erases a narrow building, the
    script tries 0.5 meters. If that still fails, it uses the original
    footprint.
12. Creates a local ground ring for each footprint. The ring begins 1 meter
    outside the building and extends to 5 meters outside the building.
    Neighboring building footprints are subtracted so nearby roofs do not
    contaminate the local ground estimate.
13. Reads only manifest-approved LiDAR tiles intersecting the selected
    buildings. For each point, it uses x, y, z, classification, and withheld
    flags.
14. Treats class 1 and class 6 points inside the inward footprint as roof
    candidates. Class 2 points inside the local ground ring are used as ground
    candidates. Noise, water, bridge deck, high-noise, and withheld points are
    excluded.
15. Estimates local ground elevation as the median class 2 elevation in the
    ground ring:

    ```text
    ground_elevation_m = median(local ground ring class 2 elevations)
    ```

16. Computes roof-height candidates by subtracting local ground from roof
    elevation percentiles:

    ```text
    height_p50_m = roof p50 elevation - local ground elevation
    height_p75_m = roof p75 elevation - local ground elevation
    height_p90_m = roof p90 elevation - local ground elevation
    height_p95_m = roof p95 elevation - local ground elevation
    height_max_clean_m = roof p99 elevation - local ground elevation
    ```

17. Sets the current diagnostic label to:

    ```text
    height_m = height_p90_m
    ```

    This is a working definition, not a final project decision.
18. Converts official footprint heights to meters. NYC `HEIGHT_ROO` and LA
    `HEIGHT` are in feet, so the diagnostic run uses:

    ```text
    official_height_m = official_height_ft * 0.3048
    ```

19. Compares the LiDAR-derived height with the official height using
    `lidar_minus_official_m` and `official_minus_lidar_m`.
20. Assigns a confidence tier:

    ```text
    A      strong roof and ground point support
    B      usable roof and ground point support
    C      weak support, useful mainly for validation sensitivity checks
    Reject not usable for training or validation
    ```

21. Rejects buildings with too few roof points, too few ground points, missing
    height, implausible height below 2 m or above 300 m, or clear temporal
    mismatch. For NYC, buildings with `CONSTRUCTI` after the 2013-2014 LiDAR
    collection are rejected.
22. Writes one CSV and one GeoPackage per city, plus comparison and quality
    summary tables.

### Diagnostic Metrics

The compact diagnostic table uses these columns:

| Column | Meaning |
|---|---|
| `City` | The city being evaluated. |
| `Sample` | Number of sampled buildings processed in the diagnostic run. This is 500 per city in the current diagnostic, not all buildings. |
| `Training usable` | Number of sampled buildings passing quality checks for model training. These are tier A or B buildings. |
| `p90 MAE` | Mean absolute error, in meters, comparing `height_p90_m` against official footprint height converted to meters. It is the average absolute size of the error. |
| `p90 RMSE` | Root mean squared error, in meters, comparing `height_p90_m` against official footprint height converted to meters. It penalizes large outliers more strongly than MAE. |
| `p90 bias` | Average signed error, in meters: `height_p90_m - official_height_m`. Positive values mean the LiDAR estimate is high on average; negative values mean it is low on average. |

Interpretation:

```text
MAE  = typical error size
RMSE = typical error size plus stronger penalty for large outliers
bias = average direction of the error
```

The current diagnostic outputs are:

```text
data_source/data/height_labels/generated/diagnostic_analysis/city_diagnostic_metric_summary.csv
data_source/data/height_labels/generated/diagnostic_analysis/city_height_summary_statistics.csv
data_source/data/height_labels/generated/diagnostic_analysis/los_angeles_lidar_vs_raw_height_scatter.png
data_source/data/height_labels/generated/diagnostic_analysis/new_york_city_lidar_vs_raw_height_scatter.png
```

### CRS Reference

The file below lists the AOI storage CRS, recommended local metric CRS, and
confirmed LiDAR CRS information where available:

```text
data_source/source/height_labels/city_crs_reference.csv
```

For the six U.S. cities with selected USGS 3DEP projects, the confirmed LiDAR
horizontal CRS, vertical CRS, geoid, project directories, and collection dates
come from `usgs_3dep_projects.csv`. For cities without confirmed LiDAR source
metadata yet, the file lists a recommended local UTM CRS for generic metric
geometry operations only. It should not be treated as the final LiDAR CRS for
those cities.

### Sampling Geometry Figure

A TikZ schematic of the footprint, inward roof sampling polygon, local ground
ring, LiDAR points, and height equation is stored at:

```text
data_source/source/height_labels/lidar_sampling_geometry_tikz.tex
```

The figure shows:

- the original building footprint;
- the 1 meter inward roof sampling polygon;
- the 1-5 meter local ground ring;
- roof candidate LiDAR points;
- ground-class LiDAR points; and
- the equation `height_p90 = roof_p90 - local_ground`.

### LiDAR Point Classifications

LAS/LAZ point classification codes follow the ASPRS LAS convention. Not every
dataset uses every class, and older LAS versions reserve or reinterpret some
codes. For this project, always inspect the actual class counts in each source
before assuming a class is populated.

| Class | Common meaning | Current project treatment |
|---:|---|---|
| 0 | Created, never classified | Not used directly |
| 1 | Unclassified | Used as roof candidate only inside the inward building footprint |
| 2 | Ground | Used for local ground elevation in the ground ring |
| 3 | Low vegetation | Excluded |
| 4 | Medium vegetation | Excluded |
| 5 | High vegetation | Excluded |
| 6 | Building | Used as roof candidate when present |
| 7 | Low point / low noise | Excluded |
| 8 | Model key point in older LAS; reserved in newer LAS 1.4 workflows | Not used directly |
| 9 | Water | Excluded |
| 10 | Rail | Excluded |
| 11 | Road surface | Excluded |
| 12 | Overlap in older workflows; reserved in LAS 1.4 | Not used directly |
| 13 | Wire guard / shield | Excluded |
| 14 | Wire conductor / phase | Excluded |
| 15 | Transmission tower | Excluded |
| 16 | Wire-structure connector | Excluded |
| 17 | Bridge deck | Excluded |
| 18 | High noise | Excluded |
| 19-63 | Reserved ASPRS classes | Not used directly |
| 64-255 | User-definable classes | Not used unless a source-specific metadata review documents them |

The inspected NYC and LA tiles contain ground points as class 2, but most
urban surface points are class 1 rather than class 6. That is why the script
uses the footprint geometry to identify roof candidates from class 1 points.

### Before Scaling To All Buildings

Do not immediately treat the 500-building diagnostic as final. Before running
`--all-buildings`, inspect:

- buildings rejected for missing roof or ground support;
- NYC outliers that create high RMSE despite low p90 bias;
- whether p75 or p90 is the better primary definition for NYC;
- LA high-error cases and implausible-height rejections;
- temporal mismatch flags; and
- plots comparing LiDAR height with official footprint height.

After that review, the next reasonable step is either:

1. refine the diagnostic estimator and rerun the 500-building sample; or
2. run `derive_lidar_building_heights.py --all-buildings` for NYC and LA.

## USGS 3DEP LiDAR

`download_usgs_3dep_lidar.py` discovers and downloads classified USGS 3DEP LAZ
tiles for the six U.S. cities in the current sample:

```text
boston
chicago
los_angeles
new_york_city
san_francisco
seattle
```

The script uses two official USGS services:

- The National Map products API supplies tile bounding boxes and LAZ URLs.
- The 3DEP Elevation Index supplies work-unit metadata, including acquisition
  dates, quality level, coordinate reference systems, and metadata links.

The full project tile set is not downloaded. Each tile bounding box must
intersect the city's actual 5 km GeoJSON AOI. New York City uses both the New
York and New Jersey Sandy LiDAR projects because its circular AOI crosses the
state boundary.

### Selected projects

| City | Selected USGS project directory | Reason |
|---|---|---|
| Boston | `MA_CentralEastern_2021_B21` | 2021 QL1 collection; complete AOI coverage |
| Chicago | `IL_4_County_QL1_LiDAR_2016_B16` | Cook County QL1 source; 97.63% AOI coverage, with the remainder over Lake Michigan |
| Los Angeles | `CA_LosAngeles_B23` | 2023 QL1 collection; complete AOI coverage |
| New York City | `NY_New_York_CMGP_SANDY_LiDAR_15` and `NJ_New_Jersey_SANDY_LiDAR_15` | Both projects are required for the cross-state AOI |
| San Francisco | `CA_SanFrancisco_B23` | 2023 QL0 collection; complete AOI coverage |
| Seattle | `WA_KingCounty_2021_B21` | 2021 QL1 collection; complete AOI coverage |

### Recreate the virtual environment

From the repository root:

```bash
python3 -m venv data_source/source/height_labels/venv_height_labels
data_source/source/height_labels/venv_height_labels/bin/python -m pip install \
  -r data_source/source/height_labels/requirements.txt
```

On Windows, replace `bin/python` with `Scripts/python.exe`.

Python 3.14 may require the following stable-ABI setting while building
`lazrs`:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
data_source/source/height_labels/venv_height_labels/bin/python -m pip install \
  -r data_source/source/height_labels/requirements.txt
```

### Discover tiles without downloading

Always run a dry-run first:

```bash
python3 data_source/source/height_labels/download_usgs_3dep_lidar.py \
  --dry-run \
  --estimate-sizes
```

This writes:

```text
data_source/data/height_labels/generated/usgs_3dep_projects.csv
data_source/data/height_labels/generated/usgs_3dep_tile_manifest.csv
```

### Download

Download every selected U.S. tile:

```bash
python3 data_source/source/height_labels/download_usgs_3dep_lidar.py \
  --confirm-download
```

Download one city:

```bash
python3 data_source/source/height_labels/download_usgs_3dep_lidar.py \
  --city boston \
  --confirm-download
```

The downloader uses atomic `.part` files, retries temporary network failures,
records SHA-256 checksums, and updates the tile manifest after every completed
tile. A rerun skips complete files and retries failed or partial downloads.

### Current download status

On 2026-06-18, the AOI-filtered LiDAR source tiles were downloaded and
verified for:

| City | Manifest tiles | Verified size |
|---|---:|---:|
| New York City | 68 | 2.24 GiB |
| Los Angeles | 99 | 7.45 GiB |
| **Total** | **167** | **9.70 GiB** |

Every selected tile matches its expected HTTP byte count and the SHA-256 value
recorded in `usgs_3dep_tile_manifest.csv`. No `.part` files remain.

Five additional New York `.copc.laz` variants are present locally for tile IDs
that intersect the AOI. They are not members of the reviewed 167-tile manifest
and must not be used by downstream processing unless they are explicitly added
to the manifest.

## Change Log

### 2026-06-18

- Added the USGS 3DEP project and tile discovery workflow.
- Added exact 5 km AOI tile filtering for all six U.S. cities.
- Added resumable, checksummed LAZ downloads and project/tile manifests.
- Downloaded and verified all selected New York City and Los Angeles tiles.
- Added recovery of full-size, checksum-matching `.part` files when cloud
  synchronization interrupts the final atomic rename.
- Added `LIDAR_BUILDING_HEIGHT_PROCESS.md` with the proposed roof-to-local-
  ground height definition, processing stages, quality controls, output
  schema, and validation strategy.
- Added pinned LAS/LAZ inspection dependencies.

### 2026-06-24

- Added `derive_lidar_building_heights.py` for NYC/LA LiDAR-derived
  building-height diagnostics.
- Added `geopandas==1.1.3` and `pyogrio==0.12.1` to the height-label
  requirements so the script can read and write GeoPackage footprint data.
- Smoke-tested the script with three buildings for New York City and three
  buildings for Los Angeles using `--skip-sha256`. Both runs completed and
  wrote the expected diagnostic CSV, comparison CSV, quality summary CSV,
  tile inventory CSV, and GeoPackage outputs.
- Confirmed that temporal mismatch logic rejects NYC buildings constructed
  after the 2013-2014 LiDAR acquisition period.
- Ran the default 500-building diagnostic for New York City and Los Angeles
  with official footprint height units confirmed as feet and converted to
  meters for comparison:
  - Los Angeles: 500 sampled buildings, 444 usable for training, p90 MAE
    1.287 m, p90 RMSE 2.486 m, p90 bias -0.377 m.
  - New York City: 500 sampled buildings, 464 usable for training, p90 MAE
    1.825 m, p90 RMSE 7.651 m, p90 bias 0.104 m.
  - Output log:
    `data_source/data/height_labels/generated/derive_lidar_building_heights_20260624T194808Z.log`.
- Expanded this README with a step-by-step explanation of
  `derive_lidar_building_heights.py`, the diagnostic metric definitions, and
  the LiDAR point-classification guide.
- Added `city_crs_reference.csv` with AOI CRS, recommended local metric CRS,
  and confirmed U.S. LiDAR CRS metadata.
- Added `lidar_sampling_geometry_tikz.tex` as a TikZ schematic of the inward
  roof sampling polygon, local ground ring, LiDAR points, and height equation.
- Generated diagnostic analysis tables and plots under
  `data_source/data/height_labels/generated/diagnostic_analysis/`.
- Added repeated random diagnostic sampling to
  `derive_lidar_building_heights.py` with `--sample-runs`. Diagnostic samples
  are now simple random draws with replacement by default, using fresh random
  seeds unless `--random-seed` is supplied for reproducibility.
- Ran 15 independent 500-building with-replacement diagnostic samples per city:
  - Los Angeles: 7,500 sampled rows, 7,165 unique buildings, 6,702
    training-usable rows, p90 MAE 1.216 m, p90 RMSE 2.667 m, p90 bias
    -0.236 m.
  - New York City: 7,500 sampled rows, 6,907 unique buildings, 6,989
    training-usable rows, p90 MAE 2.205 m, p90 RMSE 10.302 m, p90 bias
    -0.341 m.
  - Per-run temporary files were written under each city's `temp_samples/`
    folder and merged into `building_height_diagnostics_sample.csv`.
  - Output log:
    `data_source/data/height_labels/generated/derive_lidar_building_heights_20260624T214405Z.log`.
- Rewrote `LiDAR_diagnostics_verification.R` as a reproducible metric-selection
  analysis. It reads the merged 15 x 500 diagnostics and writes:
  - `diagnostic_analysis/metric_selection/raw_vs_lidar_height_summary_statistics.csv`
  - six city/metric scatter plots comparing LiDAR p90, p95, and pmax against
    raw footprint height
  - `diagnostic_analysis/metric_selection/rmse_by_city_metric_height_bin_sample_run.csv`
  - `diagnostic_analysis/metric_selection/rmse_by_height_bin_boxplot.png`

### 2026-06-25

- Updated `derive_lidar_building_heights.py` so every positive LiDAR-derived
  height below 2.4 m is assigned 2.4 m. This rule is applied consistently to
  p50, p75, p90, p95, pmax, and the current `height_m` label.
- Removed the prior upper-height rejection rule. Tall buildings are no longer
  rejected merely because the estimated LiDAR height exceeds 300 m.
- Reran 15 independent 500-building with-replacement diagnostic samples per
  city using reproducible seed `20260625`; run-specific seeds are
  `20260625` through `20260639`.
  - Los Angeles: 7,500 sampled rows, 7,172 unique buildings, 6,808
    training-usable rows, p90 MAE 1.207 m, p90 RMSE 3.537 m, p90 bias
    -0.271 m.
  - New York City: 7,500 sampled rows, 6,929 unique buildings, 6,999
    training-usable rows, p90 MAE 2.162 m, p90 RMSE 9.344 m, p90 bias
    -0.413 m.
  - Consistency audit found zero positive LiDAR heights below 2.4 m across
    p50, p75, p90, p95, pmax, and `height_m`.
  - Output log:
    `data_source/data/height_labels/generated/derive_lidar_building_heights_20260625T174809Z.log`.
- Regenerated the metric-selection outputs from
  `LiDAR_diagnostics_verification.R` using the seeded June 25 diagnostics.
- Updated `LiDAR_diagnostics_verification.R` to exclude nonpositive
  LiDAR-derived heights from metric-selection summaries, scatter plots, and
  RMSE calculations. The script writes
  `diagnostic_analysis/metric_selection/lidar_nonpositive_height_cleaning_audit.csv`
  so these exclusions remain visible.
- Made the final label schema explicit in `derive_lidar_building_heights.py`:
  `height_label_m` is the primary label and is copied from `height_p90_m`;
  `height_p95_m` and `height_max_m` are robustness labels; `local_ground_m`
  records the local ground estimate. The older `height_m` remains as a
  backward-compatible p90 alias.
- Ran the all-building LiDAR label pipeline for Los Angeles and New York City
  with official footprint height units confirmed as feet and converted to
  meters. The run used seed metadata `20260625` and wrote outputs separately
  from the diagnostic samples:
  - Los Angeles: 79,645 footprints, 71,889 training-usable labels, 74,334
    validation-usable labels.
  - New York City: 46,744 footprints, 43,486 training-usable labels, 44,713
    validation-usable labels.
  - Full output files:
    `data_source/data/height_labels/generated/<city>/building_height_labels_all.csv`,
    `height_definition_comparison_all.csv`, `quality_tier_summary_all.csv`,
    and `lidar_building_heights_all.gpkg`.
  - Output log:
    `data_source/data/height_labels/generated/derive_lidar_building_heights_20260625T182050Z.log`.
