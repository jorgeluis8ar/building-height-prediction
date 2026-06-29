# LiDAR Building-Height Process

## Objective

Create reproducible building-level reference heights from USGS 3DEP point
clouds for New York City and Los Angeles.

The output should be described as **LiDAR-derived building height**, not
perfect or error-free "true height." LiDAR has measurement error, roofs can be
partly occluded, building footprints and LiDAR may be from different years,
and the definition of height varies between sources.

The primary project definition is:

```text
building_height_m =
robust_roof_elevation_m - local_ground_elevation_m
```

Both elevations must use the same horizontal CRS, vertical datum, geoid, and
linear unit.

## Available Inputs

### New York City

- LiDAR projects:
  - `NY_New_York_CMGP_SANDY_LiDAR_15`
  - `NJ_New_Jersey_SANDY_LiDAR_15`
- Collection period: August 2013 through April 2014.
- Quality level: QL2.
- Horizontal CRS: EPSG:6347, NAD83(2011) / UTM zone 18N.
- Vertical CRS: EPSG:5703, NAVD88 height.
- Geoid: GEOID12A.
- Footprints: 46,744 polygons.
- Independent footprint attributes:
  - `HEIGHT_ROO`: official roof-height field, apparently in feet.
  - `GROUND_ELE`: official ground-elevation field, apparently in feet.
  - `CONSTRUCTI`: construction year.

### Los Angeles

- LiDAR project: `CA_LosAngeles_B23`.
- Collection period: January 2023 through January 2024.
- Quality level: QL1.
- Horizontal CRS: EPSG:6340, NAD83(2011) / UTM zone 11N.
- Vertical CRS: EPSG:5703, NAVD88 height.
- Geoid: GEOID18.
- Footprints: 79,645 polygons.
- Independent footprint attributes:
  - `HEIGHT`: LARIAC building-height field, apparently in feet.
  - `ELEV`: LARIAC elevation field, apparently in feet.
  - `DATE_`, `SOURCE`, and `STATUS`: source-vintage and status information.

The apparent feet units must be confirmed from the original city metadata
before validation. Do not infer units only from value ranges.

## Important Classification Finding

Representative downloaded tiles were inspected directly.

- Ground points are class 2.
- Most urban surface returns are class 1, unclassified.
- Noise and non-building classes include:
  - class 7: low noise;
  - class 9: water;
  - class 17: bridge deck; and
  - class 18: high noise.
- The inspected tiles did not contain building class 6.

Therefore, the workflow cannot depend on class 6. Roof candidates must be
selected spatially with the building footprints.

## Processing Stages

### 1. Inventory and Validate Source Tiles

Read only the LAZ files listed in `usgs_3dep_tile_manifest.csv`.

For every tile:

1. Confirm that file size and SHA-256 match the manifest.
2. Record LAS version, point format, scale, bounds, CRS, point count, and class
   counts.
3. Reject or flag a tile when its CRS conflicts with the selected USGS project
   metadata.
4. Exclude local `.copc.laz` variants that are not in the reviewed manifest.

Output:

```text
data_source/data/height_labels/generated/lidar_tile_inventory.csv
```

### 2. Harmonize Footprints

For each city:

1. Read the 5 km clipped footprint GeoPackage.
2. Repair invalid polygons.
3. Remove empty geometries and exact duplicates.
4. Reproject footprints to the LiDAR horizontal CRS.
5. Preserve the original footprint geometry and source attributes.
6. Create an inward-buffered roof sampling geometry:
   - default erosion: 1 meter;
   - use 0.5 meter for narrow buildings that would otherwise disappear;
   - flag buildings that cannot support an inward buffer.
7. Create a local ground ring:
   - begin 1 meter outside the footprint;
   - extend to 5 meters outside the footprint;
   - subtract all neighboring building footprints from the ring.

The inward buffer reduces mixed roof/facade points. Removing neighboring
footprints prevents nearby roofs from contaminating local ground estimates.

### 3. Create a Ground Surface

Use only valid class 2 points. Exclude withheld points and noise classes.

Recommended method:

1. Create a 1-meter DTM in the native LiDAR CRS.
2. Aggregate class 2 elevations with a robust statistic, preferably the cell
   median.
3. Fill small urban gaps only within a bounded search distance, initially
   5 meters.
4. Record interpolation distance and whether a cell was directly observed.

For each building:

1. Sample DTM cells in the building's ground ring.
2. Require spatial support around multiple sides of the building, not only one
   cluster of points.
3. Fit a robust local plane when terrain slope is meaningful:

```text
ground_z(x, y) = a + b*x + c*y
```

4. Evaluate the fitted ground plane at the building centroid or across the
   footprint.
5. Use the median ground-ring elevation when plane fitting is unstable.

Store:

```text
ground_elevation_m
ground_method
ground_point_count
ground_ring_coverage
ground_slope
ground_interpolated_fraction
```

### 4. Create a Surface Model

Roof candidates should include class 1 and class 6 when present.

Exclude:

- class 2 ground;
- class 7 low noise;
- class 9 water;
- class 17 bridge deck;
- class 18 high noise;
- withheld points; and
- overlap points when duplicated flight-line observations create artifacts.

Create a 0.5- or 1-meter DSM using a high cell percentile rather than the raw
maximum. A cell-level 95th percentile is a reasonable initial choice.

Calculate the normalized surface:

```text
nDSM = DSM - DTM
```

Retain a direct-observation mask so interpolation is never confused with an
observed roof return.

### 5. Calculate Building-Level Roof Statistics

Extract nDSM values only inside the inward-buffered footprint.

Store several statistics:

```text
height_p50_m
height_p75_m
height_p90_m
height_p95_m
height_max_clean_m
roof_pixel_count
roof_coverage_fraction
roof_height_iqr_m
```

Recommended initial primary estimate:

```text
height_m = height_p90_m
```

The 90th percentile is high enough to represent the upper roof while being
less sensitive than the maximum to antennas, HVAC equipment, birds, noise,
and tree branches. This choice must be validated rather than assumed final.

For complicated or pitched roofs, also retain:

- median height as a typical-roof measure;
- 95th percentile as a ridge/top-roof measure; and
- within-footprint height spread as a roof-complexity indicator.

### 6. Detect Vegetation and Mixed Returns

Because building returns are often class 1, footprint overlap alone cannot
guarantee that every high point is roof.

Flag likely contamination when:

- the vertical spread within small cells is unusually high;
- multiple-return structure suggests vegetation;
- isolated high cells lack neighboring support;
- the high surface occupies only a small share of the footprint; or
- p95 is much larger than p50 without a spatially coherent roof region.

Use connected surface regions or a neighborhood filter to remove isolated
high pixels. Do not silently delete a building when contamination is
suspected; retain it with a lower confidence tier.

### 7. Apply Minimum Quality Rules

Initial rejection rules:

- fewer than 10 valid roof cells;
- roof coverage below 30% of the inward-buffered footprint;
- fewer than 10 observed ground cells in the local ring;
- ground-ring coverage below 25%;
- estimated height below 2 meters or above 300 meters;
- more than 50% of the required ground surface interpolated; or
- footprint and LiDAR dates indicate that the building did not yet exist.

Suggested confidence tiers:

| Tier | Conditions | Use |
|---|---|---|
| A | Roof coverage >=70%, strong local ground support, little interpolation, spatially coherent roof | Training and validation |
| B | Roof coverage >=40%, adequate ground support, moderate interpolation or roof complexity | Training with caution |
| C | Meets minimum thresholds but has temporal, ground, vegetation, or edge concerns | Validation sensitivity only |
| Reject | Fails minimum support or plausibility rules | Do not use |

Thresholds should be finalized after examining empirical distributions.

### 8. Handle Temporal Mismatch

LiDAR and footprint dates are not interchangeable.

New York City:

- Flag footprints with `CONSTRUCTI` later than the 2014 LiDAR collection.
- Compare footprint editing/status dates with the LiDAR period.

Los Angeles:

- The selected LiDAR is from 2023-2024, while the footprint source is LARIAC
  2020 with some older source vintages.
- Flag demolished, modified, or newly constructed buildings when source fields
  indicate mismatch.

Store:

```text
lidar_collect_start
lidar_collect_end
footprint_source_date
temporal_mismatch_flag
```

### 9. Validate the Height Definition

The existing official height attributes are valuable validation data, not
inputs to the LiDAR estimate.

1. Confirm official units and definitions from source documentation.
2. Convert feet to meters only after confirmation:

```text
height_m = height_ft * 0.3048
```

3. Compare p50, p75, p90, and p95 LiDAR estimates with official heights.
4. Select the primary percentile using a calibration subset.
5. Report accuracy on a separate held-out subset.
6. Stratify errors by:
   - building height;
   - footprint area;
   - roof complexity;
   - point density;
   - acquisition/footprint time gap;
   - confidence tier; and
   - city.

Report at least:

```text
MAE
RMSE
median_error
median_absolute_error
bias
R_squared
```

Map residuals to identify datum errors, tile seams, terrain failures, and
neighborhood-specific bias.

### 10. Final Building-Level Output

Write one GeoPackage and one lightweight Parquet/CSV table per city under:

```text
data_source/data/height_labels/generated/<city_slug>/
```

Minimum schema:

```text
building_id
city_slug
geometry
height_m
height_definition
ground_elevation_m
roof_elevation_m
height_p50_m
height_p75_m
height_p90_m
height_p95_m
roof_point_or_pixel_count
roof_coverage_fraction
ground_point_or_pixel_count
ground_ring_coverage
ground_method
lidar_project
lidar_collect_start
lidar_collect_end
horizontal_crs
vertical_crs
geoid
temporal_mismatch_flag
vegetation_or_mixed_return_flag
tile_edge_flag
confidence_tier
usable_for_training
usable_for_validation
```

The `height_definition` value should be explicit, for example:

```text
lidar_ndsm_roof_p90_minus_local_ground
```

## Recommended Implementation Order

1. Implement tile and classification inventory.
2. Implement DTM creation and inspect it visually.
3. Implement DSM/nDSM creation and inspect roof/vegetation behavior.
4. Process a small, diverse sample of approximately 500 buildings per city.
5. Compare p50-p95 estimates against official city height fields.
6. Freeze the estimator and quality thresholds.
7. Process all buildings.
8. Run spatial and statistical validation.

Do not process all 126,000+ footprints before the small-sample diagnostic
stage. A vertical-datum, unit, or roof-statistic mistake would otherwise be
expensive and difficult to diagnose.
