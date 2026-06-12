# Predicting Building Height from Satellite Imagery

## Goal

This project develops a reproducible framework for estimating building height
from satellite imagery. It combines building footprints, administrative
records, official 3D city models, LiDAR-derived measurements, and remotely
sensed features to produce and validate building-level height predictions
across cities with different urban forms and data environments.

The main unit of observation is:

```text
city x building
```

The project is intended to:

- harmonize measured and derived building-height labels across cities;
- train satellite-based machine-learning models to predict building height;
- evaluate transfer to cities with limited height data;
- compare predictions with existing global building-height products;
- quantify prediction uncertainty and error by place and building type; and
- produce maps and indicators for research in urban economics and policy.

The longer-run objective is a worldwide, building-level height dataset.

## Data

### Current Cities

The active project directory contains AOIs and data folders for 27 cities.
The final estimation and presentation sample will be selected according to
label quality, geographic coverage, temporal compatibility, urban-form
diversity, and data licensing.

| Region | Cities |
|---|---|
| North America | Chicago, Los Angeles, Montreal, New York City, Seattle |
| Latin America | Bogota, Buenos Aires, Caracas, Guadalajara, Medellin, Quito, Santiago de Chile, Sao Paulo |
| Europe | Amsterdam, Helsinki, Lyon, Marseille, Paris, Rotterdam, Utrecht, Vienna, Zurich |
| Africa | Cape Town, Nairobi |
| Asia | Jakarta, Singapore, Tokyo |

### Principal Data Components

| Component | Purpose | Examples |
|---|---|---|
| Building footprints | Define the building-level observation and spatial mask | Administrative footprints, 3DBAG, BD TOPO, Project PLATEAU, OpenStreetMap, Microsoft and Google Open Buildings |
| Height labels | Train and validate the models | Administrative roof height, official 3D geometry, LiDAR-derived height, photogrammetric height, and floor-count proxies |
| Satellite imagery | Supply predictive optical and shadow features | PlanetScope Surface Reflectance, supplemented where useful by Sentinel-1 and Sentinel-2 |
| Elevation data | Derive or validate roof-to-ground height | LiDAR point clouds, DSMs, DTMs, and local elevation products |
| Benchmark products | Provide external comparisons, not ground truth | Microsoft TEMPO, GlobalBuildingAtlas, WSF3D, GHS-BUILT-H, Overture, and OpenBuildingMap |

Raw imagery and large source datasets will not be stored in this public
repository. The repository will contain code, data documentation, acquisition
manifests, reproducible processing instructions, validation summaries, and
lightweight derived results that can be redistributed legally.

## Public LiDAR Availability

The following cities in the active sample have a public LiDAR source or a
public building product explicitly derived from LiDAR. Availability does not
guarantee that the source is citywide, current, or immediately suitable for
building-height estimation. Each source must be checked for coverage,
classification, acquisition year, vertical datum, and licensing.

| City or cities | Public source | Available product | Planned role | Access status |
|---|---|---|---|---|
| New York City | [NYS GIS LiDAR](https://gis.ny.gov/lidar), [USGS 3DEP](https://www.usgs.gov/3d-elevation-program), and NOAA topobathymetric LiDAR | Point clouds and elevation products | Validate official NYC roof-height records and test the DSM-DTM pipeline | Confirmed public |
| Los Angeles | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | Point clouds and elevation products where covered | Independent validation for LARIAC photogrammetric heights | Confirmed public; coverage must be checked |
| Chicago | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) Cook County products | Elevation rasters and potentially classified point clouds | Derive roof-to-ground heights after verifying that both surface and terrain information are available | Confirmed public; product type under review |
| Seattle | [King County LiDAR](https://www5.kingcounty.gov/lidar/) and [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | Point clouds, terrain, and elevation products | Derive building heights with explicit terrain correction | Confirmed public |
| Amsterdam, Rotterdam, Utrecht | [Actueel Hoogtebestand Nederland](https://www.ahn.nl/open-data) via AHN/PDOK | National point clouds, DSM, and DTM; 3DBAG is derived from BAG and AHN | Validate 3DBAG and independently derive selected heights | Confirmed public |
| Paris, Lyon, Marseille | [IGN LiDAR HD](https://geoservices.ign.fr/lidarhd) | Classified national point clouds and derived elevation products | Validate BD TOPO heights and measure roof-to-ground height | Confirmed public; local completion must be checked |
| Helsinki | [Helsinki laser-scanning datasets](https://hri.fi/data/en_GB/dataset/helsingin-laserkeilausaineistot) | Municipal point clouds and elevation products | Validate the Helsinki 3D city model | Confirmed public |
| Sao Paulo | [GeoSampa / Sao Paulo LiDAR](https://registry.opendata.aws/pmsp-lidar/) | Open point clouds plus MDS/MDT products | Derive citywide building-height labels | Confirmed public |
| Cape Town | City of Cape Town open building-footprint service | Public footprints and `BLD_HGHT` values derived from municipal LiDAR | Use as LiDAR-derived validation labels | Public derived product; raw point-cloud access not yet confirmed |
| Guadalajara | Municipal LiDAR portals and Visor Urbano | 2017/2019 point-cloud visualization and 3D cadastral context | Potential Latin American training and validation source | Publicly documented; bulk download and license still to be confirmed |
| Buenos Aires | Buenos Aires open-data and 3D-city resources | LAS data for selected public buildings; citywide 3D fabric is a separate source | Limited LiDAR validation and schema checks | Public but partial |
| Tokyo | Tokyo Digital Twin, national point-cloud resources, and Project PLATEAU | Point-cloud and official 3D city-model products | Prefer PLATEAU labels; use point clouds for selected validation | Public availability identified; exact coverage and download workflow under review |

Cities not listed in this table currently have no confirmed public LiDAR source
in the project catalog. They may still have administrative heights, official
3D models, floor counts, photogrammetric data, research labels, or global
footprint coverage.

## Methods

### 1. Define Comparable Areas of Interest

Maintain a documented administrative boundary for each city and construct a
comparable analysis AOI around the central business district. The intended
analysis radius is 20-25 km, subject to data coverage and advisor review.
AOIs are stored in WGS84 for imagery searches and in an appropriate local
projected CRS for distance, area, and buffer calculations.

### 2. Harmonize Footprints and Height Labels

Convert source data to a common building-level schema containing:

```text
building_id
city
country
geometry
footprint_area_m2
height_m
height_definition
height_source
source_date
confidence_tier
usable_for_training
usable_for_validation
uncertainty_m
```

Height definitions remain explicit. Administrative roof height, CityGML
measured height, geometric roof-to-ground height, LiDAR DSM-DTM height, and
floor-derived height are not treated as interchangeable without documentation.

### 3. Acquire Satellite Observations

The primary imagery source is PlanetScope Surface Reflectance, preferably
8-band SuperDove imagery. For each city, the search prioritizes cloud-free
observations close to the date of the height labels. The project will test
whether usable early- and late-day observations exist on the same day or on
nearby dates; acquisition time and sun geometry will always be retained.

Sentinel-1 SAR and Sentinel-2 optical imagery may provide reproducible
supplementary features.

### 4. Construct Building-Level Features

Features will be aggregated within each footprint and its surrounding context:

- spectral bands and indices;
- roof and neighborhood texture;
- shadow length, orientation, and contrast;
- sun elevation and azimuth;
- Sentinel-1 VV/VH backscatter and seasonal composites;
- footprint area, perimeter, compactness, and orientation;
- surrounding built density and neighboring-building morphology; and
- terrain, slope, and elevation context.

### 5. Estimate Models

The first model is a tabular gradient-boosting baseline using LightGBM or
XGBoost. It predicts a continuous building height from geometry, spectral,
shadow, SAR, terrain, and neighborhood features.

The main model will evaluate a multimodal image architecture that combines
satellite chips with building-footprint masks and tabular context. Candidate
architectures include convolutional neural networks and transformer-based
encoders such as SegFormer. Quantile models or ensembles will provide
prediction intervals.

### 6. Benchmark Existing Products

Microsoft TEMPO, GlobalBuildingAtlas, WSF3D, GHS-BUILT-H, Overture, and
OpenBuildingMap will be spatially matched to the validation buildings.
Their errors will be reported using the same AOIs, labels, height definitions,
and metrics as the project models. Predicted global products are benchmarks or
weak labels, never ground truth.

### 7. Validate Spatially

Random building splits can leak neighborhood and imagery information.
Evaluation will therefore use:

- spatially separated folds within cities;
- complete city-held-out tests;
- transfer tests across regions and income groups;
- separate evaluation for buildings above 30 m, 55 m, and 100 m; and
- visual audits of footprint alignment, shadows, vegetation, label quality,
  and imagery-label date mismatch.

Primary metrics are MAE, RMSE, median absolute error, bias, and R-squared.

### 8. Produce Research Outputs

Planned outputs include:

- per-building measured and predicted heights;
- prediction intervals and quality flags;
- height and residual maps;
- city height distributions and percentiles;
- built volume and tall-building shares;
- height gradients from central business districts; and
- documented comparisons across cities and benchmark products.

## Project Status

Data collection and source conversion are underway. The current work focuses
on finalizing the city sample, harmonizing source-specific height definitions,
selecting PlanetScope observations, and constructing the first cross-city
training table.
