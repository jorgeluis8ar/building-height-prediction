# Windows US50 HTC Dataset Extension

This guide extends the validated 44-city HTC framework with six additional
U.S. LiDAR cities. It creates new versioned four-scene and eight-scene datasets
without modifying the existing `us_44_*` datasets.

The final city split is fixed at 22 training, 22 validation, and 6 test cities.
The original 44 assignments remain unchanged. New York-Newark, Miami, and
Philadelphia are added to training; Cincinnati, Harrisburg, and Trenton are
added to validation.

Run every command from Windows CMD at `S:\building-height-prediction`.

## 1. Pull The Source Code

Generated manifests, imagery, LiDAR, nDSMs, and model datasets are data and
must not be committed to Git. Git is used only for scripts and documentation.

```cmd
cd /d S:\building-height-prediction
git status
git fetch origin us-lidar-planet-ndsm
git switch us-lidar-planet-ndsm
git pull --ff-only origin us-lidar-planet-ndsm
git log -1 --oneline
```

Before pulling, `git status` should report a clean working tree. Confirm the
extension files exist:

```cmd
dir data_source\source\ml_models\prepare_us50_htc_extension_inputs.py
dir data_source\source\ml_models\prepare_us44_htc_framework.py
dir data_source\source\height_labels\run_us_lidar_to_planet_ndsm.py
dir data_source\source\ml_models\US50_HTC_DATASET_EXTENSION_WINDOWS.md
```

## 2. Verify The Six-City Planet Inputs

The manually transferred manifests must be stored at:

```text
data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\planet_orders_manifest.csv
data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv
```

The downloaded Planet products must remain under:

```text
data_source\data\planet_imagery\source\processed_us_lidar\<city_slug>\
```

Verify that the selection contains exactly eight unique scenes per city:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -c "import pandas as pd; p=r'data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv'; d=pd.read_csv(p,dtype=str); print('Cities:',d.city_slug.nunique()); print('Rows:',len(d)); print(d.groupby('city_slug').scene_id.nunique().to_string()); assert d.city_slug.nunique()==6 and len(d)==48 and d.groupby('city_slug').scene_id.nunique().eq(8).all()"
```

Confirm that each selected scene has one Surface Reflectance raster and one
eight-band UDM2 raster by running the HTC inventory later in this guide. That
inventory is the authoritative file-level check.

## 3. Verify The Height-Label Environment

```cmd
dir data_source\source\height_labels\venv_height_labels\Scripts\python.exe
data_source\source\height_labels\venv_height_labels\Scripts\python.exe -c "import geopandas,numpy,pandas,rasterio,pyproj,shapely; print('Height-label imports passed')"
```

The six-city nDSM operation also requires the existing USGS inventory, tile
metadata, AOIs, and building-footprint GeoPackages:

```cmd
dir data_source\data\height_labels\generated\training_open_lidar\training_cities_with_open_lidar.csv
dir data_source\data\height_labels\generated\usgs_3dep_global_city_tile_metadata.csv
dir data_source\data\building_footprints\generated
```

## 4. Dry Run The Six nDSMs

This stage validates the six AOIs, building footprints, selected Planet grids,
USGS LiDAR plans, and local scene files. It does not download LiDAR.

```cmd
data_source\source\height_labels\venv_height_labels\Scripts\python.exe data_source\source\height_labels\run_us_lidar_to_planet_ndsm.py ^
  --dry-run ^
  --selected-scenes data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv ^
  --planet-root data_source\data\planet_imagery\source\processed_us_lidar ^
  --city-slug new_york_newark_23083 ^
  --city-slug miami_23064 ^
  --city-slug philadelphia_23098 ^
  --city-slug cincinnati_22957 ^
  --city-slug harrisburg_23008 ^
  --city-slug trenton_23165 ^
  --city-limit 0
```

Do not continue if this reports a failed city. Inspect the newest log under:

```text
data_source\data\height_labels\generated\us_training_planet_ndsm\logs\
```

## 5. Build The Six Three-Band nDSMs

Check free space before downloading LiDAR:

```cmd
fsutil volume diskfree S:
```

Run the same bounded six-city operation in download/build mode. The command
retains raw LiDAR; it does not use `--confirm-delete-lidar`.

```cmd
data_source\source\height_labels\venv_height_labels\Scripts\python.exe data_source\source\height_labels\run_us_lidar_to_planet_ndsm.py ^
  --confirm-download ^
  --selected-scenes data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv ^
  --planet-root data_source\data\planet_imagery\source\processed_us_lidar ^
  --city-slug new_york_newark_23083 ^
  --city-slug miami_23064 ^
  --city-slug philadelphia_23098 ^
  --city-slug cincinnati_22957 ^
  --city-slug harrisburg_23008 ^
  --city-slug trenton_23165 ^
  --city-limit 0 ^
  --download-workers 1 ^
  --minimum-free-gb 100
```

The output for each city is a Planet-aligned three-band GeoTIFF:

1. continuous nDSM in metres;
2. building-footprint-masked nDSM in metres; and
3. LiDAR QA, where 0 is invalid, 1 is valid outside buildings, and 2 is a
   valid building pixel.

The HTC framework uses band 2 for `_AGL.tif` and derives `_BLG.tif` from the
building-valid QA pixels in band 3.

Verify the six manifest records and output paths:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -c "import pandas as pd; p=r'data_source\data\height_labels\generated\us_training_planet_ndsm\us_lidar_to_planet_ndsm_manifest.csv'; d=pd.read_csv(p,dtype=str).fillna(''); c=['new_york_newark_23083','miami_23064','philadelphia_23098','cincinnati_22957','harrisburg_23008','trenton_23165']; x=d[d.city_slug.isin(c)][['city_slug','status','output_ndsm_path','error_message']].sort_values('city_slug'); print(x.to_string(index=False)); assert len(x)==6 and x.status.str.startswith('complete_').all() and x.output_ndsm_path.ne('').all()"
```

## 6. Create The Authoritative US50 Inputs

This preserves all original US44 assignments and adds the requested six-city
assignments. It also merges the 44-city and six-city scene metadata.

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us50_htc_extension_inputs.py ^
  --existing-scene-metadata data_source\data\planet_imagery\generated\processed_us_lidar_scene_selection\selected_processed_us_lidar_planet_scenes.csv ^
  --new-scene-metadata data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv ^
  --existing-city-split data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\city_split_manifest.csv ^
  --output-dir data_source\data\ml_models\generated\htc_dc_net\us_50_extension_inputs_v1
```

Verify the resulting city and scene counts:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -c "import pandas as pd; r=r'data_source\data\ml_models\generated\htc_dc_net\us_50_extension_inputs_v1'; s=pd.read_csv(r+'\\selected_processed_us_lidar_planet_scenes_us50.csv',dtype=str); c=pd.read_csv(r+'\\city_split_manifest_us50.csv',dtype=str); print('Scene cities:',s.city_slug.nunique()); print('Scene rows:',len(s)); print(c.groupby('split').size().to_string()); print(c[c.city_slug.isin(['new_york_newark_23083','miami_23064','philadelphia_23098','cincinnati_22957','harrisburg_23008','trenton_23165'])][['city_slug','split']].sort_values('city_slug').to_string(index=False)); assert s.city_slug.nunique()==50 and len(s)==400 and c.groupby('split').size().to_dict()=={'test':6,'train':22,'val':22}"
```

## 7. Configure The US50 HTC Commands

Set reusable CMD variables once in the current terminal:

```cmd
set HTC_PY=data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe
set HTC_SCRIPT=data_source\source\ml_models\prepare_us44_htc_framework.py
set US50_ARGS=--ndsm-manifest data_source\data\height_labels\generated\us_training_planet_ndsm\us_lidar_to_planet_ndsm_manifest.csv --scene-metadata data_source\data\ml_models\generated\htc_dc_net\us_50_extension_inputs_v1\selected_processed_us_lidar_planet_scenes_us50.csv --scene-root data_source\data\planet_imagery\source\processed_us_lidar --expected-cities 50 --expected-scenes-per-city 8 --train-city-count 22 --val-city-count 22 --test-city-count 6 --city-split-manifest data_source\data\ml_models\generated\htc_dc_net\us_50_extension_inputs_v1\city_split_manifest_us50.csv --staging-name us_50_city_staging_v1 --dataset-4-name us_50_rgbnir_4scene_v1 --dataset-8-name us_50_rgbnir_8scene_v1
```

These variables last only for the current CMD window. Set them again after
opening a new terminal.

## 8. Run Inventory And Split

Inventory verifies all 50 nDSMs and all 400 SR/UDM2 scene pairs before any
chips are created:

```cmd
%HTC_PY% %HTC_SCRIPT% --stage inventory %US50_ARGS%
type data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\pipeline_status.json
type data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\missing_inputs.csv
```

The inventory must report exactly 50 complete cities. Then install the fixed
22/22/6 split:

```cmd
%HTC_PY% %HTC_SCRIPT% --stage split %US50_ARGS%
type data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\city_split_manifest.csv
```

## 9. Create Canonical Targets And Aligned Images

Canonical chips use each city's 3 m nDSM grid, fixed non-overlapping 256 by
256 windows, padded edges, and only geographic chips with valid buildings.
The alignment stage reprojects each RGB+NIR scene and UDM2 raster to those
exact windows.

```cmd
%HTC_PY% %HTC_SCRIPT% --stage canonical-chips --resume %US50_ARGS%
%HTC_PY% %HTC_SCRIPT% --stage align-scenes --resume %US50_ARGS%
```

Per-city staging outputs are written below:

```text
data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\<city_slug>\image\
data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\<city_slug>\mask\
data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\<city_slug>\ndsm\
data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\<city_slug>\qa\
```

## 10. Build The Four- And Eight-Scene Datasets

The four-scene dataset selects four independent observations per retained
geographic chip using the existing seasonal/directional rules and records all
fallbacks. The eight-scene dataset retains all eight independent observations.
Scenes are not stacked or averaged.

```cmd
%HTC_PY% %HTC_SCRIPT% --stage build-datasets --resume %US50_ARGS%
%HTC_PY% %HTC_SCRIPT% --stage statistics %US50_ARGS%
%HTC_PY% %HTC_SCRIPT% --stage validate --qa-chips-per-city 3 %US50_ARGS%
```

The final datasets are:

```text
data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_4scene_v1\
data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_8scene_v1\
```

Each contains `image`, `mask`, `ndsm`, `qa`, split lists, manifests, training-
city statistics, alignment diagnostics, exclusions, and visual QA panels.

## 11. Validate The Final Infrastructure

```cmd
type data_source\data\ml_models\generated\htc_dc_net\us_50_city_staging_v1\pipeline_status.json
dir data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_4scene_v1
dir data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_8scene_v1
dir data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_4scene_v1\visual_qa
dir data_source\data\ml_models\generated\htc_dc_net\us_50_rgbnir_8scene_v1\visual_qa
```

Confirm split counts and observation-file completeness in both datasets:

```cmd
%HTC_PY% -c "from pathlib import Path; import pandas as pd; root=Path(r'data_source\data\ml_models\generated\htc_dc_net'); names=['us_50_rgbnir_4scene_v1','us_50_rgbnir_8scene_v1']; [(lambda d,m: print(d.name,'rows=',len(m),'cities=',m.city_slug.nunique(),'splits=',m.drop_duplicates('city_slug').groupby('split').size().to_dict(),'missing=',sum(not (d/k/f'{o}{s}').is_file() for o in m.observation_id.astype(str) for k,s in [('image','_IMG.tif'),('mask','_BLG.tif'),('ndsm','_AGL.tif')]))))(root/n,pd.read_csv(root/n/'chips_manifest.csv',low_memory=False)) for n in names]"
```

Expected city split counts are `train=22`, `val=22`, and `test=6`; `missing`
must be zero. Successful validation is indicated by `"status": "success"` in
`pipeline_status.json`.

## 12. Resume And Recovery

The expensive stages are independently resumable. Rerun an interrupted stage
with the same `US50_ARGS` and `--resume`:

```cmd
%HTC_PY% %HTC_SCRIPT% --stage canonical-chips --resume %US50_ARGS%
%HTC_PY% %HTC_SCRIPT% --stage align-scenes --resume %US50_ARGS%
%HTC_PY% %HTC_SCRIPT% --stage build-datasets --resume %US50_ARGS%
```

Do not delete the existing `us_44_*` outputs. Do not use `--overwrite` unless
an output is known to be invalid and a deliberate rebuild is required. After
rebuilding chips or datasets, rerun `statistics` and `validate`.

The complete framework prepares data only. It does not start model training or
cross-validation.
