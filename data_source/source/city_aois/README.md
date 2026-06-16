# city_aois Source README

This folder contains scripts for creating city area-of-interest files.

## create_city_buffers.py

### 2026-06-16 — Local virtual environment

- Created a local virtual environment for this task:
  - `data_source/source/city_aois/venv_city_aois/`
- Added `data_source/source/city_aois/requirements.txt` with the pinned package
  needed to read the WUP 2018 `.xls` workbook:
  - `xlrd==2.0.2`
- Updated `create_city_buffers.py` so it automatically relaunches itself using
  the local venv Python when run from another Python interpreter.
- Standard run command remains:
  - `python data_source/source/city_aois/create_city_buffers.py`
- If the virtual environment must be recreated, run:
  - `python3 -m venv data_source/source/city_aois/venv_city_aois`
  - `data_source/source/city_aois/venv_city_aois/bin/python -m pip install -r data_source/source/city_aois/requirements.txt`
- Added `venv_city_aois/` to `.gitignore` so the environment itself is not
  committed.

#### Recreate the virtual environment on another machine

The virtual environment folder is intentionally not committed to Git. To run
this task on another machine, recreate it from `requirements.txt`.

From the repository root on macOS or Linux:

```bash
python3 -m venv data_source/source/city_aois/venv_city_aois
data_source/source/city_aois/venv_city_aois/bin/python -m pip install -r data_source/source/city_aois/requirements.txt
python data_source/source/city_aois/create_city_buffers.py
```

From the repository root on Windows PowerShell:

```powershell
py -m venv data_source/source/city_aois/venv_city_aois
data_source/source/city_aois/venv_city_aois/Scripts/python.exe -m pip install -r data_source/source/city_aois/requirements.txt
python data_source/source/city_aois/create_city_buffers.py
```

The script automatically relaunches itself with the local venv Python, so the
final run command is the same after the environment has been created.

### 2026-06-16 — 29-city LiDAR-ready sample rerun

- Updated the script mappings for the 29-city LiDAR-ready sample now listed in
  the root `README.md`.
- Added WUP 2018 country/name mappings for newly added cities, including:
  - `Boston` -> `Boston`
  - `San Francisco` -> `San Francisco-Oakland`
  - `Vancouver` -> `Vancouver`
  - `Barcelona` -> `Barcelona`
  - `Birmingham` -> `Birmingham (West Midlands)`
  - `Copenhagen` -> `København (Copenhagen)`
  - `London` -> `London`
  - `Madrid` -> `Madrid`
  - `Manchester` -> `Manchester`
  - `Oslo` -> `Oslo`
  - `Valencia` -> `Valencia`
  - `Hong Kong` -> `Hong Kong`
- Updated the city-specific output step to remove old `.geojson` files before
  writing new ones. This prevents removed cities from lingering in
  `city_buffers_5km_by_city/`.
- Reran the script using WUP 2018 CBD coordinates. Verified outputs:
  - `cities_sample.csv`: 29 city rows
  - `city_buffers_5km.geojson`: 29 features
  - `city_buffers_5km_by_city/`: 29 single-city GeoJSON files

### 2026-06-16 — WUP 2018 CBD coordinates

- Updated the script to rebuild `cities_sample.csv` from:
  - `README.md`, using the `Current Cities` table
  - `data_source/data/city_aois/source/WUP2018-F22-Cities_Over_300K_Annual_V7.xls`
- The WUP 2018 file provides city-center/CBD coordinates using the columns:
  - `Country or area`
  - `Urban Agglomeration`
  - `Latitude`
  - `Longitude`
- Added city-name aliases for WUP 2018 agglomeration names, including:
  - `Los Angeles` -> `Los Angeles-Long Beach-Santa Ana`
  - `Marseille` -> `Marseille-Aix-en-Provence`
  - `New York City` -> `New York-Newark`
  - `Vienna` -> `Wien (Vienna)`
- Running the script now:
  1. reads the README current-city list
  2. matches those cities to WUP 2018 coordinates
  3. rewrites `data_source/data/city_aois/generated/cities_sample.csv`
  4. regenerates the combined 5km GeoJSON
  5. regenerates one 5km GeoJSON per city
- Dependency note: WUP 2018 is an old `.xls` workbook, so the script requires
  the Python package `xlrd` to read it.

### 2026-06-16

- Updated the script to use the current project paths:
  - input: `data_source/data/city_aois/generated/cities_sample.csv`
  - combined output: `data_source/data/city_aois/generated/city_buffers_5km.geojson`
- Added city-specific GeoJSON output files for Planet scene searches:
  - output folder: `data_source/data/city_aois/generated/city_buffers_5km_by_city/`
  - filename pattern: `<city_name>_5km.geojson`, using lowercase names with underscores.
- Preserved the main run order:
  1. check inputs
  2. create city centroids
  3. create 5km buffers
  4. save combined GeoJSON
  5. save one GeoJSON per city
  6. run optional visualization
- The script uses base Python only and writes GeoJSON directly, so it can run
  before a full geospatial environment is installed.
