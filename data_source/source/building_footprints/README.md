# Building Footprint Processing

This folder contains the script that turns raw city building footprints into
project-ready 5km AOI footprints.

## Script

```bash
python3 data_source/source/building_footprints/clip_building_footprints.py
```

The script automatically relaunches inside:

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

The run also writes:

```text
data_source/data/building_footprints/generated/building_footprints_clip_summary.csv
```

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
