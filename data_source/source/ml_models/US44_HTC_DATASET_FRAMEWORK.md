# Windows US44 HTC Dataset Framework

`prepare_us44_htc_framework.py` builds four-scene and eight-scene RGB+NIR HTC
datasets for the 44 processed U.S. LiDAR cities. It prepares data only; it does
not train a model or run cross-validation.

Run from Windows Command Prompt at `S:\building-height-prediction` using the
existing 64-bit Python 3.9 HTC environment.

## 1. Synchronize the Repository

Open Windows Command Prompt and move to the repository:

```cmd
cd /d S:\building-height-prediction
git status
git branch --show-current
git fetch origin us-lidar-planet-ndsm
git switch us-lidar-planet-ndsm
git pull --ff-only origin us-lidar-planet-ndsm
git log -3 --oneline
```

Before pulling, `git status` should report `nothing to commit, working tree
clean`. Do not discard local changes without first confirming that they are no
longer needed. The commands above assume that the working branch is
`us-lidar-planet-ndsm`. If the work has been merged into `main`, replace the
branch name in the fetch, switch, and pull commands with `main`.

The recent `git log` output should include these framework changes:

```text
fix(ml): correct US44 inventory discovery
feat(ml): add Windows US44 HTC dataset framework
```

If Git reports that the local `us-lidar-planet-ndsm` branch does not exist,
create it from the fetched remote branch, then pull:

```cmd
git switch --track -c us-lidar-planet-ndsm origin/us-lidar-planet-ndsm
git pull --ff-only origin us-lidar-planet-ndsm
```

If `git status` reports local modifications, stop before pulling. Inspect them
with `git diff` and either commit them or preserve them with `git stash push`.
Do not delete local work merely to make the pull succeed.

Confirm that the new pipeline files are present:

```cmd
dir data_source\source\ml_models\prepare_us44_htc_framework.py
dir data_source\source\ml_models\test_prepare_us44_htc_framework.py
dir data_source\source\ml_models\US44_HTC_DATASET_FRAMEWORK.md
```

## 2. Verify Python and the Virtual Environment

Check the available Python version and architecture:

```cmd
python --version
python -c "import platform,struct; print(platform.python_version()); print(struct.calcsize('P')*8, 'bit')"
```

The supported Windows runtime is 64-bit Python 3.9.12. Check whether the HTC
environment already exists:

```cmd
dir data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe
```

If it does not exist, create it with the working Python 3.9 installation:

```cmd
python -m venv data_source\source\ml_models\venv_htc_dc_net
```

If `python` is not the Python 3.9 executable, use its complete path instead.
Do not use `py -3.9.12`; the Windows launcher accepts a major/minor selector
such as `py -3.9`, not a full patch-version selector.

## Expected Inputs

```text
data_source\data\height_labels\generated\us_training_planet_ndsm\us_lidar_to_planet_ndsm_manifest.csv
data_source\data\planet_imagery\generated\processed_us_lidar_scene_selection\selected_processed_us_lidar_planet_scenes.csv
data_source\data\planet_imagery\source\training_lidar_94\<city_slug>\...
```

The nDSM must contain continuous nDSM, building-only nDSM, and LiDAR QA bands.
Every selected scene must contain Surface Reflectance and eight-band UDM2
rasters. CLI paths are repository-relative and may override these defaults.

## 3. Install and Verify Dependencies

```cmd
cd /d S:\building-height-prediction
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -m pip install -r data_source\source\ml_models\htc_dc_net_setup\requirements-windows-cpu.txt
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -c "import torch,rasterio,numpy,pandas,matplotlib,cv2,skimage; print('Python imports passed'); print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available())"
cd data_source\source\ml_models
venv_htc_dc_net\Scripts\python.exe -m unittest test_prepare_us44_htc_framework.py -v
cd /d S:\building-height-prediction
```

The tests must finish with `OK`. CPU execution is expected when the final line
reports `CUDA: False`; dataset preparation does not require a GPU.

## 4. Verify Inputs

Confirm the required manifests and downloaded-data roots before starting:

```cmd
dir data_source\data\height_labels\generated\us_training_planet_ndsm\us_lidar_to_planet_ndsm_manifest.csv
dir data_source\data\planet_imagery\generated\processed_us_lidar_scene_selection\selected_processed_us_lidar_planet_scenes.csv
dir data_source\data\planet_imagery\source\training_lidar_94
```

Confirm that the scene root contains city directories and that the nDSM output
root exists:

```cmd
dir /ad /b data_source\data\planet_imagery\source\training_lidar_94
dir /s /b data_source\data\height_labels\generated\us_training_planet_ndsm\*.tif
```

The inventory stage performs the authoritative check for exactly 44 complete
cities and eight SR/UDM2 scene pairs per city. Do not proceed to later stages
if inventory fails.

The selected-scene manifest defines the authoritative 44-city universe. The
nDSM manifest may contain additional processed cities; these are intentionally
excluded and recorded in:

```text
data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\ignored_ndsm_only_cities.csv
```

Planet filenames differ between sensor generations. Inventory identifies SR
and UDM2 products using the scene ID and product name and prefers the clipped
raster, rather than requiring one exact 4-band or 8-band filename suffix.

## 5. Check Available Disk Space

The framework creates staging rasters plus physical copies in two datasets.
Check the free space on drive `S:` before processing:

```cmd
wmic logicaldisk where "DeviceID='S:'" get DeviceID,FreeSpace,Size
```

If `wmic` is unavailable, use:

```cmd
fsutil volume diskfree S:
```

Do not begin unless there is enough free space for the staging directory, the
four-scene dataset, the eight-scene dataset, temporary files, and logs.

## 6. Recommended Staged Run

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage inventory
type data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\pipeline_status.json
type data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\missing_inputs.csv

data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage split
type data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\city_split_manifest.csv

data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage canonical-chips --resume
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage align-scenes --resume
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage build-datasets --resume
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage statistics
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage validate --qa-chips-per-city 3
```

The equivalent end-to-end resumable command is:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage all --resume
```

Use `--overwrite` only when intentionally rebuilding outputs. It cannot be
combined with `--resume`.

The staged workflow is recommended for the first run because it allows each
large operation to be checked before the next one begins. The single `all`
command is most useful after a partial run or after the workflow has already
been validated on that computer.

## 7. Monitor Progress and Read Failures

Inspect the machine-readable status at any time:

```cmd
type data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\pipeline_status.json
```

List logs with the newest files first:

```cmd
dir /b /o-d data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\logs\*.log
```

After identifying the relevant filename, display it with:

```cmd
type data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\logs\LOG_FILENAME.log
```

Successful completion is indicated only by:

```text
"status": "success"
```

A `failed` status includes the error and Python traceback. Fix the reported
problem and rerun the failed stage with `--resume`.

## Outputs

```text
data_source\data\ml_models\generated\htc_dc_net\us_44_city_staging_v1\
data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\
data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\
```

The split is a deterministic pure-random 19/19/6 city assignment using seed
`20261005`. The four-scene version targets summer north, summer south, winter
north, and winter south. Any fallback is recorded. The eight-scene version
uses every downloaded selected scene. Statistics use training cities only.

Every stage writes a timestamped log. `pipeline_status.json` remains marked
`failed` with the traceback if processing stops, so a partial run cannot appear
successful.

## 8. Verify Final Outputs

Confirm the expected dataset directories and core files:

```cmd
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\image_stats.pickle
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\image_stats.pickle
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\alignment_validation_summary.csv
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\alignment_validation_summary.csv
```

Count split-file observations:

```cmd
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\train.txt
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\val.txt
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\test.txt
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\train.txt
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\val.txt
find /c /v "" data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\test.txt
```

Inspect the visual QA outputs:

```cmd
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1\visual_qa
dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_8scene_v1\visual_qa
```

The validation stage is successful only when it finishes without an exception
and `pipeline_status.json` reports success.

## 9. Safe Recovery

For an interrupted canonical-chip, alignment, or dataset-copy stage, rerun the
same command with `--resume`. Valid existing files will be reused.

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage FAILED_STAGE --resume
```

Replace `FAILED_STAGE` with `canonical-chips`, `align-scenes`, or
`build-datasets`. Statistics and validation can simply be rerun.

Do not manually delete individual outputs unless their log identifies them as
invalid. To deliberately rebuild a stage, use `--overwrite` without
`--resume`:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\prepare_us44_htc_framework.py --stage FAILED_STAGE --overwrite
```

Always rerun `statistics` and `validate` after rebuilding chips or datasets.
