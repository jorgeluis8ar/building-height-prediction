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

## Data Files Are Not Distributed Through Git

Git contains the scripts and this guide only. Planet imagery, nDSM rasters,
and generated CSV manifests must not be added to Git. Transfer the six-city
order manifest from the computer that created the Planet orders and save it on
Windows at exactly:

```text
S:\building-height-prediction\data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\planet_orders_manifest.csv
```

For the later 50-city dataset extension, transfer the selected-scene metadata
to this adjacent path as well:

```text
S:\building-height-prediction\data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv
```

Verify both manually transferred files from CMD:

```cmd
dir data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\planet_orders_manifest.csv
dir data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\selected_six_us_lidar_postyear_planet_scenes.csv
```

When the Planet orders are ready, first check their states without downloading:

```cmd
data_source\source\planet_imagery\venv_planet_imagery\Scripts\python.exe data_source\source\planet_imagery\download_ordered_planet_scenes.py ^
  --manifest data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\planet_orders_manifest.csv ^
  --output-root data_source\data\planet_imagery\source\processed_us_lidar ^
  --dry-run
```

Then download every completed order into the existing processed-LiDAR scene
root. This command is resumable and can be rerun as remaining orders finish:

```cmd
data_source\source\planet_imagery\venv_planet_imagery\Scripts\python.exe data_source\source\planet_imagery\download_ordered_planet_scenes.py ^
  --manifest data_source\data\planet_imagery\generated\six_us_lidar_postyear_scene_selection_v1\planet_orders_manifest.csv ^
  --output-root data_source\data\planet_imagery\source\processed_us_lidar ^
  --confirm-download
```

The downloaded scene directories must therefore be stored below:

```text
S:\building-height-prediction\data_source\data\planet_imagery\source\processed_us_lidar\<city_slug>\<selection>_<scene_id>\
```

Do not use `git add` on either data path. The manifest is updated locally with
Planet order states and downloaded-file paths as the downloader runs.

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
data_source\data\planet_imagery\source\processed_us_lidar\<city_slug>\...
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
dir data_source\data\planet_imagery\source\processed_us_lidar
```

Confirm that the scene root contains city directories and that the nDSM output
root exists:

```cmd
dir /ad /b data_source\data\planet_imagery\source\processed_us_lidar
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

## LiDAR Provenance Report

Generate a reproducible record of the USGS projects and acquisition dates used
by exactly the original 44-city dataset:

```cmd
data_source\source\height_labels\venv_height_labels\Scripts\python.exe data_source\source\height_labels\report_us_lidar_provenance.py
```

The command reads the authoritative 44-city scene universe and the nDSM run
manifest. When the USGS metadata service is available, it refreshes exact
project-level acquisition dates and horizontal CRS codes. It writes:

```text
data_source\data\height_labels\generated\us_training_planet_ndsm\provenance\us44_v1\us44_lidar_city_provenance.csv
data_source\data\height_labels\generated\us_training_planet_ndsm\provenance\us44_v1\us44_lidar_project_provenance.csv
data_source\data\height_labels\generated\us_training_planet_ndsm\provenance\us44_v1\provenance_summary.json
```

Display the compact city-level table from CMD:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe -c "import pandas as pd; p=r'data_source\data\height_labels\generated\us_training_planet_ndsm\provenance\us44_v1\us44_lidar_city_provenance.csv'; d=pd.read_csv(p,dtype=str).fillna(''); print(d[['city_slug','selected_project_directories','lidar_collect_start','lidar_collect_end','lidar_acquisition_years','status']].to_string(index=False))"
```

Use `--offline` only when the USGS metadata service is unavailable. The offline
report retains the city-level dates recorded during nDSM creation but cannot
refresh separate dates and CRS codes for every selected project.

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

On Windows, cloud synchronization, antivirus scanning, or an open editor can
briefly lock `pipeline_status.json`. The writer retries transient sharing
violations automatically. If Windows still reports `PermissionError`, close
any program displaying that JSON file, wait for synchronization to finish,
and rerun the same stage. A stage log that already reports its main output as
created can be rerun safely; the deterministic split will be identical.

## 10. Two-City Selected-Model Pilot

The mini-training runner can select complete cities reproducibly from the 19
training cities. It uses every retained observation for each selected city;
the four PlanetScope scenes remain independent four-channel RGB+NIR training
observations rather than being stacked into a 16-channel image. The run writes
`selected_training_cities.csv` with the selected city names and observation
counts.

Run a one-epoch infrastructure smoke test first:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\run_htc_mini_training.py ^
  --dataset-dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1 ^
  --training-city-count 2 ^
  --in-channels 4 ^
  --epochs 1 ^
  --seed 20260721 ^
  --lr 0.00003 ^
  --batch-size 8 ^
  --num-workers 0 ^
  --patience 20 ^
  --height-loss-weighting bins ^
  --height-bin-edges 3,6,10,25,50 ^
  --height-bin-weights 4,3,2,1,3,8 ^
  --background-loss-weight 0.05 ^
  --save-predictions-every 1 ^
  --save-checkpoints-every 1 ^
  --collapse-std-threshold 0.05 ^
  --collapse-min-share 0.8 ^
  --collapse-patience 1 ^
  --stop-on-collapse ^
  --run-name us44_4scene_2city_selected_params_smoke_seed20260721_epoch1
```

After the smoke test completes without collapse or file errors, run the
selected 50-epoch configuration. The same seed selects the same two cities:

```cmd
data_source\source\ml_models\venv_htc_dc_net\Scripts\python.exe data_source\source\ml_models\run_htc_mini_training.py ^
  --dataset-dir data_source\data\ml_models\generated\htc_dc_net\us_44_rgbnir_4scene_v1 ^
  --training-city-count 2 ^
  --in-channels 4 ^
  --epochs 50 ^
  --seed 20260721 ^
  --lr 0.00003 ^
  --batch-size 8 ^
  --num-workers 0 ^
  --patience 20 ^
  --height-loss-weighting bins ^
  --height-bin-edges 3,6,10,25,50 ^
  --height-bin-weights 4,3,2,1,3,8 ^
  --background-loss-weight 0.05 ^
  --save-predictions-every 10 ^
  --save-checkpoints-every 10 ^
  --collapse-std-threshold 0.05 ^
  --collapse-min-share 0.8 ^
  --collapse-patience 1 ^
  --stop-on-collapse ^
  --run-name us44_4scene_2city_selected_params_seed20260721_epoch50
```

This pilot intentionally diagnoses training behavior on two cities. It does
not replace evaluation on the unchanged validation and test city splits.

## 11. Run The Two-City Pilot On macOS

The original Mac checkout uses the same dataset and seed as the Windows pilot.
Seed `20260721` selects `greenville_23005` and `rochester_23118`, containing
1,056 four-channel scene observations in total.

### Make Dropbox Data Local

Training cannot read Dropbox placeholders. The simplest option is to locate
`us_44_rgbnir_4scene_v1` in Finder, right-click it, and choose **Make Available
Offline**. The dataset has about 8.2 GB of logical content, before the
run-specific mini-dataset copy.

To prioritize only the two pilot cities, open each of `image`, `mask`, and
`ndsm` in Finder. Search by filename for `c17_`, select every result, and choose
**Make Available Offline**; then repeat for `c31_`. City `c17` is Greenville
and city `c31` is Rochester. The pilot runner validates only those selected
city files, so the other cities may remain online-only.

From the repository root, verify that no image TIFF is a zero-allocation
sparse placeholder:

```bash
python3 -c 'from pathlib import Path; p=Path("data_source/data/ml_models/generated/htc_dc_net/us_44_rgbnir_4scene_v1/image"); bad=[f for f in p.glob("*.tif") if f.stat().st_size and getattr(f.stat(),"st_blocks",1)==0]; print("Sparse placeholders:",len(bad)); print(*bad[:10],sep="\n"); raise SystemExit(bool(bad))'
```

When downloading the complete dataset, the required result is `Sparse
placeholders: 0` and exit status zero. For the two-city-only option, verify the
selected prefixes instead:

```bash
python3 -c 'from pathlib import Path; root=Path("data_source/data/ml_models/generated/htc_dc_net/us_44_rgbnir_4scene_v1"); paths=[p for d in ("image","mask","ndsm") for prefix in ("c17_","c31_") for p in (root/d).glob(prefix+"*.tif")]; bad=[p for p in paths if p.stat().st_size and getattr(p.stat(),"st_blocks",1)==0]; print("Selected files:",len(paths)); print("Sparse placeholders:",len(bad)); print(*bad[:10],sep="\n"); raise SystemExit(bool(bad))'
```

The selected-city check must report `Selected files: 3168` and `Sparse
placeholders: 0`.

### Verify The Environment And Selection

```bash
cd "/Users/jorgeochoa/Library/CloudStorage/Dropbox-Brown/Jorge Ochoa/Research/building-height-prediction"
export MPLCONFIGDIR=/private/tmp/matplotlib_htc
export WANDB_MODE=offline
export PYTORCH_ENABLE_MPS_FALLBACK=1

data_source/source/ml_models/venv_htc_dc_net/bin/python -c 'import torch,rasterio,numpy,pandas,yaml,skimage,cv2,matplotlib,timm,fvcore; print("Imports passed"); print("Torch:",torch.__version__); print("MPS:",torch.backends.mps.is_available()); assert torch.backends.mps.is_available()'

data_source/source/ml_models/venv_htc_dc_net/bin/python -c 'import csv,random; from pathlib import Path; p=Path("data_source/data/ml_models/generated/htc_dc_net/us_44_rgbnir_4scene_v1"); ids={x.strip() for x in (p/"train.txt").read_text().splitlines() if x.strip()}; rows=list(csv.DictReader((p/"chips_manifest.csv").open())); cities=sorted({r["city_slug"] for r in rows if r["observation_id"] in ids}); print(sorted(random.Random(20260721).sample(cities,2)))'
```

The final command must print:

```text
['greenville_23005', 'rochester_23118']
```

### One-Epoch Smoke Test

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --dataset-dir data_source/data/ml_models/generated/htc_dc_net/us_44_rgbnir_4scene_v1 \
  --training-city-count 2 \
  --in-channels 4 \
  --epochs 1 \
  --seed 20260721 \
  --lr 0.00003 \
  --batch-size 8 \
  --device mps \
  --num-workers 0 \
  --patience 20 \
  --height-loss-weighting bins \
  --height-bin-edges 3,6,10,25,50 \
  --height-bin-weights 4,3,2,1,3,8 \
  --background-loss-weight 0.05 \
  --save-predictions-every 1 \
  --save-checkpoints-every 1 \
  --collapse-std-threshold 0.05 \
  --collapse-min-share 0.8 \
  --collapse-patience 1 \
  --stop-on-collapse \
  --run-name us44_4scene_2city_selected_params_smoke_seed20260721_epoch1
```

### Fifty-Epoch Run

Run this only after the smoke test writes a checkpoint, prediction summaries,
and a non-collapsed final prediction check:

```bash
data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --dataset-dir data_source/data/ml_models/generated/htc_dc_net/us_44_rgbnir_4scene_v1 \
  --training-city-count 2 \
  --in-channels 4 \
  --epochs 50 \
  --seed 20260721 \
  --lr 0.00003 \
  --batch-size 8 \
  --device mps \
  --num-workers 0 \
  --patience 20 \
  --height-loss-weighting bins \
  --height-bin-edges 3,6,10,25,50 \
  --height-bin-weights 4,3,2,1,3,8 \
  --background-loss-weight 0.05 \
  --save-predictions-every 10 \
  --save-checkpoints-every 10 \
  --collapse-std-threshold 0.05 \
  --collapse-min-share 0.8 \
  --collapse-patience 1 \
  --stop-on-collapse \
  --run-name us44_4scene_2city_selected_params_seed20260721_epoch50
```

Both runs explicitly use the Apple Metal GPU through `--device mps`. The
fallback environment variable sends an operation to CPU only when PyTorch has
no MPS implementation for it; the resolved device is recorded in
`config_used.yaml` and `run_metadata.json`. Their outputs are written under
`data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/`. Monitor
an active process with `ps -o pid,etime,%cpu,%mem,command -p $(pgrep -f
'run_htc_mini_training.py' | head -1)`.
