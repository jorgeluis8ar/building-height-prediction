# HTC-DC Net Repo-Style Parameters

This note documents the parameters used in the 40-chip HTC-DC Net calibration
run that was designed to mimic the public GlobalBuildingAtlas / HTC-DC Net
settings while staying on a small sample.

The run used:

```text
20 NYC chips + 20 LA chips
lr: 0.0001
batch_size: 32
optimizer: AdamW
backbone: efficientnetb0
chamfer_weight: 0.01
patience: 20
collapse guard: on
prediction checks every 10 epochs
```

The goal of this run was not final accuracy. The goal was to test whether the
model can train on a small balanced sample without producing constant
prediction rasters.

## Where Parameters Enter

Our local runner is:

```text
data_source/source/ml_models/run_htc_mini_training.py
```

The runner builds a temporary mini-dataset, creates a config dictionary, and
passes that config into the vendored HTC-DC Net code:

```text
run_htc_mini_training.py
  -> prepare_cfg()
  -> get_train_val_dataloaders(cfg)
  -> get_model_and_optimizer(cfg)
  -> UBins(cfg)
```

The upstream HTC-DC Net implementation is vendored at:

```text
data_source/source/ml_models/external/HTC-DC-Net/
```

The most relevant upstream files are:

```text
configs/htcdc.yaml
configs/configs1.yaml
build.py
dataloaders.py
htcdc.py
train.py
```

## Parameter Summary

| Parameter | Current repo-style value | Enters where | Main effect |
|---|---:|---|---|
| `lr` | `0.0001` | Optimizer in `build.py` | Size of weight updates |
| `batch_size` | `32` | DataLoader in `dataloaders.py` | Number of chips per optimizer step |
| `optimizer` | `AdamW` | Optimizer selection in `build.py` | How gradients update weights |
| `backbone` | `efficientnetb0` | `UBins` in `htcdc.py` | Image feature extractor |
| `chamfer_weight` | `0.01` | Loss in `UBins.get_losses()` | Weight on adaptive-bin regularization |
| `patience` | `20` | Upstream `train.py`; recorded in mini config | Early-stopping tolerance in full trainer |
| `collapse guard` | on | Local mini runner | Diagnostic stop rule for flat predictions |
| `prediction checks` | every `10` epochs | Local mini runner | Frequency for exported QA rasters |

## Learning Rate

Current value:

```text
lr: 0.0001
```

Where it enters:

```text
data_source/source/ml_models/external/HTC-DC-Net/build.py
```

The optimizer is created as:

```python
optimizer = optim(filter(lambda x: x.requires_grad, model.parameters()), lr=cfgs["lr"])
```

What it does:

The learning rate controls how large the weight update is after each backward
pass. In this model, it affects all trainable parts: EfficientNet backbone,
decoder, adaptive-bin transformer layers, and prediction heads.

How changing it affects the model:

- Higher learning rate: faster movement, but higher risk of instability or
  shortcut solutions.
- Lower learning rate: more stable, but slower learning.
- In our tests, `0.001` caused immediate constant-raster collapse on the
  40-chip setup.
- The upstream value, `0.0001`, avoided full collapse under batch size `32`.

Practical interpretation:

```text
0.001   too aggressive for current 40-chip HTC run
0.0001  closest to upstream; currently the safest baseline
```

## Batch Size

Current value:

```text
batch_size: 32
```

Where it enters:

```text
data_source/source/ml_models/external/HTC-DC-Net/dataloaders.py
```

The DataLoader receives:

```python
torch.utils.data.DataLoader(..., batch_size=batch_size, shuffle=not is_validation)
```

What it does:

Batch size controls how many chips are used to compute one gradient update.
With 40 training chips:

```text
batch_size = 1   -> 40 optimizer steps per epoch
batch_size = 32  -> 2 optimizer steps per epoch
```

How changing it affects the model:

- Smaller batch size gives noisier, more frequent updates.
- Larger batch size gives smoother, fewer updates.
- Larger batches can stabilize training, but each step is heavier in memory and
  compute.

Why it mattered here:

The earlier batch-size-1 runs often moved toward flat, central-height
predictions. The repo-style batch size `32` avoided full collapse through the
exported checkpoints up to epoch 40.

Practical caveat:

On the local CPU/Mac setup, batch size `32` was slower than expected. It may be
more appropriate on CUDA/GPU, which is what the upstream config assumes.

## Optimizer

Current value:

```text
optimizer: AdamW
```

Where it enters:

```text
data_source/source/ml_models/external/HTC-DC-Net/build.py
```

What it does:

The optimizer defines how gradients are converted into weight updates. AdamW is
an adaptive optimizer with decoupled weight decay. The code uses PyTorch's
default AdamW settings except for the learning rate supplied by config.

How changing it affects the model:

- `AdamW`: stable default for deep networks; upstream choice.
- `Adam`: similar adaptive behavior, but different weight decay handling.
- `SGD`: often requires more careful learning-rate schedules and momentum.
- `NAdam`: adaptive variant with Nesterov-style momentum.

Recommended baseline:

Keep `AdamW` while we are validating data and architecture behavior. Changing
optimizer now would make it harder to compare against the upstream repo.

## Backbone

Current value:

```text
backbone: efficientnetb0
```

Where it enters:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
```

The model loads:

```text
tf_efficientnet_b0_ap
```

from Torch Hub and removes the classification head:

```python
basemodel.global_pool = nn.Identity()
basemodel.classifier = nn.Identity()
```

What it does:

The backbone converts the RGB PlanetScope chip into image features. Those
features then feed the decoder and adaptive-bin height head.

How changing it affects the model:

- Larger EfficientNet backbones may learn richer features, but need more memory
  and training data.
- Smaller/simple backbones are faster, but may underfit.
- The upstream default is `efficientnetb0`, so this is the right first baseline.

Practical interpretation:

Do not change the backbone until the current `efficientnetb0` setup can
reliably produce non-collapsed predictions and reasonable validation behavior.

## Chamfer Weight

Current value:

```text
chamfer_weight: 0.01
```

Where it enters:

```text
data_source/source/ml_models/external/HTC-DC-Net/htcdc.py
UBins.get_losses()
```

The total loss includes:

```text
masked MAE + chamfer_weight * bin_chamfer
```

What it does:

HTC-DC Net predicts height through adaptive bins. The Chamfer loss regularizes
the predicted bin locations against the distribution of target heights.

How changing it affects the model:

- Higher `chamfer_weight`: stronger pressure for bin centers to match the
  target-height distribution.
- Lower `chamfer_weight`: the model focuses more on pixel-wise MAE.
- Too high: bin regularization may dominate pixel prediction.
- Too low: adaptive bins may become less meaningful.

Recommended baseline:

Keep `0.01`, because this is the upstream setting. Revisit only after we have a
stable training/validation loop.

## Patience

Current value:

```text
patience: 20
```

Where it enters upstream:

```text
data_source/source/ml_models/external/HTC-DC-Net/train.py
```

What it does in the upstream trainer:

Patience controls early stopping. If the monitored validation metrics stop
improving for `20` epochs, training stops.

Important local distinction:

Our mini runner does not currently implement the full upstream validation
early-stopping logic. It records `patience` in the config, but the active local
stop rule is the collapse guard.

How changing it affects the model:

- In the full upstream trainer: larger patience trains longer before stopping.
- In our current mini runner: changing `patience` does not stop training by
  itself.

Recommended next step:

If we want true repo-style early stopping, the mini runner should add validation
metrics and monitor `mae`, `rmse`, and `val/loss_total`, matching upstream
`train.py`.

## Collapse Guard

Current setting:

```text
collapse guard: on
collapse_std_threshold: 0.05 m
collapse_min_share: 0.8
collapse_patience: 1
```

Where it enters:

```text
data_source/source/ml_models/run_htc_mini_training.py
```

What it does:

This is not an upstream HTC-DC Net parameter. It is our diagnostic guardrail.
For each exported prediction chip, the runner computes the standard deviation
of predicted height over evaluated building pixels.

A chip is flagged as collapsed when:

```text
pred_std_m < 0.05
```

The run stops when:

```text
collapsed_share >= 0.8
```

How changing it affects the model:

It does not change training gradients or predictions. It only changes whether
we stop and flag bad runs.

Why it matters:

Without this guardrail, a run can report reasonable MAE while producing
nearly constant height rasters. Those predictions are not useful for our
project because the final goal is a spatial height raster.

## Prediction Checks

Current value:

```text
save_predictions_every: 10
```

Where it enters:

```text
data_source/source/ml_models/run_htc_mini_training.py
```

What it does:

Every 10 epochs, the runner exports prediction GeoTIFFs and writes
`predictions_summary_epoch_XXX.csv`.

How changing it affects the run:

- Smaller interval: more frequent QA, but slower runs and more output files.
- Larger interval: faster runs, but less visibility into collapse and learning
  dynamics.

Recommended use:

For calibration:

```text
save_predictions_every: 10
```

For fast smoke tests:

```text
save_predictions_every: 1
```

For larger runs:

```text
save_predictions_every: 25 or 50
```

## What The 40-Chip Repo-Style Run Showed

The run was interrupted after epoch 47 because the full `500` epoch ceiling was
too slow for local CPU calibration. The last complete exported prediction
checkpoint was epoch 40.

Collapse diagnostics:

```text
Epoch 10:  0 / 40 collapsed
Epoch 20: 13 / 40 collapsed
Epoch 30: 14 / 40 collapsed
Epoch 40:  8 / 40 collapsed
```

Epoch 40 summary:

```text
LA mean MAE: 3.3754 m
LA mean RMSE: 4.6536 m
NYC mean MAE: 16.7513 m
NYC mean RMSE: 23.6691 m
```

Interpretation:

The repo-style settings did not solve the model, but they produced the best
behavior so far. They avoided the full constant-raster failure and improved NYC
relative to earlier 40-chip tests.

## Recommended Next Calibration

For the next small run, use a shorter explicit epoch count so the run completes
and writes final model artifacts:

```bash
WANDB_MODE=offline TORCH_HOME=/private/tmp/torch_htc_cache \
  data_source/source/ml_models/venv_htc_dc_net/bin/python \
  data_source/source/ml_models/run_htc_mini_training.py \
  --nyc-chips 20 \
  --la-chips 20 \
  --epochs 50 \
  --lr 0.0001 \
  --batch-size 32 \
  --num-workers 0 \
  --patience 20 \
  --seed 20260707 \
  --save-predictions-every 10 \
  --save-checkpoints-every 50 \
  --collapse-std-threshold 0.05 \
  --collapse-min-share 0.8 \
  --collapse-patience 1 \
  --stop-on-collapse \
  --run-name nyc20_la20_seed20260707_epoch50_repo_params_guarded
```

This keeps the upstream-like parameter choices, but makes the local experiment
finish cleanly with a final checkpoint and prediction summary.
