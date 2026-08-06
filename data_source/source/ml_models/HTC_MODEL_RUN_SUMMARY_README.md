# HTC-DC Net Model Run Summary

This document summarizes the main HTC-DC Net experiments and records the model
selected for continued work.

## Selected Primary Model

The selected model is:

```text
nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded
```

It was selected using held-out validation performance. It has the lowest
pooled validation RMSE, highest validation R2, and best NYC validation RMSE
among the directly comparable off-nadir models. The exact configuration and
cross-computer reproduction checklist are in `SELECTED_HTC_DC_MODEL.md`.

| Metric scope | Buildings | MAE m | RMSE m | Bias m | R2 |
|---|---:|---:|---:|---:|---:|
| Train | 37,058 | 2.229 | 4.810 | -0.630 | 0.837 |
| Validation | 6,891 | 2.886 | 7.699 | -0.758 | 0.555 |
| Test | 9,371 | 2.712 | 8.091 | -0.932 | 0.351 |
| All | 53,320 | 2.399 | 5.937 | -0.700 | 0.731 |

## Completed Main Runs

| Run | Dataset | Channels | Epochs | LR | Batch | Loss / sampling | Status | Main result |
|---|---|---:|---:|---:|---:|---|---|---|
| `nyc5_la5_seed20260707_epoch1` | `nyc_la_rgb_v1` | 3 | 1 | default | 1 | baseline smoke | completed | Infrastructure smoke test worked; predictions had spatial variation. |
| `nyc5_la5_seed20260707_epoch50_lr001` | `nyc_la_rgb_v1` | 3 | 50 | 0.001 | 1 | baseline | completed | Collapsed to near-constant predictions. |
| `nyc20_la20_seed20260707_epoch50_lr001` | `nyc_la_rgb_v1` | 3 | 50 | 0.001 | 1 | baseline | completed | Collapsed to near-constant predictions. |
| `nyc71_la100_seed20260707_epoch5_repo_params_guarded` | `nyc_la_rgb_v1` | 3 | 5 | 0.0001 | 32 | repo-style guarded | completed | First stable repo-style guarded mini-run. |
| `nyc76_la95_6ch_lowrise_binweighted_bg005_seed20260715_epoch20_guarded` | `nyc_la_6ch_v1` | 6 | 20 | 0.0001 | 32 | low-rise/high-rise bin-weighted | completed | Improved height variation but still weak low-rise behavior. |
| `nyc76_la95_12ch_lowrise_binweighted_bg005_seed20260715_epoch20_guarded` | `nyc_la_12ch_v1` | 12 | 20 | 0.0001 | 32 | low-rise/high-rise bin-weighted | completed | LA mean MAE 4.17 m; NYC mean MAE 12.05 m; 2 collapsed chips. |
| `nyc76_la95_12ch_lowrise_binweighted_bg005_seed20260715_epoch50_guarded` | `nyc_la_12ch_v1` | 12 | 50 | 0.00003 | 8 | low-rise/high-rise bin-weighted | completed, historical | Validation RMSE 11.92 m; test RMSE 10.25 m; 0 collapsed chips. |
| `nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded` | `nyc_la_off_nadir_rgb_nir_v1` | 4 | 50 | 0.00003 | 8 | low-rise/high-rise bin-weighted | **selected primary model** | Validation RMSE 7.70 m, validation R2 0.555, and test RMSE 8.09 m. |
| `nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_seed20260723_epoch50` | `nyc_la_off_nadir_rgb_nir_full_recipe_v1` | 4 | 50 | 0.0001 | 8 | full HTC, Gaussian foreground/uniform background, unweighted positive L1 | completed, not best | Best checkpoint epoch 40; building RMSE 8.71 m validation and 8.13 m test. NYC validation RMSE remains 21.16 m. |
| `nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_binweighted_seed20260723_epoch50` | `nyc_la_off_nadir_rgb_nir_full_recipe_v1` | 4 | 50 | 0.0001 | 8 | full HTC plus target-height bin weighting `4,3,2,1,3,8` | completed | Best checkpoint epoch 15; pooled building RMSE 8.41 m validation and 8.60 m test; city-balanced validation RMSE 11.62 m. |

## Full Paper-Recipe Result

The EfficientNet-B5 full HTC-DC reproduction completed successfully and passed
the architecture, loss, gradient, alignment, and sliding-window preflight. It
did not improve on the simpler off-nadir RGB+NIR EfficientNet-B0 model: pooled
building validation RMSE was 8.71 m instead of 7.70 m, while test RMSE was
similar at 8.13 m instead of 8.09 m. The selected checkpoint is epoch 40, based
on a city-balanced validation building RMSE of 11.97 m. This experiment is
therefore retained as a complete methodological reproduction, not promoted as
the current best predictive model.

## Height-Weighted Full Paper-Recipe Result

Adding target-height loss weights improved the comparable Full B5
city-balanced validation checkpoint metric from 11.97 m to 11.62 m. Pooled
validation RMSE also improved from 8.71 m to 8.41 m, while pooled test RMSE
worsened from 8.13 m to 8.60 m. The result is therefore a modest validation
gain rather than a decisive overall improvement. NYC validation and test RMSE
remain about 20 m, and the prediction range remains strongly compressed for
the tallest buildings.

## Selected Model Parameters

| Parameter | Value |
|---|---|
| Dataset | `nyc_la_off_nadir_rgb_nir_v1` |
| Input channels | 4 |
| PlanetScope inputs | 1 off-nadir RGB+NIR scene per city |
| Model | HTC-DC Net |
| Backbone | EfficientNet-B0 |
| Epochs | 50 |
| Learning rate | 0.00003 |
| Batch size | 8 |
| Optimizer | AdamW |
| Chamfer weight | 0.01 |
| Loss | Bin-weighted low-rise/high-rise height loss |
| Height bin edges | `3,6,10,25,50` m |
| Height bin weights | `4,3,2,1,3,8` |
| Background loss weight | 0.05 |
| Patience | 20 |
| Collapse guard | On |
| Prediction exports | Every 10 epochs |

The selected checkpoint is `model_epoch_050.pth`. Because all files under
`data_source/data/` are Git-ignored, the checkpoint and model-ready dataset
must be transferred separately or rebuilt after cloning the repository.
