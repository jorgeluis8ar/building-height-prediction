# GlobalBuildingAtlas building-height pipeline: paper + code audit

Date reviewed: 2026-07-21

## Scope and evidence

This note answers questions about the lower-middle **Global Building Height Estimation** module in Figure 2 of Zhu et al., *GlobalBuildingAtlas: An Open Global and Complete Dataset of Building Polygons, Heights and LoD1 3D Models* (the supplied arXiv v1, `2506.04106v1.pdf`). I cross-checked:

- the supplied paper, especially Sections 4.2, 4.4, and 4.5;
- the released [GlobalBuildingAtlas repository](https://github.com/zhu-xlab/GlobalBuildingAtlas) at commit [`d58507e`](https://github.com/zhu-xlab/GlobalBuildingAtlas/tree/d58507e22af2bd940de5f98da7ed6763bd32147c);
- its `im2bh` submodule, [HTC-DC-Net](https://github.com/zhu-xlab/HTC-DC-Net), at commit [`adae55e`](https://github.com/zhu-xlab/HTC-DC-Net/tree/adae55edc8be589757cec57f839d59a681d93364); and
- Chen et al., [*HTC-DC Net: Monocular Height Estimation From Single Remote Sensing Images*](https://arxiv.org/abs/2309.16486), which contains the mathematics that the atlas paper omits.

The atlas repository does **not** include the actual atlas training split lists, training imagery/nDSMs, archived run configuration, normalization statistics, or trained checkpoint. Consequently, some exact run settings cannot be recovered from code alone. Where the atlas paper, generic HTC-DC paper, and example config differ, I distinguish them explicitly.

## Executive answer

The height model is a **dense, pixel-wise monocular nDSM predictor**. Its model input is a normalized 3-channel PlanetScope optical patch, not a building polygon. An EfficientNet-B5 encoder-decoder extracts multi-scale features. An HTC-AdaBins head uses a small ViT to create **image-adaptive height intervals (bins)** and a categorical probability over those bins at every pixel. A head-tail classifier separates foreground/tall structure pixels from background/near-zero pixels to reduce the severe long-tail bias. The final continuous height is the expected value of the bin centers under the per-pixel probabilities. During training, L1 height loss, Chamfer bin-edge loss, binary head-tail loss, and distribution-shape regularization supervise the network.

At global inference, overlapping 256 x 256 windows with stride 128 predict each interior pixel up to four times. The mean becomes the height raster and the population variance of those overlapping predictions becomes the reported uncertainty raster. Only **after** this raster exists do fused building footprints enter: the maximum predicted pixel height inside each footprint is assigned to that building, and the variance at that same maximum-height pixel becomes its uncertainty.

## End-to-end pipeline

### 1. Global PlanetScope acquisition and mosaicking

The Earth is tiled into 0.2 degree x 0.2 degree cells. Cells intersecting built-up areas in the Global Urban Footprint (GUF) are retained. The authors downloaded **approximately 800,000 PlanetScope Surface Reflectance scenes**, primarily from 2019, each about 287.5 km2 at 3 m resolution. Scenes were filtered to less than 10% cloud cover; 2018 imagery filled locations without suitable 2019 coverage. For each cell, overlapping scenes were mosaicked using Planet Unusable Data Masks to prioritize clear and complete pixels (atlas paper Section 4.2).

Important interpretation: 800,000 is the number of scenes acquired for global mosaicking, **not** the number of training samples and not necessarily the number of distinct scenes contributing to the final mosaic after filtering/selection. The release gives no exact retained-scene count or average number of source scenes per mosaic.

The model later reads only the first three bands and optionally reverses their order depending on the `rgb` flag; the global script calls inference with `rgb=False`, so the three bands are reversed before channel-wise mean/std normalization ([normalization and input code](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/utils.py#L175-L198)). The precise band-order convention and the normalization statistics file used for the published run are not released.

### 2. Training sample definition

For the atlas model, a training sample is a spatially aligned pair:

1. a **256 x 256 PlanetScope image patch** (3 m GSD, so about 768 m x 768 m on the ground), and
2. a **256 x 256 LiDAR-derived nDSM target** resampled/rasterized to 3 m.

The nDSM is above-ground height, produced from government LiDAR rather than absolute elevation. The atlas training corpus contains **231,656 patches from 168 city-scale regions of interest**, mostly in North America, Europe, and Oceania (atlas paper Section 4.4.1). This geographic imbalance matters: the paper itself reports worse transfer in less-represented regions, especially South America.

The loader uses a text file containing sample basenames; for each basename it resolves `_IMG.tif`, `_AGL.tif`, and `_BLG.tif` under `image/`, `ndsm/`, and `mask/` ([dataset resolution](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/dataloaders.py#L193-L209)). It reads the image as float32, changes HWC to CHW, replaces nDSM NaNs with zero, clips negative target heights to zero, optionally loads a binary footprint mask, resizes, and normalizes the image ([sample loader](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/dataloaders.py#L176-L185)).

What is **not reported or released** for the atlas run:

- how the 231,656 samples are divided among training, validation, and test;
- whether splitting is by city/RoI, by original LiDAR tile, or randomly by patch;
- overlap policy between adjacent training patches;
- nodata masking rules for the LiDAR target;
- per-RoI sample counts and the exact 168-RoI list;
- whether empty/background-only patches are retained and in what proportion; and
- the published run's augmentation settings.

This is a serious reproducibility gap. Patch-random splitting could leak nearly identical neighboring content into train and validation; for a model intended to generalize geographically, a city- or RoI-held-out split is much more informative.

Do not confuse this atlas corpus with the older HTC-DC paper's public **GBH benchmark**: that benchmark has 20,532 patches from 19 cities (14,971 train / 3,660 validation / 1,901 test) plus Los Angeles, Sao Paulo, and Guangzhou held out for testing. The atlas paper expanded the corpus to 168 RoIs and 231,656 samples but does not publish its split details.

### 3. Feature extraction: EfficientNet-B5 encoder-decoder

The atlas paper explicitly says the published model uses **EfficientNet-B5** as the backbone, followed by an HTC-DC Net classification-regression head (Section 4.4.2). The code loads a pretrained `tf_efficientnet_b5_ap` model when `backbone: efficientnetb5`, removes its global pooling/classifier, preserves intermediate block outputs, and passes skip features to a bilinear-upsample decoder ([backbone construction](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/baselines/adabins_htc.py#L333-L345)).

The decoder repeatedly upsamples deeper, semantic features and concatenates shallower, spatially detailed EfficientNet features. HTC-DC is multi-level: adaptive-bin heads can be attached at several decoder resolutions. Intermediate height outputs are training-only deep supervision; the final, highest-resolution decoder output is used at inference.

Why this stage exists: one 3 m RGB pixel cannot uniquely determine height. The backbone learns visual cues correlated with height - shadows, roof texture, context, building scale, neighborhood morphology - and the decoder restores spatial detail. This is still monocular inference, so it learns statistical associations rather than geometric height from stereo/parallax.

### 4. Transformer encoder: local-global interaction

For each selected decoder feature map `F`:

- a local branch applies a 3 x 3 convolution;
- a patch convolution (paper/code patch size **4**) converts `F` to 128-dimensional tokens;
- a ViT encoder with **4 transformer layers, 4 attention heads, and a 1024-dimensional feed-forward layer** mixes information globally ([transformer implementation](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/parts/layers.py#L5-L25)); and
- token/local-feature dot products create range-attention maps, which combine image-wide height-distribution context with pixel-local evidence ([mViT implementation](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/parts/miniViT.py#L8-L46)).

The first transformer token is sent through an MLP to predict relative bin widths for the whole input patch. Subsequent tokens act as queries against every local feature vector to create pixel-wise range-attention maps.

### 5. Adaptive bin edges

The model uses **N = 256 bins** in the HTC-DC design. These are not 256 fixed one-meter classes. For each input image and each supervised feature scale, the network predicts 256 positive normalized widths with a softmax. Widths are scaled to the allowed height range `[h_min, h_max]`; a cumulative sum converts widths into 257 ordered edges:

`b_0 = h_min`, and `b_i = b_(i-1) + width_i`.

Centers are `c_i = (b_(i-1) + b_i) / 2`. The maximum height comes from the maximum observed nDSM stored in `ndsm_stats.pickle`, so the exact `h_max` of the atlas run is unavailable. The released inference implementation defaults to 256 bins, patch size 4, and a 1 m head-tail threshold ([bin configuration](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/baselines/adabins_htc.py#L300-L331)).

Why adaptive bins help: a mostly low-rise patch can allocate finer intervals to low heights, while a high-rise patch can allocate capacity differently. The Chamfer bin-edge loss encourages learned edges to follow the patch's empirical target-height distribution, ideally resembling its quantiles.

### 6. Bin probabilities and the head-tail cut (HTC)

A 1 x 1 convolution plus softmax maps the range-attention maps to a 256-class probability vector `P_i(x,y)` at every pixel. This says how likely the pixel's height is to fall in each adaptive interval.

The distinctive HTC step tackles the extreme long tail: background/ground pixels dominate, while high buildings are rare. Pixels above **1 m** are treated as foreground and those at or below 1 m as background. The full HTC-DC design makes separate foreground and background range-attention/probability representations and predicts a sigmoid head-tail gate. It then selects/combines the foreground or background bin distribution at each pixel. A binary cross-entropy loss supervises the gate against `nDSM > 1 m` (HTC-DC paper, equations 7-9 and 14).

The released global inference code also contains an alternate `htc_source: bf` path that can replace the learned gate with a supplied footprint mask. That mode asserts a mask is available ([alternate footprint-gated path](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/baselines/adabins_htc.py#L220-L247)). However, `htc_source` defaults to `pred`, and the actual global inference loop calls `model(imgP)` with **only the image tensor**, no footprint argument ([global call](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/utils.py#L315-L327)). Thus the released operational path does not use footprints as model input.

### 7. Hybrid regression: probabilities to continuous height

For every pixel, the final continuous height is the probability-weighted average of bin centers:

`H(x,y) = sum_i P_i(x,y) * c_i`.

This is the expectation of the learned discrete distribution. It avoids staircase artifacts from simply taking the most probable bin and produces a continuous-valued nDSM. The implementation computes this weighted sum at each supervised decoder level and upsamples the final output to the input patch size ([bin construction and weighted sum](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/baselines/adabins_htc.py#L254-L267)).

### 8. How the model is trained

For the atlas production run, the paper reports:

- EfficientNet-B5 backbone;
- 150 epochs;
- batch size 8; and
- AdamW optimizer.

The learning rate is not stated in the atlas paper. The generic HTC-DC implementation/paper uses AdamW at **1e-4**, 256 bins, transformer patch size 4, and multi-level loss weights 0.125, 0.25, 0.5, and 1 (discarding the earliest feature level). These are strong clues, but without the atlas run's archived `config.yaml` they should not be asserted as verified atlas settings.

The HTC-DC objective has four components:

1. **Pixel height loss:** L1 between predicted and LiDAR nDSM. The released training implementation applies its L1 height term only where target nDSM is greater than zero, so its exact behavior differs slightly from the paper's all-pixel notation.
2. **Bin-edge loss:** Chamfer distance between predicted bin representatives and flattened ground-truth heights, weighted by 0.01 in the generic config.
3. **HTC loss:** binary cross-entropy for foreground/background (threshold 1 m).
4. **Distribution constraint (DC):** KL divergence between predicted bin probabilities and a reference symmetric distribution centered at the ground-truth height. The HTC-DC paper's selected configuration is Gaussian for foreground and uniform for background. It derives the reference distribution's scale from the probability assigned to the ground-truth bin, integrates the reference density over every bin, and minimizes KL divergence to those probabilities.

Why DC exists: a weighted mean can be numerically correct even when probability mass is diffuse or multimodal. The DC shapes probabilities into a coherent unimodal neighborhood around ground truth, so the expected value is encouraged to coincide with the distribution mode rather than being an accidental average of distant bins.

The generic code combines losses across decoder levels, weighting later/higher-resolution predictions more strongly ([loss implementation](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/htcdc.py#L374-L419)).

## Do building footprints enter the height model?

**Production atlas answer: no, not as a required height-network input.**

There are three different roles that are easy to conflate:

1. **Optional training/evaluation mask in generic HTC-DC:** the data loader can read `_BLG.tif` when `use_mask` is enabled. The original HTC-DC paper says footprints are included for testing/building-wise metrics. The head-tail target can be generated directly from nDSM (`height > 1 m`), so a footprint is not necessary for the standard learned gate.
2. **Optional experimental footprint-gated architecture:** the atlas inference model contains `htc_source: bf`, but this needs a mask argument and is not the default global call.
3. **Required raster-to-building postprocessing:** fused footprints are used after height-map inference. The code masks the predicted height raster by each polygon, chooses `np.max`, and assigns that value as building height; it takes uncertainty from the same pixel ([LoD1 assignment](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/make_lod1/main.py#L355-L371)).

Therefore, a precise description is: **GBA.Height is footprint-free image-to-raster inference; GBA.LoD1 is footprint-dependent raster-to-instance aggregation.**

## Model selection and cross-validation

### What they do

The released loader expects one explicit `train.txt` and one `val.txt`; it does not create folds ([split loading](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/dataloaders.py#L9-L46)). Training evaluates the validation set each epoch and saves separate best checkpoints for configured metrics, including `checkpoint_best_rmse.pth.tar` ([checkpoint logic](https://github.com/zhu-xlab/HTC-DC-Net/blob/adae55edc8be589757cec57f839d59a681d93364/train.py#L149-L187)). Atlas inference explicitly loads `checkpoint_best_rmse.pth.tar` ([inference selection](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/main.py#L26-L35)). This is **single validation-set checkpoint selection**, not cross-validation.

The generic HTC-DC repository also implements patience-based early stopping. Its example config currently specifies a 500-epoch maximum, batch size 32, EfficientNet-B0, and patience 20, whereas the original HTC-DC paper reports stopping after 10 unimproved epochs. Those example settings are not the atlas run, which the atlas paper says was B5 / batch 8 / 150 epochs.

### What they do not show

There is **no k-fold, spatial cross-validation, leave-one-city-out cross-validation, or repeated-seed model selection** in the atlas paper or released atlas code. The phrase "three folds" in the HTC-DC paper's distribution-constraint ablation means results are evaluated in three groups/settings; it is not evidence that the atlas production model used three-fold CV.

The repository's several split directory names are separate predefined benchmark/test splits, not a loop that trains k models and aggregates their validation scores.

## Uncertainty quantification, exactly

### Computation

Inference is performed over each 0.2 degree mosaic with 256 x 256 windows and **stride 128**. In a regular interior region, a pixel lies in four windows (top-left, top-right, bottom-left, bottom-right contexts). For each pixel, code accumulates:

- `sum(predictions)`;
- `count`; and
- `sum(predictions^2)`.

It outputs the overlap mean

`mean = sum(predictions) / count`

and population variance

`variance = sum(predictions^2) / count - mean^2`.

This is visible in the released inference loop and variance calculation ([stride-128 inference](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/d58507e22af2bd940de5f98da7ed6763bd32147c/infer_height/utils.py#L310-L338)). Border and window-grid behavior can yield fewer or duplicate-context predictions, so "up to four" is the safe wording used by the paper.

For each final building polygon, the pipeline picks the maximum mean-height pixel within the footprint and reports the variance at that exact pixel. It does not average uncertainty over the footprint.

### What this uncertainty means

This is a **context/tiling disagreement score**: how sensitive the prediction is to seeing the same pixel at different positions and with different surrounding context in overlapping windows. High variance flags patch-boundary/context instability.

It is not classical test-time augmentation in the usual sense because the code does not show flips, rotations, scales, color transforms, dropout, ensembles, or posterior sampling. The "augmentations" are overlapping crops/context shifts. It is also not a calibrated predictive variance in square meters, does not include irreducible label noise, and can be near zero when the model is consistently wrong. The released work provides no calibration curve, interval coverage, conformal calibration, or conversion from variance to a confidence interval.

### Practical implications

- Use this variance for relative quality ranking or flagging unstable pixels/buildings.
- Do not interpret `height +/- sqrt(var)` as a statistically valid 68% confidence interval without empirical calibration.
- Because LoD1 height uses the maximum pixel, it is sensitive to outliers. A high erroneous roof/tree prediction can dominate a polygon even if most pixels are good.
- For your model, consider reporting both overlap variance and an independently calibrated error estimate on geographically held-out cities.

## Reproducibility and code-audit caveats

1. The actual atlas `config.yaml`, checkpoint, `image_stats.pickle`, `ndsm_stats.pickle`, and 168-RoI train/validation lists are absent.
2. The repository's first inference helper uses stride 64, while the batch/global helper uses stride 128. The paper's "up to 4 predictions" matches the stride-128 global helper, not stride 64 (which gives up to 16 regular overlaps).
3. `make_lod1/main.py` names the maximum `median_value`, but the operation is unequivocally `np.max`; the paper also says maximum.
4. The public HTC-DC example config (B0, batch 32, max 500) is not the atlas paper's production setting (B5, batch 8, 150 epochs).
5. Some code paths are experimental or stale. For example, optional footprint-gated HTC exists even though global inference supplies no footprint. Broad `except:` in multiprocess inference can also hide errors.
6. The atlas height training split strategy is unspecified, preventing a reliable assessment of spatial leakage.

## Recommendations for improving your own training

Based on the mechanisms and gaps above, the highest-value checks are:

1. **Split geographically.** Hold out entire cities/RoIs (and ideally countries/continents) rather than random patches. Keep overlapping/source-neighbor patches in one split.
2. **Measure by height strata and geography.** Track all-pixel RMSE, building-pixel RMSE, background RMSE, per-building RMSE, and bins such as 0-1 m, 1-5 m, 5-15 m, 15-30 m, and >30 m.
3. **Check background domination.** If tall buildings are underestimated, compare the HTC binary separation or weighted/stratified sampling against a simple regression baseline.
4. **Inspect adaptive-bin health.** Plot bin edges and probabilities. Look for collapsed widths, diffuse/multimodal probabilities, saturation at `h_max`, and bins unused across most patches.
5. **Calibrate uncertainty.** On held-out cities, relate overlap variance to absolute error; use reliability curves and quantile/conformal calibration if intervals are needed.
6. **Audit target construction.** Verify DSM-DTM coregistration, nodata masks, vegetation removal, negative clipping, temporal mismatch between Planet imagery and LiDAR, and resampling. Label problems can dominate architectural gains.
7. **Test polygon aggregation choices.** Compare maximum, high percentile (e.g. 90th/95th), robust top-k mean, and interior-eroded footprint statistics. Maximum is vulnerable to a single bad pixel.
8. **Establish controlled baselines.** Train the same B5 encoder-decoder with plain L1/Huber regression, then add adaptive bins, HTC, and DC one at a time. This identifies which component actually helps your data.

## Direct answers checklist

- **How many PlanetScope scenes?** Approximately **800,000 acquired scenes** for global mosaicking; exact retained/contributing count not given.
- **Height-model input?** Normalized 3-channel PlanetScope patch, 256 x 256 at 3 m.
- **Target?** Co-registered LiDAR-derived nDSM patch, 256 x 256.
- **Training corpus?** 231,656 patches from 168 city-scale RoIs, mostly North America, Europe, and Oceania.
- **Exact architecture?** EfficientNet-B5 encoder-decoder + multi-level HTC-AdaBins (ViT, 256 adaptive bins, learned foreground/background handling) + probability-weighted bin-center regression.
- **Footprint used in the model?** Not in the released production inference path; optional in generic/experimental paths and used after inference for LoD1 aggregation.
- **Cross-validation?** No evidence of k-fold/spatial CV; predefined train/validation split and best-validation-RMSE checkpoint selection.
- **Uncertainty?** Variance across up to four overlapping stride-128 window predictions; LoD1 stores variance at the maximum-height pixel in each footprint.
- **Published atlas training schedule?** 150 epochs, batch 8, AdamW; paper does not report learning rate, split, augmentation, or actual run config.

