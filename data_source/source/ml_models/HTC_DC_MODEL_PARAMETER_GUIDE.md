# HTC-DC Net Model Parameter Guide

This document explains every parameter used in the completed Full HTC-DC Net
RGB+NIR, EfficientNet-B5, height-weighted experiment. It describes what each
parameter controls, how changing it can affect RMSE, and which changes are most
useful for future experiments.

The reference run is:

```text
nyc76_la95_offnadir_rgbnir_4ch_full_htcdc_b5_binweighted_seed20260723_epoch50
```

Its selected checkpoint was epoch 15. The city-balanced validation building
RMSE was 11.6229 m, while pooled validation building RMSE was 8.4126 m.

## Important Principle

Changing a parameter does not guarantee a lower RMSE. A parameter should be
judged on held-out validation data, using the same split and metric. Training
RMSE alone is not a suitable selection criterion because a more flexible model
can memorize training chips while performing worse in new locations.

Parameters fall into four groups:

1. Data and input definitions determine what information reaches the model.
2. Architecture parameters determine model capacity and spatial reasoning.
3. Loss and optimization parameters determine what errors training emphasizes.
4. Evaluation parameters determine how the best checkpoint is selected.

## Dataset And Input Parameters

### Dataset

**Reference value:** `nyc_la_off_nadir_rgb_nir_full_recipe_v1`

The dataset defines the imagery, nDSM targets, building masks, normalization
statistics, and train/validation/test membership. It is the foundation of the
experiment rather than a conventional model hyperparameter.

**Effect on RMSE:** Dataset quality often has a larger effect than architecture
tuning. Better image-to-LiDAR temporal agreement, accurate co-registration,
representative height distributions, and consistent labels can reduce RMSE.
Misalignment or distribution shift can increase RMSE even when training loss is
low.

**How to tune safely:** Compare dataset versions with the same model settings.
Never choose a dataset using test RMSE. Check performance separately for LA and
NYC because pooled RMSE can conceal city-specific failure.

### Training Sample

**Reference value:** 76 NYC and 95 LA chips, 171 chips total.

The training sample determines which neighborhoods, building types, heights,
materials, shadows, and image conditions the model observes.

**Effect on RMSE:** More diverse and representative chips usually improve
generalization. Repeated exposure to rare NYC high-rises can reduce upper-tail
error, but aggressive oversampling can hurt common low-rise predictions or
cause overfitting. A city imbalance can make pooled RMSE look good while one
city performs poorly.

**How to tune safely:** Preserve spatially independent validation and test
chips. Consider city-balanced and height-aware batches, then compare pooled and
city-balanced validation RMSE.

### Input Channels

**Reference value:** 4.

This is the number of image bands supplied to the first convolutional layer.
The value must equal the number of bands in every image chip and the number of
normalization means and standard deviations.

**Effect on RMSE:** Additional informative channels can lower RMSE by providing
spectral or temporal evidence unavailable in RGB. Uninformative, noisy, or
poorly aligned channels can increase RMSE and computational cost. Changing the
channel count also changes the first layer, so results are not directly
attributable only to the added information.

**How to tune safely:** Compare controlled 3-channel RGB and 4-channel RGB+NIR
runs on identical splits. Validate alignment and normalization before training.

### Channel Order

**Reference value:** red, green, blue, NIR.

Channel order specifies the meaning of each input tensor position. The model
does not inherently know which band is red or NIR.

**Effect on RMSE:** A mismatch between the dataset order and normalization
statistics corrupts the input and can sharply increase RMSE. Reordering bands
consistently should not fundamentally change model capacity, although pretrained
RGB weights are most meaningful when the first three channels remain RGB.

**How to tune safely:** Treat this as a fixed contract, not an RMSE tuning
parameter. Store the order in the manifest and fail on mismatches.

### PlanetScope Inputs

**Reference value:** one off-nadir RGB+NIR scene per city.

The scene determines illumination, shadows, viewing geometry, season, surface
conditions, and temporal correspondence with LiDAR.

**Effect on RMSE:** A useful off-nadir angle and distinct shadows may provide
height cues. Excessive angle, haze, seasonal mismatch, temporal mismatch, or
occlusion can increase RMSE. One scene also makes the model vulnerable to
scene-specific radiometry.

**How to tune safely:** Compare scenes or multi-scene inputs while keeping the
split and architecture otherwise fixed. Record view angle, sun elevation,
capture time, and LiDAR date difference.

### Image Size

**Reference value:** 256 by 256 pixels.

Image size defines the spatial context processed in one training example. At
3 m resolution, a chip covers approximately 768 by 768 m.

**Effect on RMSE:** Larger chips provide more neighborhood and shadow context
but use more memory and may dilute small-building information. Smaller chips
increase the relative prominence of individual buildings but remove long
shadows and urban context. They also create more boundary effects.

**How to tune safely:** Test one smaller size, such as 128, while preserving
geographic splits. Do not assume smaller chips will automatically improve
high-rise performance.

### Spatial Resolution

**Reference value:** 3 by 3 m.

Resolution defines the ground area represented by each pixel. It is inherited
from the PlanetScope-aligned dataset.

**Effect on RMSE:** Finer resolution can better separate narrow buildings and
edges if genuinely supported by the imagery. Artificial upsampling does not
create new information and can produce false precision. Coarser resolution can
reduce noise but merges small buildings and mixed pixels, potentially raising
building-level RMSE.

**How to tune safely:** Keep 3 m as the native reference. Treat resampling
experiments as dataset changes and validate all grids again.

### Normalization

**Reference value:** per-channel statistics calculated from training chips
only.

Normalization centers and scales each spectral band so that optimization is
numerically stable and no channel dominates only because of its numeric range.

**Effect on RMSE:** Correct training-only normalization generally stabilizes
learning. Statistics computed from validation or test data cause leakage.
Incorrect statistics can suppress useful variation or create distribution
mismatches that increase RMSE.

**How to tune safely:** Keep normalization enabled and recompute statistics
whenever channels or the training split change.

## Architecture Parameters

### Model

**Reference value:** Full HTC-DC Net.

HTC-DC Net combines an image encoder, adaptive height-bin prediction,
multi-scale decoding, and head/tail constraints designed to improve building
height estimation across an imbalanced height distribution.

**Effect on RMSE:** The full architecture can model richer relationships than a
simple convolutional regressor, but its added capacity and loss interactions
can overfit a small dataset. A simpler model can achieve lower validation RMSE
when data are limited.

**How to tune safely:** Compare architectures using the same data and
checkpoint metric. Do not infer superiority from parameter count or training
RMSE.

### Backbone

**Reference value:** EfficientNet-B5.

The backbone extracts multi-scale visual features from the PlanetScope image.
B5 is larger and more computationally expensive than EfficientNet-B0.

**Effect on RMSE:** A larger backbone can learn more detailed spectral and
spatial patterns, potentially lowering RMSE when enough representative data
exist. With only 171 training chips, it can overfit, train slowly, or learn
scene-specific features. The previous B0 models remain important baselines.

**How to tune safely:** Compare B0 and B5 on identical inputs and splits. Select
using validation RMSE, and inspect the train-validation gap.

### Adaptive Height Bins

**Reference value:** 256.

The model represents continuous height through learned bin centers and
per-pixel probabilities. More bins permit a finer discretization of the height
range.

**Effect on RMSE:** Too few bins can quantize predictions and miss subtle height
differences. Too many bins can make learning unstable or leave rare tall-height
regions weakly supported. More bins do not ensure a wider predicted range.

**How to tune safely:** Test a small set such as 128 and 256. Examine both RMSE
and predicted-height range, especially above 50 m.

### Patch Size

**Reference value:** 4.

Patch size controls how spatial features are grouped before transformer-style
context processing. A patch of 4 summarizes local neighborhoods while reducing
the sequence length.

**Effect on RMSE:** Smaller patches preserve finer building boundaries but cost
more memory and may emphasize noise. Larger patches are cheaper and provide
coarser context but can blur small buildings and narrow shadows.

**How to tune safely:** Treat 4 as the reference. Test one adjacent value only
after confirming that small-building errors are a dominant source of RMSE.

### Fusion Mode

**Reference value:** `third`.

Fusion mode controls how decoder predictions from multiple feature levels are
combined. The `third` mode follows the selected full-recipe implementation and
uses four supervised decoder levels.

**Effect on RMSE:** Better multi-scale fusion can improve predictions for both
small low-rise buildings and large high-rises. Poor fusion can introduce noisy
or redundant signals. Its effect is architecture-specific and must be tested
empirically.

**How to tune safely:** Keep it fixed while tuning optimization parameters.
Only compare alternative modes as a separate architecture ablation.

### Early HTC

**Reference value:** enabled.

Early head-tail cut applies height-distribution constraints at intermediate
decoder stages rather than only at the final output. This provides additional
supervision during feature learning.

**Effect on RMSE:** It can help intermediate features distinguish background,
common heights, and extreme heights. If applied too strongly or using noisy
intermediate predictions, it can constrain the model prematurely and increase
RMSE.

**How to tune safely:** Compare enabled and disabled variants with all other
loss settings fixed. Monitor high-rise and low-rise bins separately.

### HTC Threshold

**Reference value:** 1.0 m.

The threshold separates near-zero/background heights from positive foreground
heights for the head-tail constraints.

**Effect on RMSE:** A threshold that is too low can treat noise as buildings. A
threshold that is too high can classify genuine low structures as background,
raising low-rise RMSE. Since the project assigns positive buildings a minimum
height of 2.4 m, the 1 m threshold leaves a margin below valid building labels.

**How to tune safely:** Test small changes only if diagnostics show confusion
near zero. Keep the threshold below the minimum positive building label.

### HTC Source

**Reference value:** predicted height.

This setting determines whether HTC grouping is derived from the model's own
predictions or another available signal.

**Effect on RMSE:** Prediction-based grouping makes the constraint available at
inference and consistent with model behavior, but early prediction errors can
feed into the constraint. A target-derived source may train more easily but can
create a training-inference mismatch.

**How to tune safely:** Keep the prediction source as the production reference.
Use alternatives only as controlled ablations.

## Loss Parameters

### Foreground Probability Loss

**Reference value:** Gaussian.

This loss encourages the adaptive-bin probability distribution for building
pixels to concentrate around the target height with a Gaussian-like shape.

**Effect on RMSE:** It can produce smoother, ordered probability estimates and
reduce erratic height predictions. If its spread or influence is poorly suited
to the labels, it can over-smooth rare high-rise targets and compress the
prediction range.

**How to tune safely:** Examine predicted maxima and height-bin residuals. If
high-rises remain compressed, compare against a weaker or alternative
foreground probability constraint.

### Background Probability Loss

**Reference value:** uniform.

This term describes the desired adaptive-bin probability behavior for
background pixels, where no positive building height is expected.

**Effect on RMSE:** It can prevent background pixels from dominating learned
height distributions. Because primary metrics are building-level, background
loss should not overwhelm foreground learning. An overly strong background
objective may favor conservative, low-variance predictions.

**How to tune safely:** Keep the formulation fixed while checking unmasked
predictions over water and non-building areas. Tune its coefficient separately
if background artifacts interfere with shared feature learning.

### Height-Loss Weighting

**Reference value:** bin-weighted positive-height L1.

The absolute error for a positive target pixel is multiplied by a weight based
on its target-height interval. This changes which height ranges contribute most
to gradient updates.

**Effect on RMSE:** Weighting can reduce errors in rare low-rise or high-rise
groups that ordinary loss underemphasizes. Excessive weights can worsen overall
RMSE by sacrificing common mid-rise cases, create unstable gradients, or
overfit a small number of extreme buildings.

**How to tune safely:** Change weights gradually and compare per-bin RMSE,
pooled validation RMSE, and city-balanced validation RMSE. The weighted B5 run
improved the checkpoint metric modestly but did not solve upper-tail
compression.

### Height-Bin Edges

**Reference value:** 3, 6, 10, 25, and 50 m.

These edges define six loss-weight intervals: below 3 m, 3-6 m, 6-10 m,
10-25 m, 25-50 m, and 50 m or more. They do not define the model's 256 adaptive
output bins.

**Effect on RMSE:** Edges determine which targets receive each weight. Poorly
placed edges can combine buildings with very different error behavior or create
bins containing too few observations. This can make optimization noisy.

**How to tune safely:** Base edges on training-data distributions and policy-
relevant height groups, then hold them fixed while testing weights. Do not use
test outcomes to redefine bins.

### Height-Bin Weights

**Reference value:** 4, 3, 2, 1, 3, and 8.

The values correspond in order to the six intervals defined above. A 60 m
target therefore receives eight times the positive-height L1 weight of a
10-25 m target.

**Effect on RMSE:** Larger high-rise weights can increase attention to rare tall
buildings. Larger low-rise weights can reduce the model's lower prediction
floor. Very large weights can cause a few pixels to dominate training and hurt
calibration elsewhere.

**How to tune safely:** Useful next tests are modestly stronger high-rise
weights or smoother ratios between adjacent bins. Report unweighted validation
RMSE even when the training loss is weighted.

### Chamfer Weight

**Reference value:** 0.01.

The Chamfer term encourages learned adaptive height-bin centers to cover the
distribution of target heights. Its coefficient controls the contribution of
this auxiliary loss relative to the other objectives.

**Effect on RMSE:** A suitable value can improve coverage of the height range.
Too little influence may allow redundant bin centers; too much can prioritize
global bin placement over accurate per-pixel prediction and raise RMSE.

**How to tune safely:** Test nearby values such as 0.005 and 0.02 after the
primary loss is stable. Inspect learned-bin coverage and predicted maximum, not
only aggregate RMSE.

### Background-Loss Weight

**Reference value:** 0.0 for the custom background L1 term.

This coefficient controls an additional direct height-regression penalty on
background pixels. It is separate from the uniform background probability
loss.

**Effect on RMSE:** A positive value can suppress false heights over water and
other background areas. If too large, the numerous background pixels can
dominate gradients, pull predictions toward zero, and reduce building-height
variance. Because the main metric uses buildings, zero is a defensible baseline.

**How to tune safely:** If unmasked background predictions remain problematic,
test a very small value such as 0.01 or 0.02 while monitoring building RMSE and
prediction standard deviation.

## Optimization Parameters

### Maximum Epochs

**Reference value:** 50.

An epoch is one pass through the training loader. Maximum epochs sets the
training budget, not the checkpoint that must be used.

**Effect on RMSE:** Too few epochs can underfit. Too many can lower training
loss while validation RMSE worsens. In this run, epoch 15 outperformed epoch
50, demonstrating that more epochs did not mean a better selected model.

**How to tune safely:** Keep a sufficiently large maximum but select the best
checkpoint using validation RMSE. Add early stopping to avoid unnecessary
training after sustained deterioration.

### Selected Epoch

**Reference value:** 15.

This is the checkpoint with the lowest city-balanced validation building RMSE,
not a manually chosen hyperparameter.

**Effect on RMSE:** Choosing the final epoch instead of the best validation
epoch can increase held-out error. Repeatedly selecting among many checkpoints
can itself overfit the validation set, so the selection rule must be fixed in
advance.

**How to tune safely:** Continue selecting by the predefined validation metric.
Evaluate the untouched test split only after selection.

### Learning Rate

**Reference value:** 0.0001.

The learning rate controls the size of each optimizer update.

**Effect on RMSE:** A rate that is too high can cause unstable loss, collapse,
or oscillation around a good solution. A rate that is too low may converge too
slowly or remain underfit within 50 epochs. Learning-rate scheduling can allow
fast early progress and finer late-stage refinement.

**How to tune safely:** Compare values such as 0.00003 and 0.0001, or add a
decay schedule. Use identical seeds and splits when possible.

### Batch Size

**Reference value:** 8.

Batch size is the number of chips contributing to one gradient update.

**Effect on RMSE:** Larger batches produce smoother gradients but fewer updates
per epoch and may generalize differently. Smaller batches introduce gradient
noise that can regularize training but may be unstable. Batch composition also
matters because LA chips can dominate common low-rise patterns.

**How to tune safely:** Keep 8 unless memory or stability requires a change.
Height- and city-balanced sampling may be more valuable than simply increasing
batch size.

### Optimizer

**Reference value:** AdamW.

AdamW adapts the update size for each parameter and applies decoupled weight
decay. It is commonly effective for convolutional and transformer components.

**Effect on RMSE:** AdamW usually converges reliably, but optimizer choice
interacts with learning rate and regularization. Switching optimizers without
retuning the learning rate can increase RMSE.

**How to tune safely:** Retain AdamW as the baseline. Prioritize learning rate,
weight decay, and scheduling before trying another optimizer.

### Weight Decay

**Reference value:** 0.01.

Weight decay penalizes large parameter values and acts as regularization.

**Effect on RMSE:** Moderate weight decay can reduce overfitting and lower
validation RMSE. Too much causes underfitting and overly smooth predictions;
too little allows the B5 backbone to memorize training neighborhoods.

**How to tune safely:** Compare 0.001, 0.01, and possibly 0.05 while monitoring
the train-validation gap and prediction range.

### Augmentation

**Reference value:** spatial and spectral augmentation.

Spatial augmentation rotates or flips imagery and targets together. Spectral
augmentation perturbs RGB+NIR radiometry without modifying masks or nDSM
geometry.

**Effect on RMSE:** Appropriate augmentation reduces dependence on orientation
and scene-specific brightness, potentially improving validation RMSE. Excessive
spectral changes can destroy meaningful shadow or reflectance cues. Geometric
augmentation that is not synchronized corrupts labels.

**How to tune safely:** Keep spatial transformations synchronized. Compare the
current profile with weaker spectral gains if image radiometry is already
stable.

### Seed

**Reference value:** 20260723.

The seed controls reproducible random initialization, data order, and
augmentation draws where supported.

**Effect on RMSE:** Different seeds can produce different RMSE values,
especially with a small dataset. The seed does not systematically improve the
model, but a single favorable seed can exaggerate apparent gains.

**How to tune safely:** Do not select a seed because it performs best. Confirm
important parameter improvements across multiple predetermined seeds.

## Inference And Evaluation Parameters

### Checkpoint Criterion

**Reference value:** city-balanced validation building RMSE, calculated as the
average of LA and NYC building RMSE.

This criterion gives each city equal influence even though LA contains many
more building components.

**Effect on RMSE:** It does not change model gradients, but it determines which
checkpoint is reported. Pooled RMSE would favor LA because LA has more
buildings. City balancing better reflects the goal of performing well in both
cities, though it may select a checkpoint with slightly worse pooled RMSE.

**How to tune safely:** Keep this rule fixed before training. Report both pooled
and city-specific metrics alongside it.

### Inference Window

**Reference value:** 256 by 256 pixels.

The inference window is the tile size passed through the network when making
large-area predictions. It matches the training chip size.

**Effect on RMSE:** Matching the training size avoids a scale/context mismatch.
Smaller windows may lose context; larger windows may behave differently from
training and consume substantially more memory.

**How to tune safely:** Keep 256 unless the model is retrained with another chip
size.

### Inference Stride

**Reference value:** 128 pixels.

Stride controls the distance between adjacent inference windows. A 128-pixel
stride creates 50% overlap for 256-pixel windows.

**Effect on RMSE:** Overlap averaging reduces edge artifacts and can stabilize
predictions. A smaller stride increases overlap and computation, with
diminishing returns. A stride equal to the window size is faster but can create
visible seams and higher boundary error.

**How to tune safely:** Keep 128 as the reference. Test denser overlap only if
boundary diagnostics identify a material source of error.

### Collapse Guard

**Reference value:** enabled.

The collapse guard detects chips whose predictions have implausibly low
spatial variation. It is a diagnostic safeguard rather than a training loss.

**Effect on RMSE:** The guard does not directly lower RMSE, but it prevents a
near-constant model from being mistaken for a successful run. Stopping or
flagging collapse saves computation and protects model comparisons.

**How to tune safely:** Keep it enabled. Choose thresholds based on known valid
prediction variability and inspect flagged chips rather than automatically
discarding them.

## Recommended RMSE Experiments

The following experiments are the most informative next steps. Change one
factor at a time and retain the current split and checkpoint rule.

| Priority | Experiment | Why it may lower RMSE |
|---:|---|---|
| 1 | Add a learning-rate decay schedule | Epoch 15 was best while later training fluctuated; decay may refine without overshooting. |
| 2 | Compare EfficientNet-B0 and B5 under the same full loss | B5 may be over-capacity for 171 chips. |
| 3 | Tune high-rise weights moderately | NYC and the upper tail remain the dominant errors. |
| 4 | Test city- and height-balanced batches | Prevent LA low-rise examples from dominating gradient updates. |
| 5 | Tune weight decay | May reduce B5 overfitting and improve validation RMSE. |
| 6 | Test 128 versus 256 adaptive bins | May improve stability and usable height-range coverage. |
| 7 | Compare weaker foreground probability smoothing | May reduce high-rise prediction compression. |

Every experiment should report pooled, LA, NYC, and city-balanced validation
RMSE, followed by one final test evaluation only after the configuration and
checkpoint have been selected.
