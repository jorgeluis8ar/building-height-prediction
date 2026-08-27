# Batch Inference For The 94 Cities

Use `predict_global_training_cities.py` to apply the selected portable HTC-DC
Net model to the eight PlanetScope scenes stored for every city.

## Outputs Per City

The script produces **nine building-height rasters**, all measured in meters
above local ground level:

1. Eight independent rasters, one prediction from each PlanetScope scene.
2. One combined raster containing the pixel-level median of the eight aligned
   predictions.

It also creates `valid_scene_count.tif`, a quality-control raster recording how
many scenes contributed to each combined pixel. This count raster is not a
height prediction and therefore is not included among the nine height rasters.

The output structure is:

```text
global_training_predictions_v1/
└── <city>/
    ├── scene_predictions/
    │   ├── <scene_01>_predicted_agl_m.tif
    │   ├── ...
    │   └── <scene_08>_predicted_agl_m.tif
    ├── <city>_median_predicted_agl_m_8scenes.tif
    └── <city>_valid_scene_count.tif
```

The median is the combined product because it is less sensitive than the mean
to a single scene affected by haze, cloud, shadow, or an unusual prediction.
The eight original predictions remain available for comparisons and robustness
analysis.

## Safe Run Sequence

Always run `--dry-run` first. It verifies the model checkpoint and requires
exactly eight unique surface-reflectance scenes per city. The full run writes
`inference_manifest.csv` after every scene and reuses completed predictions, so
an interrupted Windows run can be restarted with the same command.

The manifest records elapsed minutes per scene. After every prediction, the
terminal also prints an updated estimate of the remaining runtime based on the
observed speed of that computer.
