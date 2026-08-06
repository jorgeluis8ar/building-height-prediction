"""
Prepare HTC-DC Net Dataset Statistics

HTC-DC Net expects `image_stats.pickle` and `ndsm_stats.pickle` at the dataset
root, saved with `torch.save`. Our dataset builder also stores richer stats
under `stats/`, but this script creates the exact root-level files the upstream
dataloader/model read with `torch.load`.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATASET_DIR = PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"


def load_pickle(path: Path):
    """Load one pickle file."""
    with path.open("rb") as file:
        return pickle.load(file)


def main() -> None:
    """Create root-level torch-compatible HTC statistics files."""
    try:
        import torch
    except ModuleNotFoundError:
        print("ERROR: PyTorch is required to write HTC-compatible stats files.")
        print("Install the smoke-test environment first:")
        print(
            "  python3 -m venv data_source/source/ml_models/venv_htc_dc_net\n"
            "  data_source/source/ml_models/venv_htc_dc_net/bin/python -m pip install "
            "-r data_source/source/ml_models/htc_dc_net_setup/requirements-apple-silicon-smoke.txt"
        )
        sys.exit(1)

    image_stats = load_pickle(DATASET_DIR / "stats/image_stats.pickle")
    ndsm_stats = load_pickle(DATASET_DIR / "stats/ndsm_stats.pickle")

    image_mean = image_stats["image_mean"]
    image_std = image_stats["image_std"]
    ndsm_mean = ndsm_stats["ndsm_positive_mean"]
    ndsm_std = ndsm_stats["ndsm_positive_std"]
    ndsm_min = 0.0
    ndsm_max = float(ndsm_mean + 6 * ndsm_std)
    count = torch.zeros(int(max(1, round(ndsm_max))) + 1)

    torch.save([image_mean, image_std], DATASET_DIR / "image_stats.pickle")
    torch.save([ndsm_mean, ndsm_std, ndsm_min, ndsm_max, count], DATASET_DIR / "ndsm_stats.pickle")

    print(f"Wrote {DATASET_DIR / 'image_stats.pickle'}")
    print(f"Wrote {DATASET_DIR / 'ndsm_stats.pickle'}")


if __name__ == "__main__":
    main()
