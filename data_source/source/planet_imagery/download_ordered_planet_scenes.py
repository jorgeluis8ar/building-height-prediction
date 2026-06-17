"""
Download Ordered Planet Scenes

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - data_source/data/planet_imagery/generated/planet_orders_manifest.csv
    - data_source/source/planet_imagery/PLANET_API.py or PL_API_KEY or saved Planet OAuth profile

Produces (outputs for later stages):
    - data_source/data/planet_imagery/source/<city_slug>/<season>_<scene_id>/
    - data_source/data/planet_imagery/generated/planet_orders_manifest.csv

Description:
    Reads Planet order IDs from the manifest, checks their current status, and
    downloads only orders that are complete. Running, queued, and failed orders
    are recorded in the manifest but are not downloaded.

Usage:
    python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --dry-run
    python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --confirm-download

Expected runtime: depends on order completion and download size
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Restart this script inside the local virtual environment."""
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()
    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing planet_imagery virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/planet_imagery/venv_planet_imagery")
        print("  data_source/source/planet_imagery/venv_planet_imagery/bin/python -m pip install -r data_source/source/planet_imagery/requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env)


relaunch_inside_venv()

import pandas as pd
from planet import Auth, Session
from planet.clients import OrdersClient


READY_STATES = {"success", "partial"}
NOT_READY_STATES = {"queued", "running"}
MANIFEST_COLUMNS = [
    "city_slug",
    "selection_season",
    "scene_id",
    "selected_asset_type",
    "product_bundle",
    "order_name",
    "order_id",
    "order_state",
    "created_on",
    "last_modified",
    "output_dir",
    "aoi_path",
    "order_request_json",
    "download_status",
    "downloaded_files",
    "download_checked_on",
]


def api_key_from_local_file() -> str | None:
    """
    Read a local, git-ignored Planet API key module when it exists.

    The expected file is PLANET_API.py in this folder with:
        PL_API_KEY = "..."
    """
    try:
        from PLANET_API import PL_API_KEY as local_api_key
    except ImportError:
        return None

    local_api_key = str(local_api_key).strip()
    return local_api_key or None


def auth_from_available_credentials() -> Auth:
    """Authenticate with local key first, then environment key, then OAuth."""
    api_key = api_key_from_local_file()
    if api_key:
        print("Authentication: local PLANET_API.py", flush=True)
        return Auth.from_key(api_key)

    api_key = os.environ.get("PL_API_KEY")
    if api_key:
        print("Authentication: PL_API_KEY environment variable", flush=True)
        return Auth.from_key(api_key)

    print("Authentication: saved Planet OAuth profile", flush=True)
    return Auth.from_user_default_session()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download completed Planet orders recorded in the manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "planet_orders_manifest.csv",
        help="Manifest written by order_selected_planet_scenes.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check order states without downloading files.",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Required to download completed orders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded files.",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        help="Optional cap for testing a small number of manifest rows.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the building-height-prediction repo root."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def project_relative_path(path: Path) -> str:
    """Store a portable path relative to the repository root."""
    return str(resolve_project_path(path).relative_to(PROJECT_ROOT))


def load_manifest(path: Path) -> pd.DataFrame:
    """Load the order manifest and add missing bookkeeping columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Planet order manifest: {path}")
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = {"order_id", "output_dir", "city_slug", "selection_season", "scene_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Order manifest is missing columns: {sorted(missing)}")
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""
    return manifest[MANIFEST_COLUMNS].copy()


def write_manifest(path: Path, manifest: pd.DataFrame) -> None:
    """Write the manifest atomically after status/download updates."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    manifest.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def joined_paths(paths: list[Any]) -> str:
    """Store downloaded file paths as one readable manifest cell."""
    return ";".join(str(path) for path in paths)


async def async_main() -> None:
    args = parse_args()
    args.manifest = resolve_project_path(args.manifest)

    if args.dry_run and args.confirm_download:
        raise SystemExit("Use either --dry-run or --confirm-download, not both.")
    if not args.dry_run and not args.confirm_download:
        raise SystemExit("Refusing to download. Use --dry-run or --confirm-download.")

    manifest = load_manifest(args.manifest)
    row_indexes = list(manifest.index)
    if args.max_orders is not None:
        row_indexes = row_indexes[: args.max_orders]

    print(f"Manifest rows to check: {len(row_indexes)}", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)
    auth = auth_from_available_credentials()

    async with Session(auth=auth) as session:
        orders_client = OrdersClient(session)
        for index in row_indexes:
            row = manifest.loc[index]
            order_id = str(row["order_id"]).strip()
            if not order_id:
                raise ValueError(f"Manifest row {index} is missing order_id")

            order = await orders_client.get_order(order_id)
            state = str(order.get("state", ""))
            manifest.loc[index, "order_state"] = state
            manifest.loc[index, "created_on"] = order.get("created_on", row.get("created_on", ""))
            manifest.loc[index, "last_modified"] = order.get("last_modified", "")
            manifest.loc[index, "download_checked_on"] = pd.Timestamp.now("UTC").isoformat()

            label = (
                f"{row['city_slug']} {row['selection_season']} "
                f"{row['scene_id']} order={order_id} state={state}"
            )
            print(label, flush=True)

            if state in NOT_READY_STATES:
                manifest.loc[index, "download_status"] = "not_ready"
                print("  skipped: order is not complete yet", flush=True)
                continue
            if state not in READY_STATES:
                manifest.loc[index, "download_status"] = f"not_downloaded_state_{state}"
                print("  skipped: order is not in a downloadable state", flush=True)
                continue
            if args.dry_run:
                manifest.loc[index, "download_status"] = "ready_dry_run"
                print("  ready: dry run only", flush=True)
                continue

            output_dir = resolve_project_path(Path(str(row["output_dir"])))
            manifest.loc[index, "output_dir"] = project_relative_path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            downloaded_paths = await orders_client.download_order(
                order_id,
                directory=output_dir,
                overwrite=args.overwrite,
                progress_bar=True,
            )
            manifest.loc[index, "download_status"] = "downloaded"
            manifest.loc[index, "downloaded_files"] = joined_paths(
                [project_relative_path(Path(path)) for path in downloaded_paths]
            )
            print(f"  downloaded_files={len(downloaded_paths)}", flush=True)

    write_manifest(args.manifest, manifest)
    print(f"UPDATED {args.manifest}", flush=True)


if __name__ == "__main__":
    asyncio.run(async_main())
