"""
Plan and create AOI-clipped Planet orders for a validated training-city set.

Environment: data_source/source/planet_imagery/venv_planet_imagery

Safety model:
    - ``--dry-run`` is local only: it validates all inputs and writes/updates
      the order manifest without authenticating or creating orders.
    - ``--confirm-order`` is the only mode that may create Planet orders.
    - Only rows marked ``training`` are accepted. A separate city inventory can
      provide this metadata for selector outputs that contain scene fields only.
    - One order is built per city, with scene IDs grouped by product bundle.
    - Exact deterministic order names allow recovery after interruption.
    - Existing order IDs and terminal status fields are never discarded.

Default input:
    data_source/data/planet_imagery/generated/global_scene_selection_split/
        training_scene_order_input.csv

Default manifest:
    data_source/data/planet_imagery/generated/global_training_orders/
        planet_training_city_orders_manifest.csv

This script never downloads imagery.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"

EXPECTED_TRAINING_CITIES = 711
EXPECTED_SPLIT_SEED = 419453
MAX_PLANET_ITEMS_PER_ORDER = 500
CREATE_RETRIES = 5

ASSET_TYPE_TO_PRODUCT_BUNDLE = {
    "ortho_analytic_8b_sr": "analytic_8b_sr_udm2",
    "ortho_analytic_4b_sr": "analytic_sr_udm2",
}

MANIFEST_COLUMNS = [
    "split_seed", "split_group", "randomized_city_rank", "city_slug",
    "city_name", "country", "selected_scene_count", "scene_ids_json",
    "scene_ids_sha256", "asset_types_json", "product_bundles_json",
    "aoi_path", "output_dir", "order_name", "order_request_sha256",
    "order_request_json", "plan_status", "order_id", "order_state",
    "order_created_on", "order_last_modified", "order_submission_status",
    "order_submission_attempts", "order_submission_checked_utc",
    "order_error", "download_status", "downloaded_files_json",
    "download_bytes", "download_checked_utc", "download_error",
]

TERMINAL_ORDER_STATES = {"success", "partial", "failed", "cancelled"}


class StreamTee:
    """Mirror stdout/stderr to the terminal and one honest dated run log."""

    def __init__(self, terminal, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = terminal
        self.file = path.open("w", encoding="utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


def start_dated_log(directory: Path, prefix: str) -> Path:
    """Capture all subsequent prints and tracebacks in one UTC-dated log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / "logs" / f"{prefix}_{timestamp}.log"
    tee = StreamTee(sys.__stdout__, path)
    sys.stdout = tee
    sys.stderr = tee

    def restore_and_close() -> None:
        """Restore live terminal streams before closing the mirrored file."""
        if sys.stdout is tee:
            sys.stdout = tee.terminal
        if sys.stderr is tee:
            sys.stderr = sys.__stderr__
        tee.close()

    atexit.register(restore_and_close)
    return path


def relaunch_inside_venv() -> None:
    """Relaunch inside the project environment before third-party imports."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(
            "ERROR: Missing Planet imagery environment. Recreate it from "
            "data_source/source/planet_imagery/requirements.txt."
        )
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

import httpx
import pandas as pd
from planet import Auth, Session, order_request
from planet.clients import OrdersClient


def parse_args() -> argparse.Namespace:
    """Define explicit confirmation and bounded resumable batch options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-scenes",
        type=Path,
        default=PROJECT_ROOT
        / (
            "data_source/data/planet_imagery/generated/global_scene_selection_split/"
            "training_scene_order_input.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / (
            "data_source/data/planet_imagery/generated/global_training_orders/"
            "planet_training_city_orders_manifest.csv"
        ),
    )
    parser.add_argument(
        "--aoi-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/source/global_training",
    )
    parser.add_argument(
        "--city-inventory",
        type=Path,
        default=None,
        help=(
            "Optional city CSV supplying city_name, country, split metadata, and "
            "randomized_city_rank when those fields are absent from selected scenes."
        ),
    )
    parser.add_argument("--expected-city-count", type=int, default=EXPECTED_TRAINING_CITIES)
    parser.add_argument(
        "--expected-scenes-per-city",
        type=int,
        default=0,
        help="Require exactly this many unique scenes per city; zero disables the check.",
    )
    parser.add_argument("--split-seed", type=int, default=EXPECTED_SPLIT_SEED)
    parser.add_argument(
        "--order-name-prefix",
        default="bhp_train",
        help="Deterministic Planet order-name prefix.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Local validation and manifest planning only; no authentication or orders.",
    )
    mode.add_argument(
        "--confirm-order",
        action="store_true",
        help="Create or recover Planet orders for the bounded city batch.",
    )
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit",
        type=int,
        default=25,
        help="Cities submitted in this call. Use 0 for all remaining cities.",
    )
    parser.add_argument("--request-pause", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=CREATE_RETRIES)
    return parser.parse_args()


def resolve_project_path(path: Path, *, output: bool = False) -> Path:
    """Resolve project-relative paths and reject paths outside the repository."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        role = "Output" if output else "Input"
        raise ValueError(f"{role} path is outside the project repository: {resolved}")
    return resolved


def portable_path(path: Path) -> str:
    """Return a Windows/Mac-portable repository-relative path."""
    return str(resolve_project_path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")


def utc_now() -> str:
    """Return one ISO-8601 UTC timestamp for auditable manifest updates."""
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashes and manifest review."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Hash deterministic plan content."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_aoi_geometry(path: Path) -> dict[str, Any]:
    """Read one city AOI and return its single clip geometry."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing global 5 km AOI: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("type") == "FeatureCollection":
        features = document.get("features") or []
        if len(features) != 1:
            raise ValueError(f"Expected one AOI feature in {path}; found {len(features)}")
        geometry = features[0].get("geometry")
    elif document.get("type") == "Feature":
        geometry = document.get("geometry")
    else:
        geometry = document
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Invalid AOI clip geometry in {path}")
    return geometry


def build_city_order_request(
    city_slug: str,
    rows: pd.DataFrame,
    aoi: dict[str, Any],
    split_seed: int,
    order_name_prefix: str,
) -> dict[str, Any]:
    """Build one strict clipped request containing all selected city scenes."""
    products = []
    for asset_type, group in rows.groupby("selected_asset_type", sort=True):
        if asset_type not in ASSET_TYPE_TO_PRODUCT_BUNDLE:
            raise ValueError(f"Unsupported selected asset type for {city_slug}: {asset_type}")
        item_ids = sorted(group["scene_id"].astype(str).unique())
        products.append(
            order_request.product(
                item_ids=item_ids,
                product_bundle=ASSET_TYPE_TO_PRODUCT_BUNDLE[asset_type],
                item_type="PSScene",
            )
        )
    scene_ids = sorted(rows["scene_id"].astype(str).unique())
    name_digest = sha256_text("|".join(scene_ids))[:10]
    order_name = f"{order_name_prefix}_s{split_seed}_{city_slug}_{name_digest}"
    return order_request.build_request(
        name=order_name,
        products=products,
        order_type="full",
        tools=[order_request.clip_tool(aoi)],
        stac={},
    )


def build_plan(
    selected_path: Path,
    aoi_dir: Path,
    download_root: Path,
    city_inventory_path: Path | None,
    expected_city_count: int,
    expected_scenes_per_city: int,
    split_seed: int,
    order_name_prefix: str,
) -> pd.DataFrame:
    """Convert the training scene table into exactly one planned row per city."""
    if not selected_path.is_file():
        raise FileNotFoundError(f"Missing training scene input: {selected_path}")
    selected = pd.read_csv(
        selected_path,
        dtype={"city_slug": str, "scene_id": str, "split_group": str},
    )
    required = {"city_slug", "scene_id", "selection_rank", "selected_asset_type"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Training scene input is missing columns: {missing}")
    if selected.empty or selected[["city_slug", "scene_id"]].duplicated().any():
        raise ValueError("Training scene input is empty or has duplicate city/scene rows")
    if city_inventory_path is not None:
        if not city_inventory_path.is_file():
            raise FileNotFoundError(f"Missing city inventory: {city_inventory_path}")
        inventory = pd.read_csv(city_inventory_path, dtype={"city_slug": str})
        inventory_required = {
            "city_slug", "city_name", "country", "split_group", "split_seed",
            "randomized_city_rank",
        }
        inventory_missing = sorted(inventory_required - set(inventory.columns))
        if inventory_missing:
            raise ValueError(f"City inventory is missing columns: {inventory_missing}")
        if inventory["city_slug"].duplicated().any():
            raise ValueError("City inventory contains duplicate city_slug rows")
        metadata_columns = [
            "city_slug", "city_name", "country", "split_group", "split_seed",
            "randomized_city_rank",
        ]
        selected = selected.drop(
            columns=[column for column in metadata_columns[1:] if column in selected.columns]
        ).merge(inventory[metadata_columns], on="city_slug", how="left", validate="many_to_one")

    metadata_required = {
        "city_name", "country", "split_group", "split_seed", "randomized_city_rank"
    }
    metadata_missing = sorted(metadata_required - set(selected.columns))
    if metadata_missing:
        raise ValueError(
            f"Selected scenes lack city/order metadata {metadata_missing}; pass --city-inventory."
        )
    if selected[list(metadata_required)].isna().any().any():
        raise ValueError("Selected scenes contain blank city/order metadata after inventory join")
    if set(selected["split_group"].astype(str).unique()) != {"training"}:
        raise ValueError("Refusing input containing validation/testing scenes")
    seeds = set(pd.to_numeric(selected["split_seed"], errors="raise").astype(int))
    if seeds != {split_seed}:
        raise ValueError(f"Expected split seed {split_seed}; found {sorted(seeds)}")
    if selected["city_slug"].nunique() != expected_city_count:
        raise ValueError(
            f"Expected {expected_city_count} training cities; found "
            f"{selected['city_slug'].nunique()}"
        )
    if expected_scenes_per_city:
        counts = selected.groupby("city_slug")["scene_id"].nunique()
        bad = counts[counts != expected_scenes_per_city]
        if not bad.empty:
            raise ValueError(
                f"Expected {expected_scenes_per_city} scenes per city; mismatches: "
                f"{bad.head(20).to_dict()}"
            )

    records: list[dict[str, Any]] = []
    for city_slug, rows in selected.groupby("city_slug", sort=False):
        rows = rows.sort_values(["selection_rank", "scene_id"])
        city_values = rows[["city_name", "country", "randomized_city_rank"]].drop_duplicates()
        if len(city_values) != 1:
            raise ValueError(f"Conflicting city metadata for {city_slug}")
        aoi_path = aoi_dir / f"{city_slug}_5km.geojson"
        aoi = load_aoi_geometry(aoi_path)
        request = build_city_order_request(
            city_slug, rows, aoi, split_seed, order_name_prefix
        )
        scene_ids = sorted(rows["scene_id"].astype(str).unique())
        if len(scene_ids) > MAX_PLANET_ITEMS_PER_ORDER:
            raise ValueError(f"{city_slug} exceeds Planet's 500-item order limit")
        assets = {
            asset: sorted(group["scene_id"].astype(str).unique())
            for asset, group in rows.groupby("selected_asset_type", sort=True)
        }
        city = city_values.iloc[0]
        request_json = canonical_json(request)
        records.append(
            {
                "split_seed": split_seed,
                "split_group": "training",
                "randomized_city_rank": int(city["randomized_city_rank"]),
                "city_slug": city_slug,
                "city_name": city["city_name"],
                "country": city["country"],
                "selected_scene_count": len(scene_ids),
                "scene_ids_json": canonical_json(scene_ids),
                "scene_ids_sha256": sha256_text(canonical_json(scene_ids)),
                "asset_types_json": canonical_json(assets),
                "product_bundles_json": canonical_json(
                    sorted(ASSET_TYPE_TO_PRODUCT_BUNDLE[key] for key in assets)
                ),
                "aoi_path": portable_path(aoi_path),
                "output_dir": portable_path(download_root / city_slug),
                "order_name": request["name"],
                "order_request_sha256": sha256_text(request_json),
                "order_request_json": request_json,
                "plan_status": "validated",
            }
        )
    plan = pd.DataFrame.from_records(records).sort_values("randomized_city_rank").reset_index(drop=True)
    if len(plan) != expected_city_count or int(plan["selected_scene_count"].sum()) != len(selected):
        raise AssertionError("City order plan does not reconcile to training scene input")
    return plan


def load_existing_manifest(path: Path) -> pd.DataFrame:
    """Load prior API/download state without treating an absent file as failure."""
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    existing = pd.read_csv(path, dtype=str).fillna("")
    missing = sorted(set(MANIFEST_COLUMNS) - set(existing.columns))
    if missing:
        raise ValueError(f"Existing global order manifest is missing columns: {missing}")
    if existing["city_slug"].duplicated().any():
        raise ValueError("Existing global order manifest has duplicate city rows")
    return existing[MANIFEST_COLUMNS].copy()


def reconcile_manifest(plan: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Refresh deterministic plan fields while preserving external state fields."""
    state_columns = [column for column in MANIFEST_COLUMNS if column not in plan.columns]
    if existing.empty:
        for column in state_columns:
            plan[column] = ""
        return plan[MANIFEST_COLUMNS]
    extra = sorted(set(existing["city_slug"]) - set(plan["city_slug"]))
    if extra:
        raise ValueError(f"Manifest contains cities absent from current training plan: {extra[:10]}")
    old = existing.set_index("city_slug", drop=False)
    rows = []
    for _, planned in plan.iterrows():
        city_slug = planned["city_slug"]
        merged = {column: planned.get(column, "") for column in MANIFEST_COLUMNS}
        if city_slug in old.index:
            prior = old.loc[city_slug]
            if prior["order_request_sha256"] and prior["order_request_sha256"] != planned["order_request_sha256"]:
                if prior["order_id"]:
                    raise ValueError(
                        f"Order plan changed for already-submitted city {city_slug}; "
                        "refusing to overwrite external state"
                    )
            for column in state_columns:
                merged[column] = prior[column]
        rows.append(merged)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def atomic_write_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Atomically checkpoint all plan and API state after every city."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def auth_from_available_credentials() -> Auth:
    """Use a local key, environment key, or the saved Planet OAuth profile."""
    try:
        from PLANET_API import PL_API_KEY as local_key
    except ImportError:
        local_key = ""
    if str(local_key).strip():
        print("Authentication: local PLANET_API.py", flush=True)
        return Auth.from_key(str(local_key).strip())
    environment_key = os.environ.get("PL_API_KEY", "").strip()
    if environment_key:
        print("Authentication: PL_API_KEY environment variable", flush=True)
        return Auth.from_key(environment_key)
    print("Authentication: saved Planet OAuth profile", flush=True)
    return Auth.from_user_default_session()


async def find_exact_order(orders: OrdersClient, name: str) -> list[dict[str, Any]]:
    """Find an exact existing name so crash recovery cannot duplicate an order."""
    matches = []
    async for order in orders.list_orders(name=name, limit=100):
        if str(order.get("name", "")) == name:
            matches.append(order)
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Planet orders have deterministic name {name!r}")
    return matches


def record_order(manifest: pd.DataFrame, index: int, order: dict[str, Any], status: str) -> None:
    """Store the external order identity and current state on one manifest row."""
    manifest.loc[index, "order_id"] = str(order.get("id", ""))
    manifest.loc[index, "order_state"] = str(order.get("state", ""))
    manifest.loc[index, "order_created_on"] = str(order.get("created_on", ""))
    manifest.loc[index, "order_last_modified"] = str(order.get("last_modified", ""))
    manifest.loc[index, "order_submission_status"] = status
    manifest.loc[index, "order_submission_checked_utc"] = utc_now()
    manifest.loc[index, "order_error"] = ""


async def submit_with_retries(
    orders: OrdersClient,
    request: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    """Create one order with exponential backoff for transient HTTP failures."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await orders.create_order(request)
        except httpx.HTTPStatusError as error:
            last_error = error
            status = error.response.status_code
            if status not in {408, 429, 500, 502, 503, 504} or attempt == max_retries:
                raise
            retry_after = error.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
            print(f"  transient_create_status={status} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
        except httpx.HTTPError as error:
            last_error = error
            if attempt == max_retries:
                raise
            delay = min(60, 2 ** attempt)
            print(f"  transient_create_error={type(error).__name__} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError("Order retry loop ended unexpectedly") from last_error


async def async_main() -> int:
    """Plan all orders locally, then optionally submit one bounded city batch."""
    args = parse_args()
    if (
        args.city_offset < 0
        or args.city_limit < 0
        or args.request_pause < 0
        or args.max_retries < 1
        or args.expected_city_count < 1
        or args.expected_scenes_per_city < 0
    ):
        raise ValueError("Offsets, limits, pauses, and retries must be nonnegative/positive")
    selected_path = resolve_project_path(args.selected_scenes)
    manifest_path = resolve_project_path(args.manifest, output=True)
    log_path = start_dated_log(manifest_path.parent, "order_planet_training_city_scenes")
    print(f"Run log: {log_path}", flush=True)
    aoi_dir = resolve_project_path(args.aoi_dir)
    download_root = resolve_project_path(args.download_root, output=True)
    city_inventory_path = (
        resolve_project_path(args.city_inventory) if args.city_inventory is not None else None
    )
    if not aoi_dir.is_dir():
        raise FileNotFoundError(f"Missing global AOI directory: {aoi_dir}")

    plan = build_plan(
        selected_path=selected_path,
        aoi_dir=aoi_dir,
        download_root=download_root,
        city_inventory_path=city_inventory_path,
        expected_city_count=args.expected_city_count,
        expected_scenes_per_city=args.expected_scenes_per_city,
        split_seed=args.split_seed,
        order_name_prefix=args.order_name_prefix,
    )
    manifest = reconcile_manifest(plan, load_existing_manifest(manifest_path))
    atomic_write_manifest(manifest, manifest_path)
    total_scenes = int(pd.to_numeric(manifest["selected_scene_count"]).sum())
    print(f"Validated plan: {len(manifest)} training-city orders, {total_scenes} scenes", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    if args.dry_run:
        print("DRY RUN COMPLETE: no authentication and no Planet orders created", flush=True)
        return 0

    indexes = list(manifest.index[args.city_offset:])
    if args.city_limit:
        indexes = indexes[: args.city_limit]
    print(f"Confirmed submission batch: {len(indexes)} cities", flush=True)
    auth = auth_from_available_credentials()
    failures = 0
    async with Session(auth=auth) as session:
        orders = OrdersClient(session)
        for batch_number, index in enumerate(indexes, start=1):
            row = manifest.loc[index]
            city_slug = row["city_slug"]
            print(f"[{batch_number}/{len(indexes)}] {city_slug}", flush=True)
            try:
                if str(row["order_id"]).strip():
                    print(f"  skipped: manifest already records order {row['order_id']}", flush=True)
                    continue
                matches = await find_exact_order(orders, str(row["order_name"]))
                if matches:
                    record_order(manifest, index, matches[0], "recovered_existing_by_exact_name")
                    print(f"  recovered_order={matches[0].get('id')}", flush=True)
                else:
                    attempts = int(str(row["order_submission_attempts"] or "0")) + 1
                    manifest.loc[index, "order_submission_attempts"] = attempts
                    manifest.loc[index, "order_submission_status"] = "submitting"
                    manifest.loc[index, "order_submission_checked_utc"] = utc_now()
                    atomic_write_manifest(manifest, manifest_path)
                    request = json.loads(row["order_request_json"])
                    order = await submit_with_retries(orders, request, args.max_retries)
                    record_order(manifest, index, order, "created")
                    print(f"  created_order={order.get('id')} state={order.get('state')}", flush=True)
                atomic_write_manifest(manifest, manifest_path)
            except Exception as error:
                failures += 1
                manifest.loc[index, "order_submission_status"] = "failed"
                manifest.loc[index, "order_submission_checked_utc"] = utc_now()
                manifest.loc[index, "order_error"] = f"{type(error).__name__}: {error}"
                atomic_write_manifest(manifest, manifest_path)
                print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            if args.request_pause and batch_number < len(indexes):
                await asyncio.sleep(args.request_pause)
    if failures:
        print(f"FAILED: {failures} city order submissions failed; successful rows are checkpointed", flush=True)
        return 1
    print("SUCCESS: confirmed city batch is fully recorded in the manifest", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(async_main()))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
