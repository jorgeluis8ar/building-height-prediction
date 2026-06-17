"""
Order Selected Planet Scenes

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/<city_slug>_5km.geojson
    - data_source/source/planet_imagery/PLANET_API.py or PL_API_KEY or saved Planet OAuth profile

Produces (outputs for later stages):
    - data_source/data/planet_imagery/generated/planet_orders_manifest.csv

Description:
    Builds AOI-clipped Planet Orders API requests for reviewed scene IDs. The
    script only creates orders when --confirm-order is supplied. It never
    downloads imagery. Order IDs and metadata are saved to a manifest so the
    download step can be run later, after Planet finishes processing.

Usage:
    python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --dry-run
    python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --confirm-order

Expected runtime: depends on the number of orders created
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from planet import Auth, Session, order_request
from planet.clients import DataClient, OrdersClient


PREFERRED_ASSET_TYPES = [
    "ortho_analytic_8b_sr",
    "ortho_analytic_8b",
    "ortho_analytic_4b_sr",
    "ortho_analytic_4b",
]
ASSET_TYPE_TO_PRODUCT_BUNDLE = {
    "ortho_analytic_8b_sr": "analytic_8b_sr_udm2",
    "ortho_analytic_8b": "analytic_8b_udm2",
    "ortho_analytic_4b_sr": "analytic_sr_udm2",
    "ortho_analytic_4b": "analytic_udm2",
}
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
        description="Create AOI-clipped Planet orders for selected scenes."
    )
    parser.add_argument(
        "--selected-scenes",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "selected_planet_city_scenes.csv",
        help="CSV created by select_planet_city_scenes.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "planet_orders_manifest.csv",
        help="Output manifest that records Planet order IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "source",
        help="Base folder where downloaded imagery will later be stored.",
    )
    parser.add_argument(
        "--aoi-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data_source"
        / "data"
        / "city_aois"
        / "generated"
        / "city_buffers_5km_by_city",
        help="Folder containing <city_slug>_5km.geojson AOI files for order clipping.",
    )
    parser.add_argument(
        "--asset-type",
        action="append",
        dest="asset_types",
        help="Override selected_asset_type using this preferred asset type order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate rows and print planned orders without creating Planet orders.",
    )
    parser.add_argument(
        "--confirm-order",
        action="store_true",
        help="Required to create Planet orders.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Optional cap for testing a small number of selected scenes.",
    )
    return parser.parse_args()


def load_existing_manifest(path: Path) -> pd.DataFrame:
    """Load the order manifest if it exists, otherwise return an empty table."""
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = pd.read_csv(path)
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""
    return manifest[MANIFEST_COLUMNS].copy()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the repository root."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def project_relative_path(path: Path) -> str:
    """Store a portable path relative to the repository root."""
    return str(resolve_project_path(path).relative_to(PROJECT_ROOT))


def write_manifest(path: Path, manifest: pd.DataFrame) -> None:
    """Write the manifest atomically so a partial write does not corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    manifest.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def choose_asset_type(available_assets: dict[str, Any], preferred_asset_types: list[str]) -> str:
    """Pick the first preferred asset type that Planet says is available."""
    for asset_type in preferred_asset_types:
        if asset_type in available_assets:
            return asset_type
    raise ValueError(
        "None of the preferred asset types are available. "
        f"Preferred: {preferred_asset_types}; available: {sorted(available_assets)}"
    )


def chosen_asset_type_for_row(
    row: pd.Series,
    available_assets: dict[str, Any],
    override_asset_types: list[str] | None,
) -> str:
    """Use the reviewed CSV asset type unless the command line overrides it."""
    if override_asset_types:
        return choose_asset_type(available_assets, override_asset_types)

    selected_asset_type = row.get("selected_asset_type")
    if pd.isna(selected_asset_type) or not selected_asset_type:
        return choose_asset_type(available_assets, PREFERRED_ASSET_TYPES)
    if selected_asset_type not in available_assets:
        raise ValueError(
            f"Selected asset type {selected_asset_type!r} is not available for "
            f"scene {row['id']}. Available: {sorted(available_assets)}"
        )
    return str(selected_asset_type)


def product_bundle_for_asset_type(asset_type: str) -> str:
    """Translate a Data API asset type into an Orders API product bundle."""
    try:
        return ASSET_TYPE_TO_PRODUCT_BUNDLE[asset_type]
    except KeyError as error:
        raise ValueError(
            f"No Orders API product bundle mapping is defined for asset type {asset_type!r}"
        ) from error


def load_aoi_geometry(aoi_path: Path) -> dict[str, Any]:
    """Read the city 5km AOI GeoJSON geometry used by the Planet clip tool."""
    if not aoi_path.exists():
        raise FileNotFoundError(f"Missing AOI GeoJSON for Planet order clip: {aoi_path}")
    with aoi_path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if document.get("type") == "FeatureCollection":
        features = document.get("features") or []
        if len(features) != 1:
            raise ValueError(f"Expected exactly one AOI feature in {aoi_path}")
        geometry = features[0].get("geometry")
    elif document.get("type") == "Feature":
        geometry = document.get("geometry")
    else:
        geometry = document

    if not geometry:
        raise ValueError(f"Missing AOI geometry in {aoi_path}")
    return geometry


def build_clipped_order_request(
    city_slug: str,
    season: str,
    scene_id: str,
    product_bundle: str,
    aoi_geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build one Planet Orders API request clipped to the city AOI."""
    order_name = f"{city_slug}_{season}_{scene_id}_{product_bundle}"
    return order_request.build_request(
        name=order_name,
        products=[
            order_request.product(
                item_ids=[scene_id],
                product_bundle=product_bundle,
                item_type="PSScene",
            )
        ],
        tools=[order_request.clip_tool(aoi_geometry)],
    )


def manifest_key(row: pd.Series) -> tuple[str, str, str, str]:
    """Identify whether this exact scene/product order was already recorded."""
    return (
        str(row["city_slug"]),
        str(row["selection_season"]),
        str(row["scene_id"]),
        str(row["product_bundle"]),
    )


async def async_main() -> None:
    args = parse_args()
    args.selected_scenes = resolve_project_path(args.selected_scenes)
    args.manifest = resolve_project_path(args.manifest)
    args.output_dir = resolve_project_path(args.output_dir)
    args.aoi_dir = resolve_project_path(args.aoi_dir)

    if not args.selected_scenes.exists():
        raise FileNotFoundError(f"Missing selected scenes CSV: {args.selected_scenes}")
    if args.dry_run and args.confirm_order:
        raise SystemExit("Use either --dry-run or --confirm-order, not both.")
    if not args.dry_run and not args.confirm_order:
        raise SystemExit("Refusing to create orders. Use --dry-run or --confirm-order.")

    selected = pd.read_csv(args.selected_scenes)
    required = {"city_slug", "id", "selection_season"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Selected scenes CSV is missing columns: {sorted(missing)}")

    if args.max_scenes is not None:
        selected = selected.head(args.max_scenes).copy()

    manifest = load_existing_manifest(args.manifest)
    existing_keys = set()
    if not manifest.empty:
        existing_keys = {
            manifest_key(row)
            for _, row in manifest.dropna(subset=["order_id"]).iterrows()
            if str(row.get("order_id", "")).strip()
        }

    print(f"Selected scenes to review: {len(selected)}", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)
    auth = auth_from_available_credentials()

    created_order_count = 0
    async with Session(auth=auth) as session:
        data_client = DataClient(session)
        orders_client = OrdersClient(session)
        for _, row in selected.iterrows():
            city_slug = str(row["city_slug"])
            scene_id = str(row["id"])
            season = str(row["selection_season"])
            aoi_path = args.aoi_dir / f"{city_slug}_5km.geojson"
            scene_dir = args.output_dir / city_slug / f"{season}_{scene_id}"

            assets = await data_client.list_item_assets("PSScene", scene_id)
            asset_type = chosen_asset_type_for_row(row, assets, args.asset_types)
            product_bundle = product_bundle_for_asset_type(asset_type)
            aoi_geometry = load_aoi_geometry(aoi_path)
            request = build_clipped_order_request(
                city_slug=city_slug,
                season=season,
                scene_id=scene_id,
                product_bundle=product_bundle,
                aoi_geometry=aoi_geometry,
            )

            key = (city_slug, season, scene_id, product_bundle)
            print(
                f"{city_slug} {season} {scene_id}: "
                f"asset={asset_type}, bundle={product_bundle}, clipped_order=true",
                flush=True,
            )
            if key in existing_keys:
                print("  skipped: matching order already exists in manifest", flush=True)
                continue
            if args.dry_run:
                print(f"  dry_run_order_name={request['name']}", flush=True)
                continue

            order = await orders_client.create_order(request)
            print(f"  created_order={order['id']} state={order.get('state', '')}", flush=True)
            new_row = {
                "city_slug": city_slug,
                "selection_season": season,
                "scene_id": scene_id,
                "selected_asset_type": asset_type,
                "product_bundle": product_bundle,
                "order_name": request["name"],
                "order_id": order["id"],
                "order_state": order.get("state", ""),
                "created_on": order.get("created_on", ""),
                "last_modified": order.get("last_modified", ""),
                "output_dir": project_relative_path(scene_dir),
                "aoi_path": project_relative_path(aoi_path),
                "order_request_json": json.dumps(request, sort_keys=True),
            }
            manifest = pd.concat([manifest, pd.DataFrame([new_row])], ignore_index=True)
            write_manifest(args.manifest, manifest[MANIFEST_COLUMNS])
            existing_keys.add(key)
            created_order_count += 1
            print(
                f"  manifest_saved=true total_manifest_orders={len(manifest)}",
                flush=True,
            )

    if args.confirm_order:
        if created_order_count:
            print(f"WROTE {args.manifest} ({len(manifest)} total orders)", flush=True)
        else:
            write_manifest(args.manifest, manifest[MANIFEST_COLUMNS])
            print(f"No new orders created. Manifest unchanged: {args.manifest}", flush=True)


if __name__ == "__main__":
    asyncio.run(async_main())
