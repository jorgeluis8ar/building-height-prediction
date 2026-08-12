"""
Check and download completed global Planet training-city orders on Windows.

Environment: data_source/source/planet_imagery/venv_planet_imagery

The script reads the city-level manifest written by
``order_planet_training_city_scenes.py``. ``--dry-run`` checks and checkpoints
Planet order states but never downloads. ``--confirm-download`` downloads only
orders in the ``success`` state. Partial orders are never accepted silently.

Downloads are resumable by city. A previously downloaded order is skipped only
when every file listed in its Planet ``manifest.json`` still exists. Each
download is followed by the same completeness check before the central order
manifest is marked ``downloaded``.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"
DOWNLOAD_RETRIES = 5
STATUS_RETRIES = 5


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
    """Relaunch inside the task environment before third-party imports."""
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
from planet import Auth, Session
from planet.clients import OrdersClient


def parse_args() -> argparse.Namespace:
    """Define safe modes and bounded resumable download batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / (
            "data_source/data/planet_imagery/generated/global_training_orders/"
            "planet_training_city_orders_manifest.csv"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check states only; download nothing.")
    mode.add_argument("--confirm-download", action="store_true", help="Download successful orders.")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit",
        type=int,
        default=10,
        help="Manifest cities checked in this call. Use 0 for all remaining cities.",
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    parser.add_argument("--max-retries", type=int, default=DOWNLOAD_RETRIES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path, *, output: bool = False) -> Path:
    """Resolve portable paths from the repository and reject outside paths."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        role = "Output" if output else "Input"
        raise ValueError(f"{role} path is outside the project repository: {resolved}")
    return resolved


def portable_path(path: Path) -> str:
    """Store a path relative to the repository for cross-computer use."""
    return str(resolve_project_path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> pd.DataFrame:
    """Require the complete city-level order/download schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing global training order manifest: {path}")
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "split_group", "city_slug", "order_id", "order_state", "output_dir",
        "download_status", "downloaded_files_json", "download_bytes",
        "download_checked_utc", "download_error",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Global training order manifest is missing columns: {missing}")
    if manifest.empty or manifest["city_slug"].duplicated().any():
        raise ValueError("Global training order manifest is empty or has duplicate cities")
    if set(manifest["split_group"].unique()) != {"training"}:
        raise ValueError("Refusing a manifest containing validation/testing cities")
    return manifest


def atomic_write_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Checkpoint status and download state after every city."""
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


def planet_manifest_files(output_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return delivered manifest files and every file they require."""
    manifest_paths = sorted(output_dir.rglob("manifest.json")) if output_dir.is_dir() else []
    expected: list[Path] = []
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Planet delivery manifest: {manifest_path}") from error
        for entry in payload.get("files", []):
            relative = entry.get("path")
            if relative:
                expected.append(manifest_path.parent / relative)
    return manifest_paths, expected


def verify_download(output_dir: Path) -> tuple[bool, list[Path], int, str]:
    """Accept a download only when Planet manifests list complete local files."""
    manifests, expected = planet_manifest_files(output_dir)
    if not manifests:
        return False, [], 0, "no Planet manifest.json found"
    if not expected:
        return False, manifests, 0, "Planet manifest lists no files"
    missing = [path for path in expected if not path.is_file()]
    if missing:
        return False, [*manifests, *expected], 0, f"{len(missing)} manifest files are missing"
    paths = sorted(set([*manifests, *expected]))
    total_bytes = sum(path.stat().st_size for path in paths)
    return True, paths, total_bytes, ""


async def get_order_with_retries(
    orders: OrdersClient,
    order_id: str,
    max_retries: int,
) -> dict[str, Any]:
    """Get current order state with rate-limit-aware exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            return await orders.get_order(order_id)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in {408, 429, 500, 502, 503, 504} or attempt == max_retries:
                raise
            retry_after = error.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
            print(f"  transient_status_check={status} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
        except httpx.HTTPError as error:
            if attempt == max_retries:
                raise
            delay = min(60, 2 ** attempt)
            print(f"  transient_status_error={type(error).__name__} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError("Status retry loop ended unexpectedly")


async def download_with_retries(
    orders: OrdersClient,
    order_id: str,
    output_dir: Path,
    max_retries: int,
    overwrite: bool,
) -> list[Path]:
    """Download one city order with bounded retries and no progress-bar noise."""
    for attempt in range(1, max_retries + 1):
        try:
            return await orders.download_order(
                order_id,
                directory=output_dir,
                overwrite=overwrite or attempt > 1,
                progress_bar=False,
            )
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in {408, 429, 500, 502, 503, 504} or attempt == max_retries:
                raise
            retry_after = error.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
            print(f"  transient_download_status={status} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
        except httpx.HTTPError as error:
            if attempt == max_retries:
                raise
            delay = min(60, 2 ** attempt)
            print(f"  transient_download_error={type(error).__name__} retry_in={delay}s", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError("Download retry loop ended unexpectedly")


async def async_main() -> int:
    """Check or download one bounded manifest batch with honest checkpointing."""
    args = parse_args()
    if args.city_offset < 0 or args.city_limit < 0 or args.minimum_free_gb < 0 or args.max_retries < 1:
        raise ValueError("Offsets, limits, disk threshold, and retries must be valid")
    manifest_path = resolve_project_path(args.manifest, output=True)
    log_path = start_dated_log(manifest_path.parent, "download_planet_training_city_orders")
    print(f"Run log: {log_path}", flush=True)
    manifest = load_manifest(manifest_path)
    indexes = list(manifest.index)
    if args.city_slugs:
        requested = set(args.city_slugs)
        known = set(manifest["city_slug"])
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown city slugs in download request: {unknown}")
        indexes = [index for index in indexes if manifest.loc[index, "city_slug"] in requested]
    indexes = indexes[args.city_offset:]
    if args.city_limit:
        indexes = indexes[: args.city_limit]
    print(f"Manifest city orders to check: {len(indexes)}", flush=True)
    auth = auth_from_available_credentials()
    failures = 0
    async with Session(auth=auth) as session:
        orders = OrdersClient(session)
        for batch_number, index in enumerate(indexes, start=1):
            row = manifest.loc[index]
            city_slug = row["city_slug"]
            order_id = str(row["order_id"]).strip()
            print(f"[{batch_number}/{len(indexes)}] {city_slug} order={order_id or '<missing>'}", flush=True)
            if not order_id:
                manifest.loc[index, "download_status"] = "not_ordered"
                manifest.loc[index, "download_checked_utc"] = utc_now()
                atomic_write_manifest(manifest, manifest_path)
                print("  skipped: no order ID", flush=True)
                continue
            try:
                order = await get_order_with_retries(orders, order_id, args.max_retries)
                state = str(order.get("state", ""))
                manifest.loc[index, "order_state"] = state
                manifest.loc[index, "order_created_on"] = str(order.get("created_on", row.get("order_created_on", "")))
                manifest.loc[index, "order_last_modified"] = str(order.get("last_modified", ""))
                manifest.loc[index, "download_checked_utc"] = utc_now()
                print(f"  state={state}", flush=True)

                output_dir = resolve_project_path(Path(row["output_dir"]), output=True)
                complete, verified_paths, total_bytes, verification_error = verify_download(output_dir)
                if complete and not args.overwrite:
                    manifest.loc[index, "download_status"] = "downloaded_verified"
                    manifest.loc[index, "downloaded_files_json"] = json.dumps(
                        [portable_path(path) for path in verified_paths], separators=(",", ":")
                    )
                    manifest.loc[index, "download_bytes"] = total_bytes
                    manifest.loc[index, "download_error"] = ""
                    atomic_write_manifest(manifest, manifest_path)
                    print(f"  skipped: verified existing download ({total_bytes:,} bytes)", flush=True)
                    continue
                if state == "partial":
                    manifest.loc[index, "download_status"] = "blocked_partial_order"
                    manifest.loc[index, "download_error"] = "Partial orders require manual review; not downloaded"
                    atomic_write_manifest(manifest, manifest_path)
                    print("  BLOCKED: partial order requires manual review", flush=True)
                    failures += 1
                    continue
                if state != "success":
                    manifest.loc[index, "download_status"] = f"not_ready_{state or 'unknown'}"
                    manifest.loc[index, "download_error"] = ""
                    atomic_write_manifest(manifest, manifest_path)
                    print("  skipped: order is not successful yet", flush=True)
                    continue
                if args.dry_run:
                    manifest.loc[index, "download_status"] = "ready_dry_run"
                    manifest.loc[index, "download_error"] = verification_error
                    atomic_write_manifest(manifest, manifest_path)
                    print("  ready: dry run downloads nothing", flush=True)
                    continue

                output_dir.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(output_dir).free
                minimum_bytes = int(args.minimum_free_gb * 1_000_000_000)
                if free_bytes < minimum_bytes:
                    raise OSError(
                        f"Free disk {free_bytes / 1e9:.2f} GB is below required "
                        f"{args.minimum_free_gb:.2f} GB"
                    )
                await download_with_retries(
                    orders, order_id, output_dir, args.max_retries, args.overwrite
                )
                complete, verified_paths, total_bytes, verification_error = verify_download(output_dir)
                if not complete:
                    raise IOError(f"Downloaded order failed manifest verification: {verification_error}")
                manifest.loc[index, "download_status"] = "downloaded_verified"
                manifest.loc[index, "downloaded_files_json"] = json.dumps(
                    [portable_path(path) for path in verified_paths], separators=(",", ":")
                )
                manifest.loc[index, "download_bytes"] = total_bytes
                manifest.loc[index, "download_error"] = ""
                atomic_write_manifest(manifest, manifest_path)
                print(f"  downloaded and verified: {total_bytes:,} bytes", flush=True)
            except Exception as error:
                failures += 1
                manifest.loc[index, "download_status"] = "failed"
                manifest.loc[index, "download_checked_utc"] = utc_now()
                manifest.loc[index, "download_error"] = f"{type(error).__name__}: {error}"
                atomic_write_manifest(manifest, manifest_path)
                print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
    if failures:
        print(f"FAILED: {failures} city downloads/status checks need review", flush=True)
        return 1
    print("SUCCESS: bounded download/status batch completed and checkpointed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(async_main()))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
