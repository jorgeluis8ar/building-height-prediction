"""
Extract the WUP 2018 urban agglomerations with population over 300,000.

Environment: data_source/source/city_aois/venv_city_aois

Requires (read-only source input):
    - data_source/data/city_aois/source/
      WUP2018-F22-Cities_Over_300K_Annual_V7.xls

Produces (output used by the global AOI stage):
    - data_source/data/city_aois/generated/
      wup2018_cities_over_300k_2018.csv
    - data_source/data/city_aois/generated/logs/
      extract_wup2018_cities_over_300k_<UTC timestamp>.log

Population values in the WUP workbook are expressed in thousands. Therefore,
the strict condition "population over 300,000" is POP2018 > 300. The year and
threshold are command-line options so the definition remains explicit.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import traceback
import unicodedata


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_city_aois"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "CITY_AOIS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Relaunch with the pinned task environment before importing xlrd."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(
            "ERROR: Missing city AOI virtual environment. Create it from "
            "data_source/source/city_aois/requirements.txt."
        )
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

import xlrd


def parse_args() -> argparse.Namespace:
    """Define an auditable population year, threshold, and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-year", type=int, default=2018)
    parser.add_argument(
        "--minimum-population",
        type=int,
        default=300_000,
        help="Strict lower population bound in people; equality is excluded.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data_source/data/city_aois/source/WUP2018-F22-Cities_Over_300K_Annual_V7.xls",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    """Return a portable ASCII filename component."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", ascii_value)).strip("_")


def write_log(log_path: Path, lines: list[str], status: str) -> None:
    """Write an honest run log even when extraction fails."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join([f"status={status}", *lines, ""]), encoding="utf-8")


def main() -> None:
    """Read the source workbook, filter rows, validate them, and write CSV."""
    args = parse_args()
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    log_path = PROJECT_ROOT / "data_source/data/city_aois/generated/logs" / f"extract_wup2018_cities_over_300k_{timestamp}.log"
    log_lines = [f"started_utc={started.isoformat()}", f"input={args.input.relative_to(PROJECT_ROOT)}"]

    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Missing required WUP workbook: {args.input}")
        if args.minimum_population < 0:
            raise ValueError("minimum-population cannot be negative")

        workbook = xlrd.open_workbook(args.input)
        if "Data" not in workbook.sheet_names():
            raise ValueError("WUP workbook has no 'Data' sheet")
        sheet = workbook.sheet_by_name("Data")
        headers = [str(sheet.cell_value(0, column)).strip() for column in range(sheet.ncols)]
        population_column = f"POP{args.population_year}"
        required = [
            "Country or area", "urbancode", "Urban Agglomeration",
            "Latitude", "Longitude", population_column,
        ]
        missing = [column for column in required if column not in headers]
        if missing:
            raise ValueError(f"WUP workbook is missing required columns: {missing}")
        columns = {name: headers.index(name) for name in required}
        threshold_thousands = args.minimum_population / 1_000.0

        rows: list[dict[str, object]] = []
        for row_number in range(1, sheet.nrows):
            population_cell = sheet.cell_value(row_number, columns[population_column])
            if population_cell == "":
                continue
            population_thousands = float(population_cell)
            if population_thousands <= threshold_thousands:
                continue

            urban_code = int(float(sheet.cell_value(row_number, columns["urbancode"])))
            city = str(sheet.cell_value(row_number, columns["Urban Agglomeration"])).strip()
            country = str(sheet.cell_value(row_number, columns["Country or area"])).strip()
            if not city or not country:
                raise ValueError(f"Selected workbook row {row_number + 1} lacks a city or country")
            rows.append(
                {
                    "wup_urbancode": urban_code,
                    "city_slug": f"{slugify(city)}_{urban_code}",
                    "city_name": city,
                    "country": country,
                    "latitude": float(sheet.cell_value(row_number, columns["Latitude"])),
                    "longitude": float(sheet.cell_value(row_number, columns["Longitude"])),
                    "population_year": args.population_year,
                    "population_thousands": population_thousands,
                    "population_people": int(round(population_thousands * 1_000)),
                    "selection_rule": f"{population_column}>{threshold_thousands:g}_thousand",
                }
            )

        if not rows:
            raise ValueError("The population filter selected zero cities")
        codes = [row["wup_urbancode"] for row in rows]
        slugs = [row["city_slug"] for row in rows]
        if len(codes) != len(set(codes)) or len(slugs) != len(set(slugs)):
            raise ValueError("WUP urban codes or generated city slugs are not unique")

        rows.sort(key=lambda row: (-int(row["population_people"]), str(row["country"]), str(row["city_name"])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(args.output)

        log_lines.extend(
            [
                f"population_year={args.population_year}",
                f"minimum_population_people_strict={args.minimum_population}",
                f"selected_city_count={len(rows)}",
                f"output={args.output.relative_to(PROJECT_ROOT)}",
            ]
        )
        write_log(log_path, log_lines, "SUCCESS")
        print(f"SUCCESS: wrote {len(rows):,} cities to {args.output}")
        print(f"Run log: {log_path}")
    except Exception:
        log_lines.append(traceback.format_exc())
        write_log(log_path, log_lines, "FAILED")
        print(f"FAILED: see {log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
