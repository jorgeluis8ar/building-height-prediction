#!/usr/bin/env python3
"""Build the auditable first-pass open-LiDAR inventory for all WUP cities.

This script deliberately separates confirmed coverage from coverage that still
needs a spatial API/portal query.  A city that has not yet been checked must
never be reported as having no LiDAR data.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CITY_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv"
)
USGS_PROJECT_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/height_labels/generated/usgs_3dep_projects.csv"
)
USGS_GLOBAL_AVAILABILITY_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/height_labels/generated/usgs_3dep_global_city_availability.csv"
)
USGS_GLOBAL_TILE_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/height_labels/generated/usgs_3dep_global_city_tile_metadata.csv"
)
NAMED_COUNTRY_AUDIT_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/height_labels/generated/named_country_open_lidar_city_audit.csv"
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data_source/data/height_labels/generated"
OUTPUT_FILE = OUTPUT_DIRECTORY / "global_open_lidar_city_inventory.csv"
SUMMARY_FILE = OUTPUT_DIRECTORY / "global_open_lidar_city_inventory_summary.csv"
VERIFIED_ON = date(2026, 8, 18).isoformat()


SOURCE_REGISTRY = {
    "Netherlands": {
        "coverage_status": "confirmed",
        "source_program": "Actueel Hoogtebestand Nederland (AHN)",
        "official_access_link": "https://www.ahn.nl/open-data",
        "product_type": "classified point cloud; DSM; DTM",
        "format": "LAZ; GeoTIFF",
        "license": "Open data; current AHN6 release CC BY 4.0",
        "acquisition_year": "multiple national cycles; AHN2-AHN6",
        "aoi_coverage_percent": "100",
        "verification_method": "Official AHN documentation states nationwide coverage and downloadable point clouds",
    },
    "Finland": {
        "coverage_status": "confirmed",
        "source_program": "National Land Survey of Finland laser-scanning data",
        "official_access_link": "https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data",
        "product_type": "classified point cloud; DTM",
        "format": "LAZ; GeoTIFF",
        "license": "NLS open-data licence",
        "acquisition_year": "2008-present national cycles",
        "aoi_coverage_percent": "100",
        "verification_method": "Official NLS product description states availability for all Finland",
    },
    "Spain": {
        "coverage_status": "confirmed",
        "source_program": "PNOA-LiDAR",
        "official_access_link": "https://centrodedescargas.cnig.es/CentroDescargas/home",
        "product_type": "classified point cloud; DSM; DTM",
        "format": "LAZ; raster elevation products",
        "license": "CC BY 4.0 with source attribution",
        "acquisition_year": "2008-2015; 2015-2021; 2022-2025",
        "aoi_coverage_percent": "100",
        "verification_method": "Official CNIG catalog identifies first and second national coverages",
    },
    "France": {
        "coverage_status": "query_required",
        "source_program": "IGN LiDAR HD",
        "official_access_link": "https://www.ign.fr/institut/programme-lidar-hd-vers-une-nouvelle-cartographie-3d-du-territoire",
        "product_type": "classified point cloud; DSM; DTM; height model",
        "format": "LAZ; GeoTIFF",
        "license": "French open data",
        "acquisition_year": "2021-present",
        "aoi_coverage_percent": "",
        "verification_method": "Official program is open but 2025 publication coverage exceeded 80%; city AOI intersection required",
    },
    "New Zealand": {
        "coverage_status": "query_required",
        "source_program": "LINZ National Elevation Programme",
        "official_access_link": "https://www.linz.govt.nz/products-services/data/types-linz-data/elevation-data/lidar-data-coverage",
        "product_type": "point cloud; DSM; DTM",
        "format": "LAZ; GeoTIFF",
        "license": "LINZ open data",
        "acquisition_year": "multiple regional surveys",
        "aoi_coverage_percent": "",
        "verification_method": "Official LINZ documentation reports most, but not all, national coverage; city AOI intersection required",
    },
    "United States of America": {
        "coverage_status": "query_required",
        "source_program": "USGS 3D Elevation Program (3DEP)",
        "official_access_link": "https://www.usgs.gov/tools/lidarexplorer",
        "product_type": "classified point cloud; DEM",
        "format": "LAZ; GeoTIFF",
        "license": "U.S. public domain",
        "acquisition_year": "2004-present",
        "aoi_coverage_percent": "",
        "verification_method": "USGS 3DEP API query required for each 5 km AOI",
    },
}


# The old downloader uses shorter pilot slugs.  Map those records onto their
# corresponding WUP urban-agglomeration names in the global city inventory.
USGS_SLUG_TO_CITY = {
    "new_york_city": "New York-Newark",
    "los_angeles": "Los Angeles-Long Beach-Santa Ana",
    "chicago": "Chicago",
    "boston": "Boston",
    "san_francisco": "San Francisco-Oakland",
    "seattle": "Seattle",
}


OUTPUT_FIELDS = [
    "wup_urbancode",
    "city_slug",
    "city_name",
    "country",
    "centroid_latitude",
    "centroid_longitude",
    "coverage_status",
    "source_program",
    "official_access_link",
    "product_type",
    "format",
    "license",
    "acquisition_year",
    "aoi_coverage_percent",
    "verification_date",
    "verification_method",
    "download_file_count",
    "download_endpoint_status",
    "representative_download_url",
    "representative_http_status",
    "access_constraint",
    "automation_level",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required CSV and fail immediately if it is absent or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Required input contains no records: {path}")
    return rows


def load_usgs_results() -> dict[str, dict[str, str]]:
    """Aggregate existing project-level USGS checks for the six pilot cities."""
    if not USGS_PROJECT_FILE.is_file():
        return {}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for project in read_csv(USGS_PROJECT_FILE):
        city_name = USGS_SLUG_TO_CITY.get(project["city_slug"])
        if city_name:
            grouped[city_name].append(project)

    results: dict[str, dict[str, str]] = {}
    for city_name, projects in grouped.items():
        years = []
        links = []
        percentages = []
        for project in projects:
            start = project.get("collect_start", "")
            end = project.get("collect_end", "")
            if start:
                years.append(start[:4])
            if end:
                years.append(end[:4])
            if project.get("lpc_link"):
                links.append(project["lpc_link"])
            if project.get("aoi_coverage_percent"):
                percentages.append(float(project["aoi_coverage_percent"]))

        # A single project percentage is exact. Multiple project footprints can
        # overlap, so summing them would overstate union coverage; leave the
        # percentage blank until the project geometries are unioned.
        exact_percentage = f"{percentages[0]:.6f}" if len(percentages) == 1 else ""
        status = "confirmed" if projects else "query_failed"
        results[city_name] = {
            "coverage_status": status,
            "source_program": "USGS 3D Elevation Program (3DEP)",
            "official_access_link": links[0] if links else SOURCE_REGISTRY["United States of America"]["official_access_link"],
            "product_type": "classified point cloud",
            "format": "LAZ",
            "license": "U.S. public domain",
            "acquisition_year": f"{min(years)}-{max(years)}" if years else "",
            "aoi_coverage_percent": exact_percentage,
            "verification_method": "Existing USGS 3DEP project and tile manifests generated by download_usgs_3dep_lidar.py",
            "notes": (
                f"{len(projects)} matching USGS project(s); union AOI coverage must be recalculated"
                if len(projects) > 1
                else "Existing project manifest contains a measured AOI coverage percentage"
            ),
        }
    return results


def load_global_usgs_results() -> dict[str, dict[str, str]]:
    """Load metadata-only USGS results for the expanded WUP U.S. city pool."""
    if not USGS_GLOBAL_AVAILABILITY_FILE.is_file():
        return {}

    publication_years: dict[str, list[str]] = defaultdict(list)
    if USGS_GLOBAL_TILE_FILE.is_file():
        for tile in read_csv(USGS_GLOBAL_TILE_FILE):
            publication_date = tile.get("publication_date", "")
            if publication_date and publication_date[:4].isdigit():
                publication_years[tile["city_slug"]].append(publication_date[:4])

    results = {}
    for row in read_csv(USGS_GLOBAL_AVAILABILITY_FILE):
        years = publication_years.get(row["city_slug"], [])
        if years:
            acquisition_period = f"products published {min(years)}-{max(years)}"
        else:
            acquisition_period = ""
        results[row["city_name"]] = {
            "coverage_status": row["coverage_status"],
            "source_program": "USGS 3D Elevation Program (3DEP)",
            "official_access_link": row["representative_download_url"],
            "product_type": "classified point cloud",
            "format": "LAZ",
            "license": "U.S. public domain",
            "acquisition_year": acquisition_period,
            "aoi_coverage_percent": row["aoi_coverage_percent"],
            "verification_method": (
                "Official National Map LPC metadata query, union of intersecting "
                "tile bounding boxes, and one-byte representative URL check"
            ),
            "notes": (
                f"{row['intersecting_tile_count']} intersecting tile records; "
                f"{row['project_count']} project directories; "
                f"{row['download_urls_present']} download URLs; representative "
                f"URL status={row['representative_url_status']} "
                f"HTTP={row['representative_http_status']}"
            ),
        }
    return results


def load_named_country_results() -> dict[str, dict[str, str]]:
    """Load the detailed 17-country audit keyed by the stable city slug."""
    if not NAMED_COUNTRY_AUDIT_FILE.is_file():
        return {}
    rows = read_csv(NAMED_COUNTRY_AUDIT_FILE)
    if len(rows) != 249:
        raise RuntimeError(
            f"Expected 249 named-country audit rows, found {len(rows):,}"
        )
    if len({row["city_slug"] for row in rows}) != len(rows):
        raise RuntimeError("Named-country audit contains duplicate city slugs")
    return {row["city_slug"]: row for row in rows}


def main() -> None:
    """Create the city inventory and a small status-by-country summary."""
    cities = read_csv(CITY_FILE)
    if len(cities) != 1862:
        raise RuntimeError(f"Expected 1,862 WUP cities, found {len(cities):,}")

    # New global-query results override the older six-city pilot records when
    # both exist. The expanded audit may intentionally omit those six pilots.
    usgs_results = load_usgs_results()
    usgs_results.update(load_global_usgs_results())
    named_country_results = load_named_country_results()
    output_rows = []
    for city in cities:
        source = dict(SOURCE_REGISTRY.get(city["country"], {}))
        if not source:
            source = {
                "coverage_status": "not_checked",
                "source_program": "",
                "official_access_link": "",
                "product_type": "",
                "format": "",
                "license": "",
                "acquisition_year": "",
                "aoi_coverage_percent": "",
                "verification_method": "No authoritative open-LiDAR source has yet been evaluated for this city",
            }
        notes = ""
        if city["city_slug"] in named_country_results:
            audited = named_country_results[city["city_slug"]]
            source = {field: audited.get(field, "") for field in OUTPUT_FIELDS}
            notes = source.pop("notes", "")
        elif city["country"] == "United States of America" and city["city_name"] in usgs_results:
            source = usgs_results[city["city_name"]]
            notes = source.pop("notes", "")

        output_rows.append(
            {
                "wup_urbancode": city["wup_urbancode"],
                "city_slug": city["city_slug"],
                "city_name": city["city_name"],
                "country": city["country"],
                "centroid_latitude": city["latitude"],
                "centroid_longitude": city["longitude"],
                **{field: source.get(field, "") for field in OUTPUT_FIELDS[6:-1]},
                "verification_date": VERIFIED_ON,
                "notes": notes,
            }
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_counts = Counter((row["country"], row["coverage_status"]) for row in output_rows)
    with SUMMARY_FILE.open("w", encoding="utf-8", newline="") as handle:
        fields = ["country", "coverage_status", "city_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (country, status), count in sorted(summary_counts.items()):
            writer.writerow({"country": country, "coverage_status": status, "city_count": count})

    status_counts = Counter(row["coverage_status"] for row in output_rows)
    print(f"Wrote {len(output_rows):,} cities to {OUTPUT_FILE.relative_to(REPOSITORY_ROOT)}")
    print(f"Wrote summary to {SUMMARY_FILE.relative_to(REPOSITORY_ROOT)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count:,}")


if __name__ == "__main__":
    main()
