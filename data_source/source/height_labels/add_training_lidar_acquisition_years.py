#!/usr/bin/env python3
"""Add defensible LiDAR acquisition dates to the 94-city training list.

This program makes metadata-only requests.  It never downloads a point cloud,
DSM, DTM, Planet scene, or other imagery product.  Exact dates are retained
when an official catalogue exposes them.  Otherwise the program records the
narrowest documented campaign range and labels that lower precision clearly.
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CSV = REPOSITORY_ROOT / (
    "data_source/data/height_labels/generated/training_open_lidar/"
    "training_cities_with_open_lidar.csv"
)
USGS_TILES = REPOSITORY_ROOT / (
    "data_source/data/height_labels/generated/"
    "usgs_3dep_global_city_tile_metadata.csv"
)
USGS_INDEX = (
    "https://index.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevationIndex/MapServer/8/query"
)
CANADA_INDEX = (
    "https://maps-cartes.services.geo.ca/server_serveur/rest/services/"
    "NRCan/lidar_point_cloud_canelevation_en/MapServer/1/query"
)
ENGLAND_WFS = (
    "https://environment.data.gov.uk/geoservices/datasets/"
    "9f0fa3fc-a860-4729-adc9-47fe53f658d0/wfs"
)
ENGLAND_LAYER = (
    "dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:"
    "LIDAR_Point_Cloud_Index_Catalogue"
)

NEW_FIELDS = [
    "lidar_acquisition_years",
    "lidar_acquisition_start_date",
    "lidar_acquisition_end_date",
    "lidar_acquisition_date_precision",
    "lidar_acquisition_date_source",
    "lidar_acquisition_date_notes",
]

# These values come from official national or regional programme pages.  A
# range is intentional when the public catalogue does not assign one flight
# year to every tile in the 5 km AOI.  It must not be silently converted to a
# single year, because that would create false temporal precision.
CURATED = {
    # Belgium: DHMV II and the Walloon 2021-2022 campaign.
    "gent_20146": ("2013-2015", "2013-01-01", "2015-12-31", "official_campaign_range", "https://www.vlaanderen.be/digitaal-hoogtemodel-dhmv", "DHMV II acquisition campaign; exact AOI tile flight dates were not exposed by the inventory query."),
    "bruxelles_brussel_20144": ("2013-2015", "2013-01-01", "2015-12-31", "official_campaign_range", "https://www.vlaanderen.be/digitaal-hoogtemodel-dhmv", "DHMV II acquisition campaign; exact AOI tile flight dates were not exposed by the inventory query."),
    "antwerpen_20142": ("2013-2015", "2013-01-01", "2015-12-31", "official_campaign_range", "https://www.vlaanderen.be/digitaal-hoogtemodel-dhmv", "DHMV II acquisition campaign; exact AOI tile flight dates were not exposed by the inventory query."),
    "liege_20148": ("2021-2022", "2021-01-01", "2022-12-31", "official_campaign_range", "https://geoportail.wallonie.be/catalogue/6029e738-f828-438b-b10a-85e67f77af92.html", "SPW Wallonia LiDAR 2021-2022 campaign."),
    # Spain: second-coverage PNOA flight-lot dates.
    "valencia_22567": ("2015", "2015-10-01", "2015-11-30", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Valencia flight lot."),
    "bilbao_2": ("2017", "2017-08-01", "2017-10-31", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Basque Country flight lot."),
    "valladolid_22568": ("2019", "2019-09-01", "2019-10-31", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Castilla y Leon C flight lot."),
    "palma_19": ("2019-2020", "2019-11-01", "2020-02-29", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Balearic Islands flight lot."),
    "malaga_22550": ("2020", "2020-07-01", "2020-10-31", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Andalucia campaign lot covering Malaga."),
    "sevilla_23": ("2020-2021", "2020-08-01", "2021-04-30", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Andalucia campaign; retained as a range because adjacent lots overlap the AOI region."),
    "zaragoza_205149": ("2020-2021", "2020-10-01", "2021-08-31", "official_flight_lot_range", "https://pnoa.ign.es/pnoa-lidar/segunda-cobertura", "PNOA second-coverage Aragon flight lot."),
    # AHN4 official regional acquisition schedule.
    "amsterdam_21930": ("2020", "2020-01-01", "2020-12-31", "official_regional_campaign_year", "https://www.ahn.nl/ahn-4", "AHN4 Schiphol/TMA and western Netherlands acquisition schedule."),
    "s_gravenhage_the_hague_205146": ("2020", "2020-01-01", "2020-12-31", "official_regional_campaign_year", "https://www.ahn.nl/ahn-4", "AHN4 southwest Netherlands acquisition schedule."),
    "eindhoven_21935": ("2021", "2021-01-01", "2021-12-31", "official_regional_campaign_year", "https://www.ahn.nl/ahn-4", "AHN4 southern Netherlands acquisition schedule."),
    "kbenhavn_copenhagen_20894": ("2014-2015", "2014-10-01", "2015-12-21", "official_national_campaign_range", "https://dataforsyningen.dk/data/930", "Danish national elevation-model acquisition campaign."),
    "tallinn_20932": ("2018", "2018-03-01", "2018-05-31", "official_city_documentation", "https://www.tallinn.ee/en/services/city-model", "Tallinn documents the source LiDAR as collected in spring 2018."),
    "vilnius_21789": ("2019", "2019-01-01", "2019-12-31", "official_dataset_year", "https://www.geoportal.lt/metadata-catalog/catalog/search/resource/details.page?uuid=%7BE5227211-3427-4FD9-B7DD-59221F6A090B%7D", "Official Lidar_DR_LT Vilnius metadata identifies the 2019 dataset."),
    # Poland: current inventory establishes access but not per-tile flight date.
    "bydgoszcz_22129": ("2018-2025", "2018-01-01", "2025-12-31", "official_catalog_series_range", "https://www.geoportal.gov.pl/pl/dane/dane-pomiarowe-lidar-lidar/", "PL-EVRF2007 national catalogue series; query did not expose one AOI-specific acquisition year."),
    "krakow_cracow_22132": ("2018-2025", "2018-01-01", "2025-12-31", "official_catalog_series_range", "https://www.geoportal.gov.pl/pl/dane/dane-pomiarowe-lidar-lidar/", "PL-EVRF2007 national catalogue series; query did not expose one AOI-specific acquisition year."),
    "lublin_22146": ("2018-2025", "2018-01-01", "2025-12-31", "official_catalog_series_range", "https://www.geoportal.gov.pl/pl/dane/dane-pomiarowe-lidar-lidar/", "PL-EVRF2007 national catalogue series; query did not expose one AOI-specific acquisition year."),
    # Welsh Government coverage is a multi-year programme.
    "cardiff_22843": ("2020-2023", "2020-01-01", "2023-12-31", "official_campaign_range", "https://datamap.gov.wales/layers/geonode:welsh_government_lidar_coverage", "Welsh Government national LiDAR programme; exact tile flight date was not exposed by the availability inventory."),
    "newport_22865": ("2020-2023", "2020-01-01", "2023-12-31", "official_campaign_range", "https://datamap.gov.wales/layers/geonode:welsh_government_lidar_coverage", "Welsh Government national LiDAR programme; exact tile flight date was not exposed by the availability inventory."),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required CSV and fail loudly if it is missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Required input is empty: {path}")
    return rows


def get_json(url: str, parameters: dict[str, str], attempts: int = 3) -> dict:
    """Request small JSON metadata with bounded retries and a clear failure."""
    request_url = url + "?" + urllib.parse.urlencode(parameters)
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(request_url, headers={"User-Agent": "building-height-prediction/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - report the final network error
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Metadata request failed after {attempts} attempts: {request_url}: {last_error}")


def bbox_from_aoi(relative_path: str) -> tuple[float, float, float, float]:
    """Calculate a WGS84 bounding box from the city AOI GeoJSON."""
    path = REPOSITORY_ROOT / relative_path.replace("\\", "/")
    data = json.loads(path.read_text(encoding="utf-8"))
    numbers = []

    def visit(value: object) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            numbers.append((float(value[0]), float(value[1])))
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data["features"][0]["geometry"]["coordinates"])
    if not numbers:
        raise RuntimeError(f"AOI contains no coordinates: {path}")
    xs, ys = zip(*numbers)
    return min(xs), min(ys), max(xs), max(ys)


def iso_date(epoch_ms: object) -> str:
    """Convert ArcGIS epoch milliseconds into a UTC ISO calendar date."""
    if epoch_ms in (None, ""):
        return ""
    return datetime.fromtimestamp(float(epoch_ms) / 1000, tz=timezone.utc).date().isoformat()


def record_from_dates(dates: list[tuple[str, str]], precision: str, source: str, notes: str) -> tuple[str, str, str, str, str, str]:
    """Summarize one or more official collection intervals without hiding multiplicity."""
    dates = sorted(set((start, end) for start, end in dates if start or end))
    starts = [start for start, _ in dates if start]
    ends = [end for _, end in dates if end]
    years = sorted({date[:4] for pair in dates for date in pair if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)})
    if not years:
        raise RuntimeError(f"No usable acquisition year returned by {source}")
    year_text = years[0] if len(years) == 1 else ";".join(years)
    return year_text, min(starts or ends), max(ends or starts), precision, source, notes


def usgs_records(rows: list[dict[str, str]]) -> dict[str, tuple[str, str, str, str, str, str]]:
    """Query exact USGS 3DEP collection dates for each selected work unit."""
    tile_rows = read_csv(USGS_TILES)
    directories = defaultdict(set)
    for tile in tile_rows:
        directories[tile["city_slug"]].add(tile["project_directory"])
    output = {}
    for row in rows:
        if row["country"] != "United States of America":
            continue
        slug = row["city_slug"]
        wanted = directories.get(slug, set())
        if not wanted:
            raise RuntimeError(f"No USGS tile project directory recorded for {slug}")
        xmin, ymin, xmax, ymax = bbox_from_aoi(row["aoi_path"])
        payload = get_json(USGS_INDEX, {
            "f": "json", "where": "1=1", "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "false",
            "outFields": "project,collect_start,collect_end,lpc_link",
        })
        dates, projects = [], set()
        for feature in payload.get("features", []):
            attributes = feature.get("attributes", {})
            match = re.search(r"/Projects/([^/]+)", str(attributes.get("lpc_link", "")))
            if match and match.group(1) in wanted:
                dates.append((iso_date(attributes.get("collect_start")), iso_date(attributes.get("collect_end"))))
                projects.add(match.group(1))
        if not dates:
            raise RuntimeError(f"USGS returned no matching collection dates for {slug}: {sorted(wanted)}")
        output[slug] = record_from_dates(
            dates, "exact_official_work_unit_dates", USGS_INDEX,
            "USGS 3DEP collection interval(s) for AOI-intersecting project(s): " + "; ".join(sorted(projects)),
        )
    return output


def canada_records(rows: list[dict[str, str]]) -> dict[str, tuple[str, str, str, str, str, str]]:
    """Extract survey years embedded in official CanElevation project IDs."""
    output = {}
    for row in rows:
        if row["country"] != "Canada":
            continue
        xmin, ymin, xmax, ymax = bbox_from_aoi(row["aoi_path"])
        payload = get_json(CANADA_INDEX, {
            "f": "json", "where": "1=1", "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "false",
            "outFields": "project,tile_name,url", "resultRecordCount": "2000",
        })
        projects = sorted({str(f.get("attributes", {}).get("project", "")) for f in payload.get("features", [])})
        years = sorted({year for project in projects for year in re.findall(r"(?:19|20)\d{2}", project)})
        if not years:
            raise RuntimeError(f"CanElevation returned no project year for {row['city_slug']}")
        text = years[0] if len(years) == 1 else ";".join(years)
        output[row["city_slug"]] = (
            text, years[0] + "-01-01", years[-1] + "-12-31", "exact_official_project_year",
            CANADA_INDEX, "Year(s) parsed from official AOI-intersecting CanElevation project identifiers: " + "; ".join(projects),
        )
    return output


def england_records(rows: list[dict[str, str]]) -> dict[str, tuple[str, str, str, str, str, str]]:
    """Read flight dates from the Environment Agency point-cloud index."""
    output = {}
    excluded = {"cardiff_22843", "newport_22865"}
    for row in rows:
        if row["country"] != "United Kingdom" or row["city_slug"] in excluded:
            continue
        bbox = bbox_from_aoi(row["aoi_path"])
        payload = get_json(ENGLAND_WFS, {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": ENGLAND_LAYER, "outputFormat": "application/json",
            "srsName": "EPSG:4326", "bbox": ",".join(map(str, bbox)) + ",EPSG:4326",
            "count": "10000",
        })
        features = payload.get("features", [])
        latest = [f for f in features if str(f.get("properties", {}).get("latest", "")).lower() == "yes"]
        selected = latest or features
        dates = []
        for feature in selected:
            properties = feature.get("properties", {})
            start, end = str(properties.get("sd_flown", ""))[:10], str(properties.get("ed_flown", ""))[:10]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
                year = str(properties.get("year", ""))
                start = year + "-01-01" if re.fullmatch(r"\d{4}", year) else ""
                end = year + "-12-31" if start else ""
            dates.append((start, end))
        if not any(start or end for start, end in dates):
            raise RuntimeError(f"Environment Agency returned no flight date for {row['city_slug']}")
        output[row["city_slug"]] = record_from_dates(
            dates, "exact_official_tile_flight_dates", ENGLAND_WFS,
            f"Environment Agency point-cloud index; {'latest-flagged' if latest else 'all'} AOI-bbox records ({len(selected)} tiles).",
        )
    return output


def main() -> None:
    """Enrich all 94 rows, validate completeness, and replace the CSV atomically."""
    rows = read_csv(TRAINING_CSV)
    if len(rows) != 94:
        raise RuntimeError(f"Expected exactly 94 training LiDAR cities, found {len(rows)}")
    records = dict(CURATED)
    records.update(usgs_records(rows))
    records.update(canada_records(rows))
    records.update(england_records(rows))

    # Official metadata checks completed for these small national catalogues.
    records.update({
        "dublin_21542": ("2010-2011", "2010-05-01", "2011-05-31", "exact_official_project_range", "https://gsi.geodata.gov.ie/server/rest/services/Lidar/IE_GSI_LiDAR_Coverage_TII_IE26_ITM/FeatureServer/0", "Official TII coverage attribute DATECAPTUR: May 2010 - May 2011."),
        "zurich_zurich_22606": ("2024", "2024-01-01", "2024-12-31", "exact_official_tile_year", "https://api3.geo.admin.ch/rest/services/all/MapServer/identify", "swissSURFACE3D metadata tiles intersecting the city identify query report GPS time year 2024."),
        "basel_22600": ("2024", "2024-01-01", "2024-12-31", "exact_official_tile_year", "https://api3.geo.admin.ch/rest/services/all/MapServer/identify", "swissSURFACE3D metadata tiles intersecting the city identify query report GPS time year 2024."),
    })

    missing = sorted({row["city_slug"] for row in rows} - set(records))
    extra = sorted(set(records) - {row["city_slug"] for row in rows})
    if missing or extra:
        raise RuntimeError(f"Acquisition metadata key mismatch. Missing={missing}; extra={extra}")

    for row in rows:
        values = records[row["city_slug"]]
        row.update(dict(zip(NEW_FIELDS, values)))
        # Replace the old broad placeholder with the new city-specific value.
        row["lidar_acquisition_year"] = row["lidar_acquisition_years"]

    fields = list(rows[0])
    for field in NEW_FIELDS:
        if field not in fields:
            fields.append(field)
    temporary = TRAINING_CSV.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(TRAINING_CSV)

    precision_counts = defaultdict(int)
    for row in rows:
        precision_counts[row["lidar_acquisition_date_precision"]] += 1
    print(f"Updated {len(rows)} cities: {TRAINING_CSV.relative_to(REPOSITORY_ROOT)}")
    for precision, count in sorted(precision_counts.items()):
        print(f"  {precision}: {count}")
    print("No LiDAR or imagery data were downloaded; only small official metadata responses were read.")


if __name__ == "__main__":
    main()
