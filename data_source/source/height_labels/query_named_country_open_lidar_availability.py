#!/usr/bin/env python3
"""Audit open LiDAR availability for the named 17-country city pool.

No LiDAR, DSM, or DTM payload is downloaded. The script uses existing USGS
results, official national-coverage documentation, and machine-readable
ArcGIS coverage catalogs where available. It preserves the difference between
direct file verification, portal reachability, registration-gated access, and
manual interactive-portal checks.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[2]
VENV_DIRECTORY = SCRIPT_DIRECTORY / "venv_height_labels"
VENV_PYTHON = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Use the task environment that already contains the spatial packages."""
    if os.environ.get(VENV_MARKER) == "1":
        return
    if Path(sys.prefix).resolve() == VENV_DIRECTORY.resolve():
        os.environ[VENV_MARKER] = "1"
        return
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing height-label Python environment: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, shape
from shapely.ops import transform, unary_union


HTTP_USER_AGENT = "building-height-prediction/1.0 global-open-lidar-metadata-audit"
HTTP_ATTEMPTS = 4
URL_CHECK_CACHE: dict[tuple[str, bool], tuple[str, str]] = {}
CITY_FILE = REPOSITORY_ROOT / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv"
AOI_DIRECTORY = REPOSITORY_ROOT / "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city"
USGS_FILE = REPOSITORY_ROOT / "data_source/data/height_labels/generated/usgs_3dep_global_city_availability.csv"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data_source/data/height_labels/generated"
OUTPUT_FILE = OUTPUT_DIRECTORY / "named_country_open_lidar_city_audit.csv"
SOURCE_FILE = OUTPUT_DIRECTORY / "named_country_open_lidar_source_registry.csv"
LOG_DIRECTORY = OUTPUT_DIRECTORY / "logs"

NAMED_COUNTRIES = {
    "United States of America", "Spain", "Netherlands", "Finland",
    "Denmark", "Norway", "Sweden", "Switzerland", "Belgium", "Poland",
    "United Kingdom", "Ireland", "Estonia", "Latvia", "Lithuania",
    "Australia", "Canada",
}


def source(
    program: str,
    url: str,
    product: str,
    file_format: str,
    license_name: str,
    acquisition: str,
    coverage_basis: str,
    method: str = "documented_national_coverage",
    access_constraint: str = "anonymous portal/file access",
) -> dict[str, str]:
    """Create one consistently structured source-registry record."""
    return {
        "source_program": program,
        "official_access_link": url,
        "product_type": product,
        "format": file_format,
        "license": license_name,
        "acquisition_year": acquisition,
        "coverage_basis": coverage_basis,
        "adapter_method": method,
        "access_constraint": access_constraint,
    }


SOURCES = {
    "Spain": source(
        "PNOA-LiDAR", "https://centrodedescargas.cnig.es/CentroDescargas/home",
        "classified point cloud; DSM; DTM", "LAZ; raster", "CC BY 4.0",
        "national cycles 2008-2015, 2015-2021, 2022-2025",
        "Official CNIG catalog describes national first and second coverages",
    ),
    "Netherlands": source(
        "Actueel Hoogtebestand Nederland (AHN)", "https://www.ahn.nl/open-data",
        "classified point cloud; DSM; DTM", "LAZ; GeoTIFF", "Open data; AHN6 CC BY 4.0",
        "AHN2-AHN6 national cycles", "Official AHN documentation states nationwide coverage",
    ),
    "Finland": source(
        "National Land Survey laser-scanning data",
        "https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data",
        "classified point cloud; DTM", "LAZ; GeoTIFF", "NLS open-data licence",
        "2008-present", "Official NLS description states availability for all Finland",
    ),
    "Denmark": source(
        "Danmarks Højdemodel (DHM/Punktsky)", "https://dataforsyningen.dk/data/930",
        "classified point cloud; DSM; DTM", "LAZ; GeoTIFF", "Danish public-geodata terms",
        "2014-present updates", "Official product specification describes a nationwide point cloud",
        access_constraint="free account/API token may be required for machine access",
    ),
    "Norway": source(
        "Nasjonal detaljert høydemodell / Høydedata", "https://hoydedata.no/LaserInnsyn2/",
        "point cloud; DSM; DTM", "LAZ/ZLAS; GeoTIFF", "Kartverket open-data terms where marked free",
        "2010-present", "National detailed elevation project completed; Oslo falls in laser-scanned coverage",
        access_constraint="some non-NDH projects can be partner-only; selected project must be checked",
    ),
    "Sweden": source(
        "Lantmäteriet Laserdata Nedladdning, NH/skog",
        "https://www2.lantmateriet.se/en/geodata/our-products/product-list/laser-data-download-forest/",
        "classified point cloud", "LAZ/COPC", "CC0 for forest product",
        "NH 2009-2019; forest 2018-present", "National NH completed; newer forest coverage is ongoing",
        access_constraint="STAC authorization and acceptance of use terms required from June 2026",
    ),
    "Switzerland": source(
        "swissSURFACE3D", "https://www.swisstopo.admin.ch/en/height-model-swisssurface3d",
        "classified point cloud", "COPC/LAZ; zipped LAS", "swisstopo Open Government Data terms",
        "2017-present", "Official swisstopo documentation states nationwide availability since 2024",
    ),
    "Belgium-Flanders": source(
        "Digitaal Hoogtemodel Vlaanderen II", "https://remotesensing.vlaanderen.be/apps/openlidar/",
        "raw point cloud; DSM; DTM", "LAZ; raster", "Flemish open-data terms",
        "2013-2015", "Official dataset status complete for Flanders and Brussels",
    ),
    "Belgium-Wallonia": source(
        "SPW LiDAR 2021-2022", "https://geoportail.wallonie.be/catalogue/218f0248-a265-4cac-878c-43029d6ada8a.html",
        "point cloud; DSM; DTM", "LAS/LAZ; raster", "CC BY 4.0",
        "2021-2022", "Official 500 m tile catalog covers the Walloon Region",
    ),
    "Poland": source(
        "GUGiK national LiDAR point cloud", "https://www.geoportal.gov.pl/pl/dane/dane-pomiarowe-lidar-lidar/",
        "classified ALS point cloud; DSM; DTM", "LAS/LAZ", "Polish public open-data terms",
        "multiple ISOK/national cycles", "Official GUGiK download service provides national LiDAR sheets",
    ),
    "United Kingdom-England": source(
        "Environment Agency National LiDAR Programme", "https://environment.data.gov.uk/survey",
        "classified point cloud; DSM; DTM", "LAZ; GeoTIFF", "Open Government Licence",
        "2017-2023 and updates", "Official program states 1 m coverage for all England",
    ),
    "United Kingdom-Wales": source(
        "Welsh Government LiDAR 2020-2023", "https://datamap.gov.wales/maps/lidar-viewer/",
        "LiDAR-derived DSM and DTM", "Cloud Optimized GeoTIFF", "Open Government Licence",
        "2020-2023", "Official Welsh catalog provides nationwide DSM/DTM tile links",
    ),
    "United Kingdom-Scotland": source(
        "Scottish Remote Sensing Portal", "https://remotesensingdata.gov.scot/",
        "point cloud and derived elevation products", "portal tiles and WMS", "OGL or dataset-specific",
        "multiple partial surveys; national program 2025-2027", "Official portal is partial while national capture is underway",
        method="interactive_partial_catalog", access_constraint="coverage requires portal/WFS inspection",
    ),
    "United Kingdom-Northern Ireland": source(
        "Open Data NI LiDAR collections", "https://admin.opendatani.gov.uk/dataset/historic-environment-division-lidar",
        "mostly LiDAR-derived DSM/DTM", "ASCII/raster/ArcGIS services", "Open Government Licence",
        "2008-present partial surveys", "Official collections are coastal or site-specific rather than national",
        method="interactive_partial_catalog",
    ),
    "Ireland": source(
        "Geological Survey Ireland Open Topographic LiDAR", "https://gsi.geodata.gov.ie/server/rest/services/Lidar",
        "LiDAR-derived DSM and DTM", "GeoTIFF", "CC BY 4.0",
        "multiple surveys", "Official ArcGIS coverage services include download links",
        method="arcgis_coverage_union",
    ),
    "Estonia": source(
        "Estonian Land and Spatial Development Board elevation data",
        "https://geoportaal.maaamet.ee/eng/spatial-data/elevation-data/download-elevation-data-p664.html",
        "classified point cloud; DSM; DTM", "LAZ and raster", "Estonian open-data terms",
        "national repeated cycles", "Official page states all Estonia is covered by LiDAR elevation data",
    ),
    "Latvia": source(
        "Latvian Geospatial Information Agency digital height model basic data",
        "https://www.lgia.gov.lv/en/digital-height-models-0", "classified point cloud; DSM; DTM",
        "LAS; TIFF; ASCII", "CC BY 4.0", "2013-2019 and updates",
        "Official agency describes continuous national airborne laser scanning and open basic data",
    ),
    "Lithuania": source(
        "Lidar_DR_LT", "https://www.geoportal.lt/geoportal/subscribe/-/asset_publisher/I0YH9ZsWns4x/content/lidar_dr_lt-lazerinio-skenavimo-tasku-duomenu-parsisiuntimas-ir-perziura",
        "classified point cloud", "LAZ", "Lithuanian spatial-data portal terms",
        "2019-2025", "Official program states full national collection completed by 2023 with updates",
        access_constraint="interactive order limits the number of sheets per request",
    ),
    "Australia": source(
        "ELVIS Elevation and Depth", "https://elevation.fsdf.org.au/",
        "point cloud and LiDAR-derived DEM where available", "LAS/LAZ and raster", "dataset-specific Creative Commons",
        "multiple surveys", "National discovery portal contains partial survey coverage",
        method="interactive_partial_catalog", access_constraint="public portal ordering; no stable documented query API found",
    ),
    "Canada": source(
        "LiDAR Point Clouds - CanElevation Series",
        "https://open.canada.ca/data/en/dataset/7069387e-9986-4297-9f55-0288e9676947",
        "classified point cloud", "COPC/LAZ", "Open Government Licence - Canada",
        "2014-present", "Official NRCan ArcGIS tile index with direct cloud URLs",
        method="arcgis_tile_union",
    ),
}


AUDIT_FIELDS = [
    "wup_urbancode", "city_slug", "city_name", "country", "centroid_latitude",
    "centroid_longitude", "coverage_status", "aoi_coverage_percent",
    "source_program", "official_access_link", "product_type", "format", "license",
    "acquisition_year", "download_file_count", "download_endpoint_status",
    "representative_download_url", "representative_http_status", "access_constraint",
    "verification_date", "verification_method", "automation_level", "notes",
]


def request_json(url: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request JSON with retries; metadata requests only."""
    request_url = url
    if parameters:
        request_url = f"{url}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": HTTP_USER_AGENT})
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
            if attempt == HTTP_ATTEMPTS:
                raise
            time.sleep(4 * attempt)
    raise RuntimeError("Unreachable retry state")


def check_url(url: str, byte_range: bool = False) -> tuple[str, str]:
    """Check a portal or one byte of a file without retaining response data."""
    cache_key = (url, byte_range)
    if cache_key in URL_CHECK_CACHE:
        return URL_CHECK_CACHE[cache_key]
    headers = {"User-Agent": HTTP_USER_AGENT}
    if byte_range:
        headers["Range"] = "bytes=0-0"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response.read(1)
            code = str(response.status)
        result = ("reachable" if int(code) in (200, 206) else "unexpected_status", code)
    except urllib.error.HTTPError as error:
        result = ("http_error", str(error.code))
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        result = ("connection_error", str(error))
    URL_CHECK_CACHE[cache_key] = result
    return result


def load_aoi(city_slug: str):
    """Read one canonical WUP disk geometry."""
    path = AOI_DIRECTORY / f"{city_slug}_5km.geojson"
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features") or []
    if len(features) != 1:
        raise ValueError(f"Expected one feature in {path}")
    geometry = shape(features[0]["geometry"])
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid AOI: {path}")
    return geometry


def local_area_tools(geometry):
    """Project an AOI to local UTM for defensible area percentages."""
    centroid = geometry.centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    transformer = Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True)
    return transform(transformer.transform, geometry), transformer


def esri_geometry_to_shape(geometry: dict[str, Any]):
    """Convert ArcGIS polygon rings to Shapely geometry."""
    polygons = []
    for ring in geometry.get("rings") or []:
        if len(ring) >= 4:
            polygon = Polygon(ring)
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    return unary_union(polygons) if polygons else None


def query_arcgis_layer(aoi, layer_url: str) -> list[dict[str, Any]]:
    """Query polygon records intersecting a WGS84 AOI envelope."""
    min_x, min_y, max_x, max_y = aoi.bounds
    payload = request_json(
        f"{layer_url}/query",
        {
            "f": "json", "where": "1=1", "geometry": f"{min_x},{min_y},{max_x},{max_y}",
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
            "returnGeometry": "true", "outSR": 4326, "resultRecordCount": 50000,
        },
    )
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("features") or []


def coverage_from_features(aoi, features: list[dict[str, Any]]) -> float:
    """Calculate union coverage so overlapping surveys are not double-counted."""
    projected_aoi, transformer = local_area_tools(aoi)
    footprints = []
    for feature in features:
        geometry = esri_geometry_to_shape(feature.get("geometry") or {})
        if geometry is not None:
            intersection = geometry.intersection(aoi)
            if not intersection.is_empty:
                footprints.append(transform(transformer.transform, intersection))
    if not footprints:
        return 0.0
    covered = unary_union(footprints).intersection(projected_aoi)
    return min(100.0, 100.0 * covered.area / projected_aoi.area)


def query_canada(city: dict[str, str], aoi) -> dict[str, Any]:
    """Query NRCan's official tile index and verify a direct COPC/LAZ URL."""
    layer = "https://maps-cartes.services.geo.ca/server_serveur/rest/services/NRCan/lidar_point_cloud_canelevation_en/MapServer/1"
    features = query_arcgis_layer(aoi, layer)
    coverage = coverage_from_features(aoi, features)
    urls = sorted({str(f.get("attributes", {}).get("url", "")) for f in features if f.get("attributes", {}).get("url")})
    endpoint_status, http_status = check_url(urls[0], byte_range=True) if urls else ("missing_url", "")
    status = "not_found" if not features else ("ready_for_download" if coverage >= 99 else "incomplete")
    return {
        "coverage_status": status, "aoi_coverage_percent": coverage,
        "download_file_count": len(urls), "download_endpoint_status": endpoint_status,
        "representative_download_url": urls[0] if urls else "",
        "representative_http_status": http_status,
        "verification_method": "NRCan ArcGIS tile union plus one-byte representative COPC/LAZ request",
        "automation_level": "file-level API",
    }


IRELAND_SERVICES = [
    "IE_GSI_LiDAR_Coverage_GSI_DCHG_DP_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_GSI_Phase2_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_NYU_Dublin_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_OPW_Cork_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_OPW_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_OPW_NASC_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_TII_IE26_ITM",
    "IE_GSI_LiDAR_Coverage_WH_CoCo_IE26_ITM",
]


def query_ireland(city: dict[str, str], aoi) -> dict[str, Any]:
    """Union all official GSI LiDAR coverage services for Dublin."""
    features = []
    urls = set()
    queried_layers = 0
    for service in IRELAND_SERVICES:
        base = f"https://gsi.geodata.gov.ie/server/rest/services/Lidar/{service}/FeatureServer"
        metadata = request_json(base, {"f": "json"})
        for layer in metadata.get("layers") or []:
            layer_features = query_arcgis_layer(aoi, f"{base}/{layer['id']}")
            queried_layers += 1
            features.extend(layer_features)
            for feature in layer_features:
                for value in (feature.get("attributes") or {}).values():
                    if isinstance(value, str) and value.startswith("http"):
                        urls.add(value)
    coverage = coverage_from_features(aoi, features)
    usable_urls = sorted(urls)
    endpoint_status, http_status = check_url(usable_urls[0], byte_range=True) if usable_urls else ("catalog_only", "")
    status = "not_found" if not features else ("ready_for_download" if coverage >= 99 else "incomplete")
    return {
        "coverage_status": status, "aoi_coverage_percent": coverage,
        "download_file_count": len(usable_urls), "download_endpoint_status": endpoint_status,
        "representative_download_url": usable_urls[0] if usable_urls else "",
        "representative_http_status": http_status,
        "verification_method": f"Union of {queried_layers} layers from official GSI ArcGIS coverage services",
        "automation_level": "coverage API; raster links where exposed",
    }


SCOTLAND_CITIES = {"Glasgow", "Edinburgh"}
WALES_CITIES = {"Cardiff", "Newport", "Swansea"}
NORTHERN_IRELAND_CITIES = {"Belfast"}
FLANDERS_CITIES = {"Antwerpen", "Gent"}
WALLONIA_CITIES = {"Liège", "Charleroi"}


def source_key_for_city(city: dict[str, str]) -> str:
    """Route federal/devolved countries to the correct official source."""
    if city["country"] == "Belgium":
        if city["city_name"] in WALLONIA_CITIES:
            return "Belgium-Wallonia"
        return "Belgium-Flanders"  # DHMV II metadata includes Brussels.
    if city["country"] == "United Kingdom":
        if city["city_name"] in SCOTLAND_CITIES:
            return "United Kingdom-Scotland"
        if city["city_name"] in WALES_CITIES:
            return "United Kingdom-Wales"
        if city["city_name"] in NORTHERN_IRELAND_CITIES:
            return "United Kingdom-Northern Ireland"
        return "United Kingdom-England"
    return city["country"]


def documented_result(record: dict[str, str]) -> dict[str, Any]:
    """Return a conservative result for an official documented-coverage source."""
    method = record["adapter_method"]
    endpoint_status, http_status = check_url(record["official_access_link"])
    if method == "interactive_partial_catalog":
        status = "manual_portal_check_required"
        coverage = ""
    elif "authorization" in record["access_constraint"]:
        status = "registration_required"
        coverage = "100"
    else:
        status = "ready_for_download"
        coverage = "100"
    return {
        "coverage_status": status, "aoi_coverage_percent": coverage,
        "download_file_count": "", "download_endpoint_status": endpoint_status,
        "representative_download_url": "", "representative_http_status": http_status,
        "verification_method": record["coverage_basis"] + "; official access portal reachability check",
        "automation_level": "documented coverage plus portal check",
    }


def load_cities() -> list[dict[str, str]]:
    """Load and validate the named-country subset."""
    with CITY_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        cities = [row for row in csv.DictReader(handle) if row["country"] in NAMED_COUNTRIES]
    if len(cities) != 249:
        raise RuntimeError(f"Expected 249 named-country cities, found {len(cities)}")
    return cities


def load_usgs() -> dict[str, dict[str, str]]:
    """Load the already completed 144-city USGS metadata audit."""
    if not USGS_FILE.is_file():
        raise FileNotFoundError(f"Missing required USGS audit: {USGS_FILE}")
    with USGS_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 144 or any(row["query_status"] != "success" for row in rows):
        raise RuntimeError("USGS audit must contain 144 successful city records")
    return {row["city_slug"]: row for row in rows}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write atomically so a partial process never resembles a complete audit."""
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Run all available adapters, retaining failures as explicit records."""
    cities = load_cities()
    usgs = load_usgs()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIRECTORY / f"query_named_country_open_lidar_{timestamp}.log"
    verified_at = datetime.now(timezone.utc).date().isoformat()
    rows = []

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started UTC: {datetime.now(timezone.utc).isoformat()}\n")
        log.write("Metadata-only audit; no LiDAR/DSM/DTM payload downloads\n")
        for index, city in enumerate(cities, start=1):
            key = source_key_for_city(city)
            record = SOURCES.get(key)
            print(f"[{index}/{len(cities)}] {city['city_slug']} ({key})", flush=True)
            try:
                aoi = load_aoi(city["city_slug"])
                if city["country"] == "United States of America":
                    old = usgs[city["city_slug"]]
                    result = {
                        "coverage_status": old["coverage_status"],
                        "aoi_coverage_percent": old["aoi_coverage_percent"],
                        "download_file_count": old["download_urls_present"],
                        "download_endpoint_status": old["representative_url_status"],
                        "representative_download_url": old["representative_download_url"],
                        "representative_http_status": old["representative_http_status"],
                        "verification_method": "Existing official USGS LPC tile-union audit",
                        "automation_level": "file-level API",
                    }
                    record = source(
                        "USGS 3D Elevation Program (3DEP)", "https://www.usgs.gov/tools/lidarexplorer",
                        "classified point cloud", "LAZ", "U.S. public domain", "2004-present",
                        "Official National Map LPC API", method="file-level API",
                    )
                elif city["country"] == "Canada":
                    result = query_canada(city, aoi)
                elif city["country"] == "Ireland":
                    result = query_ireland(city, aoi)
                else:
                    if record is None:
                        raise KeyError(f"No source registry record for {key}")
                    result = documented_result(record)
                notes = ""
            except Exception as error:
                if record is None:
                    record = source("", "", "", "", "", "", "No source record", method="none")
                result = {
                    "coverage_status": "query_failed", "aoi_coverage_percent": "",
                    "download_file_count": "", "download_endpoint_status": "not_checked",
                    "representative_download_url": "", "representative_http_status": "",
                    "verification_method": f"{record['adapter_method']} adapter failed",
                    "automation_level": record["adapter_method"],
                }
                notes = f"{type(error).__name__}: {error}"
                print(f"  FAILED: {notes}", flush=True)

            row = {
                "wup_urbancode": city["wup_urbancode"], "city_slug": city["city_slug"],
                "city_name": city["city_name"], "country": city["country"],
                "centroid_latitude": city["latitude"], "centroid_longitude": city["longitude"],
                **result,
                "source_program": record["source_program"],
                "official_access_link": record["official_access_link"],
                "product_type": record["product_type"], "format": record["format"],
                "license": record["license"], "acquisition_year": record["acquisition_year"],
                "access_constraint": record["access_constraint"],
                "verification_date": verified_at, "notes": notes,
            }
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()

    source_fields = ["source_key", *next(iter(SOURCES.values())).keys()]
    source_rows = [{"source_key": key, **value} for key, value in sorted(SOURCES.items())]
    write_csv(OUTPUT_FILE, AUDIT_FIELDS, rows)
    write_csv(SOURCE_FILE, source_fields, source_rows)
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    failures = sum(row["coverage_status"] == "query_failed" for row in rows)
    print(f"Wrote {len(rows)} city records to {OUTPUT_FILE.relative_to(REPOSITORY_ROOT)}")
    print(f"Wrote {len(source_rows)} source records to {SOURCE_FILE.relative_to(REPOSITORY_ROOT)}")
    print(f"Log: {log_path.relative_to(REPOSITORY_ROOT)}")
    if failures:
        raise RuntimeError(f"{failures} country-city checks failed; inspect output and log")


if __name__ == "__main__":
    main()
