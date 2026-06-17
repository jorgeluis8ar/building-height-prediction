"""
Create City Buffers

Environment: data_source/source/city_aois/venv_city_aois

Requires (inputs from earlier stages):
    - README.md
    - data_source/data/city_aois/source/WUP2018-F22-Cities_Over_300K_Annual_V7.xls

Produces (outputs for later stages):
    - data_source/data/city_aois/generated/cities_sample.csv
    - data_source/data/city_aois/generated/city_buffers_5km.geojson
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/<city>_5km.geojson

Description:
    Reads the current city list from README.md, matches those cities to the
    WUP 2018 city-center/CBD coordinates, creates Point geometries, creates 5km buffer
    zones for satellite data extraction and analysis.

Usage:
    python data_source/source/city_aois/create_city_buffers.py

Expected runtime: < 1 minute
"""

import sys
import csv
import json
import math
import os
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv_city_aois"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "CITY_AOIS_VENV_ACTIVE"


def relaunch_inside_venv():
    """
    Relaunch this script with the local virtual environment Python.

    This keeps the command simple for the user:
        python data_source/source/city_aois/create_city_buffers.py

    If that command is run with the system Python, the script immediately
    re-executes itself with `venv_city_aois/bin/python` so the pinned
    dependencies in this folder are used.
    """
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()

    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing city_aois virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/city_aois/venv_city_aois")
        print("  data_source/source/city_aois/venv_city_aois/bin/python -m pip install -r data_source/source/city_aois/requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env)


relaunch_inside_venv()

# Anchor project paths to this script so the current shell folder cannot send
# outputs to a different repository or sibling directory.
PROJECT_ROOT = SCRIPT_DIR.parents[2]
README_FILE = PROJECT_ROOT / "README.md"
WUP2018_CBD_FILE = PROJECT_ROOT / "data_source/data/city_aois/source/WUP2018-F22-Cities_Over_300K_Annual_V7.xls"
CITIES_SAMPLE_FILE = PROJECT_ROOT / "data_source/data/city_aois/generated/cities_sample.csv"
CITY_BUFFERS_FILE = PROJECT_ROOT / "data_source/data/city_aois/generated/city_buffers_5km.geojson"
CITY_BUFFERS_BY_CITY_DIR = PROJECT_ROOT / "data_source/data/city_aois/generated/city_buffers_5km_by_city"
BUFFER_RADIUS_KM = 5
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_PROJECTED = "local_geodesic_5km_buffer"
FIGURES_DIR = PROJECT_ROOT / "data_source/analysis/figures"

CITY_COUNTRIES = {
    "Boston": "United States of America",
    "Chicago": "United States of America",
    "Los Angeles": "United States of America",
    "Montreal": "Canada",
    "New York City": "United States of America",
    "San Francisco": "United States of America",
    "Seattle": "United States of America",
    "Vancouver": "Canada",
    "Bogota": "Colombia",
    "Buenos Aires": "Argentina",
    "Caracas": "Venezuela (Bolivarian Republic of)",
    "Guadalajara": "Mexico",
    "Medellin": "Colombia",
    "Quito": "Ecuador",
    "Santiago de Chile": "Chile",
    "Sao Paulo": "Brazil",
    "Amsterdam": "Netherlands",
    "Barcelona": "Spain",
    "Birmingham": "United Kingdom",
    "Copenhagen": "Denmark",
    "Helsinki": "Finland",
    "London": "United Kingdom",
    "Lyon": "France",
    "Madrid": "Spain",
    "Manchester": "United Kingdom",
    "Marseille": "France",
    "Oslo": "Norway",
    "Paris": "France",
    "Rotterdam": "Netherlands",
    "Utrecht": "Netherlands",
    "Valencia": "Spain",
    "Vienna": "Austria",
    "Zurich": "Switzerland",
    "Cape Town": "South Africa",
    "Hong Kong": "China, Hong Kong SAR",
    "Nairobi": "Kenya",
    "Jakarta": "Indonesia",
    "Singapore": "Singapore",
    "Tokyo": "Japan",
}

CITY_ALIASES = {
    "Boston": ["Boston-Providence"],
    "Bogota": ["Bogotá"],
    "Copenhagen": ["København (Copenhagen)", "Copenhagen"],
    "Hong Kong": ["Hong Kong"],
    "Los Angeles": ["Los Angeles-Long Beach-Santa Ana"],
    "Marseille": ["Marseille-Aix-en-Provence"],
    "Medellin": ["Medellín"],
    "Montreal": ["Montréal"],
    "New York City": ["New York-Newark"],
    "San Francisco": ["San Francisco-Oakland"],
    "Santiago de Chile": ["Santiago"],
    "Sao Paulo": ["São Paulo"],
    "Tokyo": ["Tokyo"],
    "Vienna": ["Wien (Vienna)"],
    "Zurich": ["Zürich (Zurich)", "Zürich"],
}


def normalize_text(value):
    """
    Normalize city and country names for matching.

    The WUP file includes accents and parenthetical names. This helper removes
    accents, drops punctuation, and lowercases text so names like Bogota and
    Bogotá can match safely.
    """
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_current_cities_from_readme(readme_file):
    """
    Read the current project city list from the README.md table.

    Returns
    -------
    list
        City names in the same order they appear in README.md.
    """
    print(f"Reading current cities from: {readme_file}")

    text = readme_file.read_text(encoding="utf-8")
    current_section = text.split("### Current Cities", 1)[1].split("### Principal Data Components", 1)[0]

    cities = []
    for line in current_section.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or "Region" in line:
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 2:
            continue

        city_names = [city.strip() for city in parts[1].split(",")]
        cities.extend(city for city in city_names if city)

    if not cities:
        raise ValueError("No cities were found in the README.md Current Cities table")

    print(f"Loaded {len(cities)} README current cities")
    return cities


def load_wup2018_city_centers(wup_file):
    """
    Read WUP 2018 city-center/CBD coordinates from the Excel workbook.

    Returns
    -------
    dict
        Lookup keyed by normalized country and urban agglomeration name.
    """
    print(f"Reading WUP 2018 city centers from: {wup_file}")

    try:
        import xlrd
    except ImportError:
        print("ERROR: Missing required Python package: xlrd")
        print("Install xlrd before running this script because WUP 2018 is an old .xls workbook.")
        sys.exit(1)

    workbook = xlrd.open_workbook(wup_file)
    sheet = workbook.sheet_by_name("Data")

    headers = [sheet.cell_value(0, col) for col in range(sheet.ncols)]
    columns = {name: index for index, name in enumerate(headers)}
    required_columns = ["Country or area", "Urban Agglomeration", "Latitude", "Longitude"]
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(f"WUP 2018 workbook is missing columns: {missing_columns}")

    lookup = {}
    for row_number in range(1, sheet.nrows):
        country = sheet.cell_value(row_number, columns["Country or area"])
        city = sheet.cell_value(row_number, columns["Urban Agglomeration"])
        lat = sheet.cell_value(row_number, columns["Latitude"])
        lon = sheet.cell_value(row_number, columns["Longitude"])

        if not country or not city:
            continue

        lookup[(normalize_text(country), normalize_text(city))] = {
            "country": country,
            "city": city,
            "lat": float(lat),
            "lon": float(lon),
        }

    print(f"Loaded {len(lookup)} WUP 2018 city-center records")
    return lookup


def write_cities_sample_from_wup2018(readme_file, wup_file, output_file):
    """
    Build cities_sample.csv from README.md cities and WUP 2018 coordinates.

    This step makes the coordinate source explicit and reproducible before
    the buffers are created.
    """
    readme_cities = read_current_cities_from_readme(readme_file)
    wup_lookup = load_wup2018_city_centers(wup_file)

    rows = []
    unmatched = []
    for city in readme_cities:
        country = CITY_COUNTRIES.get(city)
        if not country:
            unmatched.append(city)
            continue

        city_candidates = [city] + CITY_ALIASES.get(city, [])
        matched_record = None
        for city_candidate in city_candidates:
            matched_record = wup_lookup.get((normalize_text(country), normalize_text(city_candidate)))
            if matched_record:
                break

        if matched_record:
            rows.append({
                "city_id": len(rows) + 1,
                "city_name": city,
                "lat": f"{matched_record['lat']:.4f}",
                "lon": f"{matched_record['lon']:.4f}",
            })
        else:
            unmatched.append(city)

    if unmatched:
        raise ValueError(f"These README cities did not match WUP 2018: {unmatched}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["city_id", "city_name", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} WUP 2018 city centers to: {output_file}")


def check_required_inputs():
    """
    Check that all required input files exist before starting.
    Fail loudly if any are missing (CRITICAL RULE 2).
    """
    required_files = [
        README_FILE,
        WUP2018_CBD_FILE,
        CITIES_SAMPLE_FILE,
    ]

    missing = [f for f in required_files if not f.exists()]

    if missing:
        print("ERROR: Missing required input files:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Please confirm README.md and the WUP 2018 source file exist.")
        print("The script creates data_source/data/city_aois/generated/cities_sample.csv")
        print("with columns: city_id, city_name, lat, lon.")
        sys.exit(1)


def create_city_centroids(city_list_file):
    """
    Read city list CSV and create GeoDataFrame with Point geometries.

    Parameters
    ----------
    city_list_file : Path
        Path to cities_sample.csv file

    Returns
    -------
    list
        List of city records with point coordinates in WGS84 (EPSG:4326)
    """
    print(f"Reading city list from: {city_list_file}")

    # Read CSV using Python's built-in csv module so the script can run even
    # before the full geospatial Python environment is installed.
    with city_list_file.open(newline="", encoding="utf-8-sig") as file:
        df = list(csv.DictReader(file))

    required_columns = {"city_id", "city_name", "lat", "lon"}
    missing_columns = required_columns.difference(df[0].keys() if df else [])
    if missing_columns:
        raise ValueError(f"Missing required columns in city list: {sorted(missing_columns)}")

    print(f"Loaded {len(df)} cities:")
    if df and "treated" in df[0]:
        treated_count = sum(int(row["treated"]) for row in df)
        print(f"  - Treated cities: {treated_count}")
        print(f"  - Control cities: {len(df) - treated_count}")
    else:
        print("  - Treatment status column not found; skipping treated/control summary")

    # Convert coordinate text from the CSV into numeric longitude/latitude.
    # GeoJSON stores coordinates as [longitude, latitude].
    for row in df:
        row["lat"] = float(row["lat"])
        row["lon"] = float(row["lon"])
        row["geometry"] = {
            "type": "Point",
            "coordinates": [row["lon"], row["lat"]],
        }

    print(f"Created GeoDataFrame with {len(df)} city points")

    return df


def create_city_buffers(centroids_gdf, radius_km=5):
    """
    Create equal-area buffers around city centroids.

    Parameters
    ----------
    centroids_gdf : list
        List of city records with point coordinates in geographic CRS
    radius_km : float
        Buffer radius in kilometers (default: 5)

    Returns
    -------
    list
        List of city records with buffer polygons
    """
    print(f"\nCreating {radius_km}km buffers around city centers...")

    # The original project template used a projected CRS and GeoPandas.
    # Here we create each 5km circle directly from the WGS84 centroid using
    # spherical destination-point math, which keeps the same GeoJSON output
    # without requiring unavailable geospatial packages.
    print(f"  - Projecting to {CRS_PROJECTED}")

    # Create buffers (radius in meters)
    radius_m = radius_km * 1000
    print(f"  - Creating buffers with radius = {radius_m}m")
    earth_radius_m = 6_371_008.8

    def destination_point(lon, lat, bearing_degrees, distance_m):
        """Return lon/lat reached by moving distance_m from lon/lat."""
        bearing = math.radians(bearing_degrees)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        angular_distance = distance_m / earth_radius_m

        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
        )

        return [math.degrees(lon2), math.degrees(lat2)]

    gdf_proj = []
    for row in centroids_gdf:
        polygon_ring = [
            destination_point(row["lon"], row["lat"], bearing, radius_m)
            for bearing in range(0, 360, 5)
        ]
        polygon_ring.append(polygon_ring[0])

        buffered_row = dict(row)
        buffered_row["geometry"] = {
            "type": "Polygon",
            "coordinates": [polygon_ring],
        }
        buffered_row["area_km2"] = math.pi * (radius_km ** 2)
        gdf_proj.append(buffered_row)

    # Calculate buffer area for verification
    expected_area = math.pi * (radius_km ** 2)
    print(f"  - Expected buffer area: ~{expected_area:.2f} km²")
    mean_area = sum(row["area_km2"] for row in gdf_proj) / len(gdf_proj)
    print(f"  - Actual mean buffer area: {mean_area:.2f} km²")

    print(f"Created {len(gdf_proj)} buffer polygons")

    return gdf_proj


def save_buffers(buffers_gdf, output_file):
    """
    Save buffer GeoDataFrame to GeoJSON file.

    Parameters
    ----------
    buffers_gdf : list
        List of city records with buffer polygons
    output_file : Path
        Output file path (GeoJSON format)
    """
    print(f"\nSaving buffers to: {output_file}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to GeoJSON (portable format). Each city becomes one Feature, and
    # the 5km buffer polygon is stored in that Feature's geometry.
    features = []
    for row in buffers_gdf:
        properties = {key: value for key, value in row.items() if key != "geometry"}
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": row["geometry"],
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "city_buffers_5km",
        "crs": {
            "type": "name",
            "properties": {"name": CRS_GEOGRAPHIC},
        },
        "features": features,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(geojson, file, indent=2)

    print(f"Saved {len(buffers_gdf)} buffers to GeoJSON")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")


def slugify_city_name(city_name):
    """
    Convert a city name into a safe lowercase filename stem.

    Examples
    --------
    New York City -> new_york_city
    Sao Paulo -> sao_paulo
    """
    slug = city_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def save_city_specific_buffers(buffers_gdf, output_dir):
    """
    Save one single-city GeoJSON file for each buffer polygon.

    Parameters
    ----------
    buffers_gdf : list
        List of city records with buffer polygons
    output_dir : Path
        Folder where the city-specific GeoJSON files are written
    """
    print(f"\nSaving city-specific buffers to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.geojson"):
        old_file.unlink()

    written_files = []
    for row in buffers_gdf:
        city_slug = slugify_city_name(row["city_name"])
        city_output_file = output_dir / f"{city_slug}_5km.geojson"

        properties = {key: value for key, value in row.items() if key != "geometry"}
        geojson = {
            "type": "FeatureCollection",
            "name": f"{city_slug}_5km",
            "crs": {
                "type": "name",
                "properties": {"name": CRS_GEOGRAPHIC},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": row["geometry"],
                }
            ],
        }

        with city_output_file.open("w", encoding="utf-8") as file:
            json.dump(geojson, file, indent=2)

        written_files.append(city_output_file)

    print(f"Saved {len(written_files)} city-specific GeoJSON files")
    if written_files:
        print(f"  First file: {written_files[0]}")
        print(f"  Last file: {written_files[-1]}")


def visualize_buffers(buffers_gdf):
    """
    Create a simple visualization of the city buffers.

    Parameters
    ----------
    buffers_gdf : geopandas.GeoDataFrame
        GeoDataFrame with buffer polygons
    """
    try:
        import matplotlib.pyplot as plt

        print("\nCreating visualization...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot buffers colored by treatment status
        buffers_gdf.plot(
            ax=ax,
            column='treated',
            cmap='RdYlGn',
            alpha=0.5,
            edgecolor='black',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': 'Treatment Status', 'orientation': 'horizontal'}
        )

        # Add city labels
        for idx, row in buffers_gdf.iterrows():
            ax.annotate(
                text=row['city_name'],
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Study Cities - {BUFFER_RADIUS_KM}km Buffers\n'
                     f'(Green = Treated, Red = Control)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Save figure
        fig_path = FIGURES_DIR / 'city_buffers_map.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {fig_path}")

    except ImportError:
        print("Warning: Matplotlib not available - skipping visualization")
    except AttributeError:
        print("Warning: GeoPandas plotting not available - skipping visualization")


def main():
    """Main execution function."""
    print("=" * 70)
    print("CITY CENTROIDS AND BUFFERS CREATION")
    print("=" * 70)
    print()

    try:
        # Step 0: Create the city sample from README.md and WUP 2018 CBD coordinates
        write_cities_sample_from_wup2018(README_FILE, WUP2018_CBD_FILE, CITIES_SAMPLE_FILE)

        # Check required inputs exist
        check_required_inputs()

        # Step 1: Create city centroids
        centroids = create_city_centroids(CITIES_SAMPLE_FILE)

        # Step 2: Create buffers
        buffers = create_city_buffers(centroids, radius_km=BUFFER_RADIUS_KM)

        # Step 3: Save to file
        save_buffers(buffers, CITY_BUFFERS_FILE)

        # Step 4: Save one GeoJSON per city for Planet scene searches
        save_city_specific_buffers(buffers, CITY_BUFFERS_BY_CITY_DIR)

        # Step 5: Visualize (optional)
        visualize_buffers(buffers)

        print()
        print("=" * 70)
        print("SUCCESS: City buffers created successfully")
        print("=" * 70)
        print()
        print("Output file:", CITY_BUFFERS_FILE)
        print("City-specific output folder:", CITY_BUFFERS_BY_CITY_DIR)
        print()
        print("Summary:")
        print(f"  - Total cities: {len(buffers)}")
        if buffers and "treated" in buffers[0]:
            treated_count = sum(int(row["treated"]) for row in buffers)
            print(f"  - Treated cities: {treated_count}")
            print(f"  - Control cities: {len(buffers) - treated_count}")
        else:
            print("  - Treatment status: not provided")
        print(f"  - Buffer radius: {BUFFER_RADIUS_KM} km")
        print(f"  - Coordinate system: {CRS_PROJECTED}")
        print()
        print("Next steps:")
        print("  1. Download satellite data (python code/download_modis_lst/download_modis_lst_v1.py)")
        print("  2. Convert HDF4 to arrays (python code/convert_hdf_to_arrays/convert_hdf_to_arrays_v1.py)")
        print()

    except Exception as e:
        print()
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
