# Planet Imagery

This folder contains scripts for searching Planet metadata, selecting reviewed
city scenes, creating AOI-clipped Planet orders, and downloading completed
orders.

## Scripts

Search metadata only:

```bash
python3 data_source/source/planet_imagery/search_planet_city_scenes.py
```

Select two reviewed scenes per city:

```bash
python3 data_source/source/planet_imagery/select_planet_city_scenes.py
```

Create Planet orders after review:

```bash
python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --dry-run
python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --confirm-order
```

Download completed Planet orders:

```bash
python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --dry-run
python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --confirm-download
```

The old all-in-one script is intentionally deprecated:

```bash
python3 data_source/source/planet_imagery/download_selected_planet_scenes.py
```

It exits with instructions and does not create orders or download data.

## Credentials

Planet authentication is checked in this order:

1. `data_source/source/planet_imagery/PLANET_API.py`
2. `PL_API_KEY` environment variable
3. Saved Planet OAuth profile

`PLANET_API.py` is intentionally ignored by Git. It should contain:

```python
PL_API_KEY = "your_key_here"
```

Never commit this file or paste the key into tracked scripts, notebooks,
README files, logs, or manifests.

## Source-Date Manifest

Scene selection uses:

```text
data_source/source/planet_imagery/building_footprint_source_dates.csv
```

This manifest records the building-footprint reference date used to center the
Planet scene selection, the dataset update date where known, source URLs, and
date-confidence notes.

## Selected Scene Output

The selector writes:

```text
data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv
```

The output contains two rows per city:

- `winter_jan_dec`
- `summer_jun_jul`

Each row keeps the original Planet scene metadata and adds selection fields
such as the footprint reference date, pair gap in days, and whether the strict
zero-cloud/100%-AOI condition was met.

The selector also checks Planet asset availability and writes:

```text
available_asset_types
has_ortho_analytic_8b_sr
selected_asset_type
selected_asset_type_reason
```

`selected_asset_type` is assigned consistently within each city pair. If both
selected scenes have `ortho_analytic_8b_sr`, both rows use it. If either scene
lacks it, both rows use the best shared fallback asset type. At this stage,
Oslo is the only reviewed city pair using `ortho_analytic_4b_sr`.

## Order Manifest

The order script writes:

```text
data_source/data/planet_imagery/generated/planet_orders_manifest.csv
```

This manifest is the bridge between ordering and downloading. It records:

- city and season
- scene ID
- selected asset type
- Orders API product bundle
- Planet order name and order ID
- order state and timestamps
- intended output folder
- AOI path
- serialized order request
- download status and downloaded files

The order script does not wait for order completion and never downloads data.
This keeps Planet's asynchronous order processing separate from local file
downloads.

## Download Output

The download script reads `planet_orders_manifest.csv`, checks each order ID,
and downloads only orders with state `success` or `partial`. It skips orders
with state `queued`, `running`, `failed`, or any other non-downloadable state.

Downloaded raw Planet assets are written to:

```text
data_source/data/planet_imagery/source/<city_slug>/<season>_<scene_id>/
```

The script updates the manifest with `download_status`, `downloaded_files`,
`order_state`, and `download_checked_on`.

## Safety Rules

- All script defaults and manifest paths resolve from the
  `building-height-prediction` repository root, regardless of the terminal's
  current working directory.
- Manifest paths remain portable and repository-relative. Scripts resolve them
  from the detected repository root and reject paths outside the repository.
- Run `--dry-run` first for both ordering and downloading.
- Use `--confirm-order` only when you are ready to create Planet orders.
- Use `--confirm-download` only after the portal or manifest shows orders are complete.
- Do not rerun order creation blindly; use the manifest to avoid duplicate orders.
- Do not manually edit downloaded Planet imagery. Treat `source/` as raw data.
