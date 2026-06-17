"""
Deprecated Planet Download Entry Point

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - None

Produces (outputs for later stages):
    - None

Description:
    The Planet workflow is intentionally split into two scripts:
    order_selected_planet_scenes.py creates AOI-clipped Planet orders and
    writes the order manifest. download_ordered_planet_scenes.py checks the
    manifest and downloads only completed orders.

Usage:
    python3 data_source/source/planet_imagery/order_selected_planet_scenes.py --dry-run
    python3 data_source/source/planet_imagery/download_ordered_planet_scenes.py --dry-run

Expected runtime: immediate
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "download_selected_planet_scenes.py has been deprecated to avoid "
        "mixing Planet order creation with downloads.\n"
        "Use order_selected_planet_scenes.py to create orders, then use "
        "download_ordered_planet_scenes.py after the orders are complete."
    )


if __name__ == "__main__":
    main()
