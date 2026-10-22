#!/usr/bin/env python3
"""Regression tests for USGS legacy project grouping and CRS fallbacks."""

import unittest

import pandas as pd
import pyproj
from shapely.geometry import box

import build_lidar_ndsm_raster as builder
from run_us_lidar_to_planet_ndsm import project_key_from_url, select_latest_acquisition


class EmptyCrsHeader:
    def parse_crs(self):
        return None


class EmbeddedCrsHeader:
    def __init__(self, epsg: int):
        self.epsg = epsg

    def parse_crs(self):
        return pyproj.CRS.from_epsg(self.epsg)


class USLidarNdsmCrsTest(unittest.TestCase):
    def test_nested_legacy_projects_are_distinct(self):
        base = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects"
        self.assertEqual(
            project_key_from_url(f"{base}/legacy/NJ_MERCERCO_2009/LAZ/a.laz"),
            "legacy/NJ_MERCERCO_2009",
        )
        self.assertEqual(
            project_key_from_url(f"{base}/legacy/NJ_CAMDEN_BURLINGTON_2004/LAZ/b.laz"),
            "legacy/NJ_CAMDEN_BURLINGTON_2004",
        )
        self.assertEqual(
            project_key_from_url(f"{base}/FL_MiamiDade_D23/workunit/LAZ/c.laz"),
            "FL_MiamiDade_D23",
        )

    def test_documented_legacy_crs_and_vertical_units(self):
        crs, scale = builder.point_crs_from_header(
            EmptyCrsHeader(), builder.PROJECT_ROOT / "KY_KENTON_CAMBELLCO_2007/tile.laz"
        )
        self.assertEqual(crs.to_epsg(), 2891)
        self.assertAlmostEqual(scale, 1200.0 / 3937.0)

        crs, scale = builder.point_crs_from_header(
            EmptyCrsHeader(), builder.PROJECT_ROOT / "NJ_MERCERCO_2009/tile.laz"
        )
        self.assertEqual(crs.to_epsg(), 3424)
        self.assertAlmostEqual(scale, 1200.0 / 3937.0)

        crs, scale = builder.point_crs_from_header(
            EmptyCrsHeader(), builder.PROJECT_ROOT / "NJ_CAMDEN_BURLINGTON_2004/tile.laz"
        )
        self.assertEqual(crs.to_epsg(), 3424)
        self.assertAlmostEqual(scale, 1200.0 / 3937.0)

        crs, scale = builder.point_crs_from_header(
            EmptyCrsHeader(), builder.PROJECT_ROOT / "NJ_New_Jersey_SANDY_LiDAR_15/tile.laz"
        )
        self.assertEqual(crs.to_epsg(), 6347)
        self.assertEqual(scale, 1.0)

    def test_documented_vertical_scale_applies_with_embedded_crs(self):
        crs, scale = builder.point_crs_from_header(
            EmbeddedCrsHeader(2891),
            builder.PROJECT_ROOT / "KY_KENTON_CAMBELLCO_2007/tile.laz",
        )
        self.assertEqual(crs.to_epsg(), 2891)
        self.assertAlmostEqual(scale, 1200.0 / 3937.0)

        _, scale = builder.point_crs_from_header(
            EmbeddedCrsHeader(6438), builder.PROJECT_ROOT / "FL_MiamiDade_D23/tile.laz"
        )
        self.assertAlmostEqual(scale, 1200.0 / 3937.0)

    def test_newest_partial_project_is_retained_before_older_complete_project(self):
        tiles = pd.DataFrame(
            [
                {
                    "download_url": "https://example/Projects/new_project/unit/a.laz",
                    "publication_date": "2025-01-01",
                    "min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.6, "max_lat": 1.0,
                },
                {
                    "download_url": "https://example/Projects/legacy/old_project/b.laz",
                    "publication_date": "2015-01-01",
                    "min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0,
                },
            ]
        )
        selected, projects, _, _, coverage = select_latest_acquisition(
            box(0.0, 0.0, 1.0, 1.0), tiles, {}, "2010-01-01", "2010-12-31"
        )
        self.assertEqual(projects, ["new_project", "legacy/old_project"])
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(coverage, 99.0)


if __name__ == "__main__":
    unittest.main()
