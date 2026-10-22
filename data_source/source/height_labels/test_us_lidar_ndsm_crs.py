#!/usr/bin/env python3
"""Regression tests for USGS legacy project grouping and CRS fallbacks."""

import unittest

import pyproj

import build_lidar_ndsm_raster as builder
from run_us_lidar_to_planet_ndsm import project_key_from_url


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


if __name__ == "__main__":
    unittest.main()
