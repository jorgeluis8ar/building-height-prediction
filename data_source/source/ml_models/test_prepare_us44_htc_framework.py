#!/usr/bin/env python3
"""Deterministic contract tests for the US44 dataset framework."""

from __future__ import annotations

import random
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prepare_us44_htc_framework import choose_delivered_raster, select_four


class US44FrameworkTest(unittest.TestCase):
    def test_city_split_is_reproducible_and_disjoint(self) -> None:
        cities = [f"city_{index:02d}" for index in range(44)]
        first, second = cities[:], cities[:]
        random.Random(20261005).shuffle(first)
        random.Random(20261005).shuffle(second)
        self.assertEqual(first, second)
        train, validation, test = set(first[:19]), set(first[19:38]), set(first[38:])
        self.assertEqual((len(train), len(validation), len(test)), (19, 19, 6))
        self.assertFalse(train & validation or train & test or validation & test)

    def test_four_scene_selection_prefers_exact_roles(self) -> None:
        roles = (("summer", "N"), ("summer", "S"), ("winter", "N"), ("winter", "S"),
                 ("summer", "E"), ("winter", "W"), ("summer", "W"), ("winter", "E"))
        frame = pd.DataFrame([{
            "city_slug": "example", "scene_id": f"scene_{index}",
            "selection_local_season": season, "selection_cardinal_direction": direction,
            "selection_filter_tier": 0, "aoi_coverage_percent": 100,
            "cloud_cover": 0, "clear_percent": 100,
        } for index, (season, direction) in enumerate(roles, 1)])
        selected, audit = select_four(frame)
        self.assertEqual(selected, {"scene_1", "scene_2", "scene_3", "scene_4"})
        self.assertTrue(all(row["exact_match"] for row in audit))

    def test_four_scene_selection_records_fallbacks(self) -> None:
        frame = pd.DataFrame([{
            "city_slug": "example", "scene_id": f"scene_{index}",
            "selection_local_season": "summer" if index <= 4 else "winter",
            "selection_cardinal_direction": "E" if index % 2 else "W",
            "selection_filter_tier": 0, "aoi_coverage_percent": 100,
            "cloud_cover": 0, "clear_percent": 100,
        } for index in range(1, 9)])
        selected, audit = select_four(frame)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(row["match_type"] == "same_season" for row in audit))

    def test_delivery_raster_prefers_clip_with_variable_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "scene_3B_AnalyticMS_SR_harmonized.tif"
            clipped = root / "scene_3B_AnalyticMS_SR_harmonized_clip.tif"
            raw.touch()
            clipped.touch()
            selected = choose_delivered_raster([raw, clipped], "SR test raster", root)
            self.assertEqual(selected, clipped.resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
