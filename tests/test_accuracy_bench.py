"""The truth bench: painted glue lines of known edges, the edge methods, the scoring."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import accuracy_bench  # noqa: E402
from segmentation import bead_edges, edge_methods, synthetic_bead  # noqa: E402


def fabric(count=500, noise=1.5, seed=0):
    """Plain fabric: a level, a slow gradient and fine noise."""
    rng = np.random.default_rng(seed)
    offsets = np.arange(-bead_edges.REACH, bead_edges.REACH + 1, dtype=np.float32)
    strip = 200.0 + 0.02 * offsets[None, :] + rng.normal(0, noise, (count, len(offsets)))
    return strip.astype(np.float32), offsets


class GeneratorTests(unittest.TestCase):
    def test_the_bead_is_painted_with_the_width_and_edges_it_reports(self):
        _, offsets = fabric(noise=0.0)
        strip = np.full((500, len(offsets)), 200.0, np.float32)          # flat, so half-depth is exact
        painted, truth = synthetic_bead.paint(strip, offsets, np.random.default_rng(1), colour='white',
                                              width=30.0, shiny_share=0.0, wander=5.0, width_wobble=0.0)
        self.assertAlmostEqual(float(np.median(truth['high'] - truth['low'])), 30.0, delta=0.01)
        # Half-way down the band's depth lies exactly on the reported edges.
        row = 250
        level = float(painted[row, :10].mean())
        depth = level - float(painted[row, int(round((truth['low'][row] + truth['high'][row]) / 2 + bead_edges.REACH))])
        half = level - depth / 2.0
        across = offsets.astype(float)
        inside = across[painted[row] < half]
        self.assertAlmostEqual(float(inside.min()), truth['low'][row], delta=1.0)
        self.assertAlmostEqual(float(inside.max()), truth['high'][row], delta=1.0)

    def test_a_defect_changes_the_true_width_where_it_is_put(self):
        strip, offsets = fabric(noise=0.0)
        _, truth = synthetic_bead.paint(strip, offsets, np.random.default_rng(1), width=30.0, width_wobble=0.0,
                                        defects=[(200, 40, 6.0)])
        width = truth['high'] - truth['low']
        self.assertAlmostEqual(float(width[220]), 36.0, delta=0.01)
        self.assertAlmostEqual(float(width[100]), 30.0, delta=0.01)
        self.assertEqual(float(truth['defect'][100]), 0.0)


class EdgeMethodTests(unittest.TestCase):
    """On a clean painted bead every method must put the edges where they are."""

    def setUp(self):
        strip, self.offsets = fabric(noise=1.0)
        self.strip, self.truth = synthetic_bead.paint(
            strip, self.offsets, np.random.default_rng(2), colour='dark', width=30.0, wander=4.0,
            shiny_share=0.0, width_wobble=0.0)
        self.read = bead_edges.read_strip(self.strip, self.offsets)

    def error(self, name):
        low, high = edge_methods.METHODS[name](self.strip, self.offsets, self.read)
        edge, width, _ = accuracy_bench.score(low, high, self.truth)
        return float(np.sqrt(np.mean(edge ** 2))), float(np.mean(width))

    def test_slope_methods_and_the_profile_fit_find_a_clear_edge_within_a_pixel(self):
        for name in ('baseline', 'slope_subpixel', 'slope_scale3', 'profile_fit', 'phase'):
            rms, width_bias = self.error(name)
            self.assertLess(rms, 1.0, name)
            self.assertLess(abs(width_bias), 1.0, name)

    def test_the_profile_fit_is_closer_than_whole_pixel_slopes(self):
        self.assertLess(self.error('profile_fit')[0], self.error('baseline')[0])


class ScoringTests(unittest.TestCase):
    def test_edges_outside_the_truth_score_positive_and_widen_the_width(self):
        count = 300
        truth = {'low': np.full(count, -15.0), 'high': np.full(count, 15.0)}
        edge, width, cover = accuracy_bench.score(np.full(count, -17.0), np.full(count, 16.0), truth)
        self.assertAlmostEqual(float(edge.mean()), 1.5)
        self.assertAlmostEqual(float(width.mean()), 3.0)
        self.assertAlmostEqual(cover, 1.0)

    def test_the_ribbon_follows_edges_given_in_strip_coordinates(self):
        strip, offsets = fabric(noise=1.0)
        painted, truth = synthetic_bead.paint(strip, offsets, np.random.default_rng(4), colour='dark',
                                              width=32.0, wander=3.0, shiny_share=0.0)
        read = bead_edges.read_strip(painted, offsets)
        low, high, _ = accuracy_bench.ribbon_edges(read['low'], read['high'], read, offsets)
        edge, width, cover = accuracy_bench.score(low, high, truth)
        self.assertGreater(cover, 0.9)
        self.assertLess(abs(float(width.mean())), 1.0)
        self.assertLess(float(np.sqrt(np.mean(edge ** 2))), 1.5)


if __name__ == '__main__':
    unittest.main()
