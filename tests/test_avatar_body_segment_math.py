import importlib
import sys
import types
import unittest

import numpy as np


def _install_import_stubs():
    # modules.typing pulls in insightface just for a type alias; stub it out
    # so this test only depends on numpy/cv2, which avatar_body.py itself
    # depends on directly.
    sys.modules["modules.typing"] = types.SimpleNamespace(Frame=object)
    sys.modules.setdefault(
        "modules.globals",
        types.SimpleNamespace(avatar_body_strength=100.0),
    )


class _Landmark:
    def __init__(self, x, y, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility
        self.presence = presence


class AvatarBodySegmentMathTest(unittest.TestCase):
    def setUp(self):
        # Importing the real "modules" package tree here would otherwise
        # leak into sys.modules for the rest of the run — other test files
        # (e.g. test_core_map_faces_fallback) stub out "modules.core"'s
        # dependencies and expect a clean re-import. Snapshot and restore so
        # this test's real imports never outlive it.
        self._sys_modules_snapshot = set(sys.modules.keys())
        self.addCleanup(self._restore_sys_modules)

        _install_import_stubs()
        self.avatar_body = importlib.import_module(
            "modules.processors.frame.avatar_body"
        )
        importlib.reload(self.avatar_body)

    def _restore_sys_modules(self):
        for name in list(sys.modules.keys()):
            if name not in self._sys_modules_snapshot:
                del sys.modules[name]

    def test_strip_quad_is_centered_and_perpendicular(self):
        p_a = np.array([0.0, 0.0], dtype=np.float32)
        p_b = np.array([10.0, 0.0], dtype=np.float32)
        quad = self.avatar_body._strip_quad(p_a, p_b, half_width=2.0)

        self.assertEqual(quad.shape, (4, 2))
        # Bone lies along +x, so the offset must be purely along y, and the
        # two long edges (a+offset->b+offset, b-offset->a-offset) must each
        # run parallel to the bone with length == half_width * 2 apart.
        np.testing.assert_allclose(quad[0][0], 0.0, atol=1e-5)
        np.testing.assert_allclose(quad[1][0], 10.0, atol=1e-5)
        np.testing.assert_allclose(quad[2][0], 10.0, atol=1e-5)
        np.testing.assert_allclose(quad[3][0], 0.0, atol=1e-5)
        self.assertAlmostEqual(abs(float(quad[0][1])), 2.0, places=5)
        self.assertAlmostEqual(float(quad[0][1]), float(quad[1][1]), places=5)
        self.assertAlmostEqual(float(quad[2][1]), float(quad[3][1]), places=5)
        self.assertAlmostEqual(float(quad[0][1]), -float(quad[2][1]), places=5)

    def test_strip_quad_degenerate_bone_does_not_crash(self):
        p_a = np.array([5.0, 5.0], dtype=np.float32)
        p_b = np.array([5.0, 5.0], dtype=np.float32)
        quad = self.avatar_body._strip_quad(p_a, p_b, half_width=1.0)
        self.assertEqual(quad.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(quad)))

    def test_clamp_length_ratio_within_band_passes_through(self):
        # live is 1.2x the avatar's bone length, well inside [0.5, 1.8].
        ratio = self.avatar_body._clamp_length_ratio(live_len=12.0, avatar_len=10.0)
        self.assertAlmostEqual(ratio, 1.2, places=5)

    def test_clamp_length_ratio_clamps_extreme_stretch(self):
        ratio = self.avatar_body._clamp_length_ratio(live_len=100.0, avatar_len=10.0)
        self.assertAlmostEqual(ratio, self.avatar_body._LENGTH_RATIO_MAX, places=5)

    def test_clamp_length_ratio_clamps_extreme_shrink(self):
        ratio = self.avatar_body._clamp_length_ratio(live_len=1.0, avatar_len=10.0)
        self.assertAlmostEqual(ratio, self.avatar_body._LENGTH_RATIO_MIN, places=5)

    def test_clamp_length_ratio_guards_zero_avatar_length(self):
        ratio = self.avatar_body._clamp_length_ratio(live_len=5.0, avatar_len=0.0)
        self.assertEqual(ratio, 1.0)

    def test_landmark_xy_scales_by_frame_size(self):
        landmarks = [_Landmark(x=0.5, y=0.25)]
        pt = self.avatar_body._landmark_xy(landmarks, 0, w=200, h=100)
        np.testing.assert_allclose(pt, [100.0, 25.0])

    def test_landmark_xy_rejects_low_visibility(self):
        landmarks = [_Landmark(x=0.5, y=0.5, visibility=0.1)]
        pt = self.avatar_body._landmark_xy(landmarks, 0, w=100, h=100)
        self.assertIsNone(pt)

    def test_landmark_xy_rejects_low_presence(self):
        landmarks = [_Landmark(x=0.5, y=0.5, presence=0.1)]
        pt = self.avatar_body._landmark_xy(landmarks, 0, w=100, h=100)
        self.assertIsNone(pt)

    def test_landmark_xy_out_of_range_index_is_none(self):
        landmarks = [_Landmark(x=0.5, y=0.5)]
        pt = self.avatar_body._landmark_xy(landmarks, 5, w=100, h=100)
        self.assertIsNone(pt)


if __name__ == "__main__":
    unittest.main()
