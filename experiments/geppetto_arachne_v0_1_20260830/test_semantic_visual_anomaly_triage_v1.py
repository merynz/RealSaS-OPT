import unittest
import numpy as np

from semantic_visual_anomaly_triage_v1 import component_view_metrics


class VisualAnomalyTriageTests(unittest.TestCase):
    def test_large_rectangle_scores_above_small_blob(self):
        a = np.zeros((1024, 1024), dtype=bool)
        a[100:900, 150:850] = True
        b = np.zeros((1024, 1024), dtype=bool)
        b[450:574, 450:574] = True
        ma = component_view_metrics(a)
        mb = component_view_metrics(b)
        self.assertGreater(ma["dominant_rectangle_score"], mb["dominant_rectangle_score"])

    def test_two_disconnected_subjects_have_multi_score(self):
        a = np.zeros((1024, 1024), dtype=bool)
        a[200:500, 100:300] = True
        a[500:850, 700:950] = True
        m = component_view_metrics(a)
        self.assertGreaterEqual(m["component_count"], 2)
        self.assertGreater(m["multi_disconnected_score"], 0.0)

    def test_single_subject_has_zero_multi_score(self):
        a = np.zeros((1024, 1024), dtype=bool)
        a[300:700, 350:650] = True
        m = component_view_metrics(a)
        self.assertEqual(m["component_count"], 1)
        self.assertEqual(m["multi_disconnected_score"], 0.0)

    def test_thin_support_is_detectably_thinner(self):
        thin = np.zeros((1024, 1024), dtype=bool)
        thin[500:504, 100:900] = True
        thick = np.zeros((1024, 1024), dtype=bool)
        thick[450:574, 100:900] = True
        mt = component_view_metrics(thin)
        mk = component_view_metrics(thick)
        self.assertLess(mt["medial_p01_px"], mk["medial_p01_px"])

    def test_offcenter_measurement(self):
        c = np.zeros((1024, 1024), dtype=bool)
        c[450:574, 450:574] = True
        e = np.zeros((1024, 1024), dtype=bool)
        e[50:174, 50:174] = True
        self.assertLess(component_view_metrics(c)["offcenter_fraction"], component_view_metrics(e)["offcenter_fraction"])


if __name__ == "__main__":
    unittest.main()
