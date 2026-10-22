import unittest

from models.iris.vnext.graphics_truth_v2 import (
    STATIC_RELATION_KEYS_V2,
    VIEW_RELATION_KEYS_V2,
)
from models.iris.vnext.relation_support_v2 import (
    BinarySupportStatusV2,
    aggregate_binary_stats_v2,
    qualify_binary_support_v2,
    qualify_relation_manifest_support_v2,
)


class RelationSupportV2Tests(unittest.TestCase):
    def test_no_support_is_not_court_eligible(self):
        q = qualify_binary_support_v2({"valid": 0, "positive": 0, "negative": 0})
        self.assertEqual(q.status, BinarySupportStatusV2.NO_VALID_SUPPORT)
        self.assertFalse(q.court_eligible)

    def test_single_class_support_is_not_court_eligible(self):
        q = qualify_binary_support_v2({"valid": 10, "positive": 10, "negative": 0})
        self.assertEqual(q.status, BinarySupportStatusV2.SINGLE_CLASS_SUPPORT)
        self.assertFalse(q.court_eligible)

    def test_binary_support_is_court_support_eligible(self):
        q = qualify_binary_support_v2({"valid": 10, "positive": 5, "negative": 5})
        self.assertEqual(q.status, BinarySupportStatusV2.BINARY_SUPPORT)
        self.assertTrue(q.court_eligible)

    def test_aggregate_preserves_counts(self):
        stats = aggregate_binary_stats_v2(
            [
                {"total": 7, "valid": 6, "abstained": 1, "positive": 4, "negative": 2},
                {"total": 5, "valid": 4, "abstained": 1, "positive": 1, "negative": 3},
            ]
        )
        self.assertEqual(stats["total"], 12)
        self.assertEqual(stats["valid"], 10)
        self.assertEqual(stats["abstained"], 2)
        self.assertEqual(stats["positive"], 5)
        self.assertEqual(stats["negative"], 5)
        self.assertEqual(stats["positive_fraction"], 0.5)

    def test_manifest_marks_current_occlusion_shape_as_single_class(self):
        static = {
            name: {
                "total": 10,
                "valid": 0,
                "abstained": 10,
                "positive": 0,
                "negative": 0,
            }
            for name in STATIC_RELATION_KEYS_V2
        }
        view = {}
        for name in VIEW_RELATION_KEYS_V2:
            if name == "occlusion_probability":
                view[name] = {
                    "total": 10,
                    "valid": 10,
                    "abstained": 0,
                    "positive": 10,
                    "negative": 0,
                }
            else:
                view[name] = {
                    "total": 10,
                    "valid": 10,
                    "abstained": 0,
                    "positive": 5,
                    "negative": 5,
                }
        manifest = {
            "subjects": [
                {
                    "views": [
                        {"static": static, "view": view},
                        {"static": static, "view": view},
                    ]
                }
            ]
        }
        result = qualify_relation_manifest_support_v2(manifest)
        self.assertEqual(
            result["occlusion_probability"].status,
            BinarySupportStatusV2.SINGLE_CLASS_SUPPORT,
        )
        self.assertFalse(result["occlusion_probability"].court_eligible)
        self.assertEqual(
            result["front_probability"].status,
            BinarySupportStatusV2.BINARY_SUPPORT,
        )
        self.assertTrue(result["front_probability"].court_eligible)
        for name in STATIC_RELATION_KEYS_V2:
            self.assertEqual(result[name].status, BinarySupportStatusV2.NO_VALID_SUPPORT)
            self.assertFalse(result[name].court_eligible)

    def test_inconsistent_stats_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "INCONSISTENT"):
            qualify_binary_support_v2({"valid": 10, "positive": 8, "negative": 1})


if __name__ == "__main__":
    unittest.main()
