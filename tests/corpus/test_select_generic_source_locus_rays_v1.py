"""Pure-geometry deterministic source-locus sampling, no labels or RGB."""
from __future__ import annotations

import copy
import unittest

from tools.corpus.select_generic_source_locus_rays_v1 import select_rays

def ray(v,a,b,px=1):
    return {
        "status":"VALID_GEOMETRIC_FULL_STACK_CANDIDATE",
        "complete":True,"has_depth_tie":False,
        "view_index":v,"pixel_x":px,"pixel_y":4,
        "hits":[{"face_index_teacher_only":a},
                {"face_index_teacher_only":b}],
    }

class ViewDiverseSelectionTests(unittest.TestCase):
    def test_prefers_opposite_views_over_adjacent_pixels(self):
        rows=[ray(0,2,8,1),ray(0,3,7,2),ray(1,10,11,1),ray(4,8,12,1)]
        selected=select_rays(rows)
        self.assertEqual([x["view_index"] for x in selected],[0,4])

    def test_deterministic_against_input_order(self):
        rows=[ray(0,2,8,1),ray(0,3,7,2),ray(4,8,12,1)]
        self.assertEqual(select_rays(rows),select_rays(list(reversed(rows))))

    def test_prefers_unseen_facepair_across_views(self):
        rows=[ray(0,2,8,1),ray(4,2,8,1),ray(4,4,10,2)]
        selected=select_rays(rows)
        self.assertEqual(selected[1]["pixel_x"],2)

    def test_more_views_bounded(self):
        rows=[ray(v,5+v,20+v) for v in range(8)]
        selected=select_rays(rows,maximum=4)
        self.assertEqual([x["view_index"] for x in selected],[0,4,2,6])

    def test_single_view_not_falsely_qualified(self):
        with self.assertRaisesRegex(ValueError,"TWO_INDEPENDENT_VIEWS_REQUIRED"):
            select_rays([ray(0,1,2),ray(0,1,3,2)])

    def test_tied_valid_ray_rejected(self):
        rows=[ray(0,1,2),ray(4,3,5)]
        rows[1]["has_depth_tie"]=True
        with self.assertRaisesRegex(ValueError,"FALSE_VALID_GEOMETRIC_RAY"):
            select_rays(rows)

    def test_self_face_skipped(self):
        rows=[ray(0,1,1),ray(4,2,3)]
        with self.assertRaisesRegex(ValueError,"TWO_INDEPENDENT_VIEWS_REQUIRED"):
            select_rays(rows)

    def test_invalid_selected_face_id(self):
        rows=[ray(0,1,2),ray(4,-1,3)]
        with self.assertRaisesRegex(ValueError,"SOURCE_FACES"):
            select_rays(rows)

    def test_invalid_quota(self):
        with self.assertRaisesRegex(ValueError,"VIEW_QUOTA"):
            select_rays([ray(0,1,2),ray(4,3,4)],maximum=1)

    def test_invalid_pixel(self):
        rows=[ray(0,1,2),ray(4,3,4)]
        rows[1]["pixel_x"]=-1
        with self.assertRaisesRegex(ValueError,"PIXEL"):
            select_rays(rows)

if __name__=="__main__":
    unittest.main()
