from __future__ import annotations
import numpy as np
from models.iris.vnext.graphics_truth_v1 import build_graphics_relation_truth_v1

def test_graphics_truth_uses_anonymous_equivalence_not_categories():
    faces=np.asarray([[0,1,2],[2,1,3]],dtype=np.int64);visible=np.ones((8,2),dtype=bool);depth=np.stack([np.asarray([.2,.4]) for _ in range(8)]);truth=build_graphics_relation_truth_v1(faces=faces,anonymous_drawable_ids=("anon-0","anon-1"),visible_face_mask=visible,face_depth=depth,occluding_face_pairs_by_view={i:((0,1),) for i in range(8)});assert truth.static_rows[0].same_drawable_affinity==0.;assert truth.static_rows[0].cut_probability==1.;assert len(truth.view_rows)==8;assert all(r.front_probability==1. for r in truth.view_rows);assert truth.metadata["categorical_semantics_used"] is False;assert truth.metadata["teacher_drawable_ids_exposed_to_inference"] is False
