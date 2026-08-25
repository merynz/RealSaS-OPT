from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
from pv5_field_representation_closure import bilinear_matrix, solve_view, stats, seed_for

rng = np.random.default_rng(20260824)
xy = rng.uniform(-0.95, 0.95, size=(4096,2))
d = 0.13*xy[:,0] - 0.21*xy[:,1] + 0.037
rows = []
for hw in (64,128,256):
    pred, s = solve_view(xy,d,hw)
    err = np.abs(pred-d)
    st = stats(err)
    if st['p95'] > 5e-6:
        raise RuntimeError((hw,st))
    A = bilinear_matrix(xy[:17],hw,hw)
    sums = np.asarray(A.sum(axis=1)).reshape(-1)
    if not np.allclose(sums,1.0,atol=1e-12,rtol=0):
        raise RuntimeError('bilinear row-sum drift')
    rows.append({'hw':hw,'linear_depth_abs':st,'solver':s})
if seed_for('asset_x',3) != seed_for('asset_x',3) or seed_for('asset_x',3)==seed_for('asset_x',4):
    raise RuntimeError('seed contract drift')
print(json.dumps({
    'schema':'RealSaS.IRISSinglePoseV2.PV5FieldRepresentationPreflight.v1',
    'status':'PASS',
    'camera_json_consumed':False,
    'neural_optimizer_steps':0,
    'grid_sample_semantics':'bilinear_border_align_corners_false',
    'rows':rows,
},indent=2))
