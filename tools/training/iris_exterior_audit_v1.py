"""Frozen Knight R512 source certificate, independently counted for every subject."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from models.iris.v4.source_hull_lattice_v4_3 import lattice_points_from_linear_indices_v43,source_hull_exterior_margin_v43

@torch.no_grad()
def exterior_scan(subject,query=None,*,resolution=512,device="cuda",chunk=32768,expected_count=None):
    total=resolution**3;certified=nonpositive=0;start=time.monotonic();last=start
    center=subject['center_xyz'].to(device);half=subject['normalization_half_extent']
    kwargs={'distance_fields_px':subject['signed_2d'].to(device).clamp_min(0.),
            'camera_origins_normalized':(subject['camera_origins'].to(device)-center)/half,
            'camera_right':subject['camera_right'].to(device),'camera_screen_up':subject['camera_screen_up'].to(device),
            'camera_forward':subject['camera_forward'].to(device),
            'camera_half_extent_normalized':subject['camera_half_extent'].to(device)/half,
            'pixel_quantization_guard_px':math.sqrt(2.)}
    for st in range(0,total,chunk):
        en=min(total,st+chunk)
        q=lattice_points_from_linear_indices_v43(torch.arange(st,en,device=device),resolution=resolution,device=device)
        c=source_hull_exterior_margin_v43(q,**kwargs)['certified']
        certified+=int(c.sum().cpu())
        if query is not None:
            pred=torch.as_tensor(query(q.cpu().numpy()),device=device).reshape(-1)
            if pred.shape!=(len(q),) or not bool(torch.isfinite(pred).all()):
                raise ValueError('R512_NONFINITE_OR_INCOMPLETE_LEARNED_FIELD')
            nonpositive+=int((c & (pred<=0.)).sum().cpu())
        if time.monotonic()-last>15:
            print(f"[R{resolution}] {subject['subject_id']} points={en}/{total} certified={certified} nonpositive={nonpositive} seconds={time.monotonic()-start:.1f}",flush=True)
            last=time.monotonic()
    if certified==0:
        raise ValueError('R512_EMPTY_SOURCE_EXTERIOR_CERTIFICATE')
    if expected_count is not None and certified!=expected_count:
        raise ValueError('R512_SOURCE_CERTIFIED_EXTERIOR_COUNT_DRIFT')
    return {'schema':'RealSaS.SharedIrisSourceExteriorSignAudit.v1','resolution':resolution,'point_count':total,
            'certified_exterior_count':certified,'expected_certified_exterior_count':expected_count,
            'nonpositive_certified_exterior_count':nonpositive if query is not None else None,
            'source_only_census':query is None,'production_r512':resolution==512,
            'sign_pass':query is not None and certified>0 and nonpositive==0,
            'source_count_drift':False,'selection_blind':True,'training_feedback':False,
            'pixel_quantization_guard_px':math.sqrt(2.),'seconds':time.monotonic()-start}
