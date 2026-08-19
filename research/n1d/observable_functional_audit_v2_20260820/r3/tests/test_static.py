from __future__ import annotations
import ast, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "src" / "observable_phase.py"
FORBIDDEN = ("sidecar", "teacher", "surface_points", "surface_normals", "surface_visibility", "camera_center", "camera_half_extent")

def load(name, path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);assert s.loader is not None;s.loader.exec_module(m);return m

def main():
    text=OBS.read_text().lower()
    for token in FORBIDDEN:
        assert token not in text, f"truth-firewall token leaked into observable source: {token}"
    ast.parse(OBS.read_text())
    ast.parse((ROOT/'src'/'evaluation_phase.py').read_text())
    ast.parse((ROOT/'src'/'mechanics_metrics.py').read_text())
    ast.parse((ROOT/'src'/'prepare_runtime.py').read_text())
    mm=load('mm',ROOT/'src'/'mechanics_metrics.py')
    idx=np.tile(np.arange(12),(16,1)) % 16
    S=mm.motif_from_neighbors(idx,3); assert 3 in S and len(S)>=1
    assert abs(mm.nrms_truth(np.ones(4),np.ones(4)))<1e-12
    assert abs(mm.nrms_effect(np.ones(4),np.ones(4)))<1e-12
    a=np.array([1.,2.,3.]); b=np.array([2.,3.,4.]); assert abs(mm.nrms_effect(a,b)-mm.nrms_effect(b,a))<1e-12
    print('STATIC_TESTS_PASS')
if __name__=='__main__':main()
