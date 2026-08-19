from __future__ import annotations
import importlib.util, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('evaluation_phase_import_smoke',str(ROOT/'src'/'evaluation_phase.py'))
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; assert spec.loader is not None; spec.loader.exec_module(mod)
assert hasattr(mod,'evaluate_family') and hasattr(mod,'aggregate')
print('IMPORT_SMOKE_PASS')
