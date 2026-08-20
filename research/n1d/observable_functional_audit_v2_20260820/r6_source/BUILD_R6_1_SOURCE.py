from __future__ import annotations
import hashlib, shutil, sys
from pathlib import Path

BASE_COMMIT = "a36133e6d97d7f848ce7b8cddcb43b6067e23596"
BASE_ROOT = "research/n1d/observable_functional_audit_v2_20260820/r3"
BASE_EVAL_SHA256 = "3d39b0afc0213986dad02dc8965df3189d205076219a4016c4e04d9bd753fabc"

IMPORT_ANCHOR = "import argparse, hashlib, importlib.util, json\n"
IMPORT_REPLACEMENT = "import argparse, hashlib, importlib.util, json, sys\n"
LOAD_ANCHOR = "    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod\n"
LOAD_REPLACEMENT = "    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; sys.modules[name] = mod; spec.loader.exec_module(mod); return mod\n"

IMPORT_TEST = '''from __future__ import annotations
import importlib.util, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('evaluation_phase_import_smoke',str(ROOT/'src'/'evaluation_phase.py'))
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; assert spec.loader is not None; spec.loader.exec_module(mod)
assert hasattr(mod,'evaluate_family') and hasattr(mod,'aggregate')
print('IMPORT_SMOKE_PASS')
'''

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(base_root: Path, out_root: Path):
    if out_root.exists():
        raise RuntimeError(f"refusing to overwrite output tree: {out_root}")
    src_eval = base_root / "src" / "evaluation_phase.py"
    got = sha256(src_eval)
    if got != BASE_EVAL_SHA256:
        raise RuntimeError(f"base evaluator SHA mismatch: {got}")
    shutil.copytree(base_root, out_root)
    dst = out_root / "src" / "evaluation_phase.py"
    text = dst.read_text()
    if text.count(IMPORT_ANCHOR) != 1 or text.count(LOAD_ANCHOR) != 1:
        raise RuntimeError("R6.1 patch anchor mismatch")
    text = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    text = text.replace(LOAD_ANCHOR, LOAD_REPLACEMENT, 1)
    dst.write_text(text)
    (out_root / "tests" / "test_imports.py").write_text(IMPORT_TEST)
    print("R6_1_SOURCE_BUILD_PASS")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: BUILD_R6_1_SOURCE.py <checked-out-base-root> <out-root>")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
