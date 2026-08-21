from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

FIXED_TIME = (2026, 8, 21, 0, 0, 0)
MODE = 0o100644 << 16
MEMBERS = (
    'research/n1d/observable_geometry_closure_20260820/rank2_g1_geometry_grounded_proposal.py',
    'research/n1d/rank2_global_relational_world_20260820/rank2_r_v5_full_pairwise_proposal.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/build_source_bundle.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/preflight_g2.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/run_pretruth_g2.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/src/rank2_g2_reciprocal_cycle_evidence.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/src/rank2_g2_reciprocal_cycle_proposal.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/src/rank2_g2_truth_evaluator.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/tests/test_g2_static.py',
    'research/n1d/observable_geometry_closure_20260821/g2_reciprocal_cycle_v1/tests/test_g2_evaluator_authority.py',
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def repo_root_from_self() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / 'research' / 'n1d').is_dir() and (parent / 'canonical').is_dir():
            return parent
    raise RuntimeError('repo root not found')


def build(repo_root: Path, out: Path) -> dict:
    missing = [x for x in MEMBERS if not (repo_root / x).is_file()]
    if missing:
        raise RuntimeError(('missing source members', missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in MEMBERS:
            data = (repo_root / rel).read_bytes()
            zi = zipfile.ZipInfo(rel, FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3
            zi.external_attr = MODE
            z.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {
        'schema': 'RealSaS.N1D.Rank2G.G2.SourceBundleBuild.v1',
        'bundle': out.name,
        'bundle_sha256': sha256_file(out),
        'bundle_size_bytes': out.stat().st_size,
        'members': {rel: sha256_file(repo_root / rel) for rel in MEMBERS},
        'zip_contract': {'timestamp': list(FIXED_TIME), 'compression': 'ZIP_DEFLATED', 'compresslevel': 9, 'unix_mode': '100644'},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, default=None)
    a = ap.parse_args()
    root = a.repo_root.resolve() if a.repo_root else repo_root_from_self()
    r = build(root, a.out.resolve())
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(r, indent=2, sort_keys=True) + '\n')
    print(json.dumps(r, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
