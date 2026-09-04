from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FAMILY_NAME = "kaykit_01_Mage"
ASSET_ID = "asset_fbc8d57f848df78bd953fbb5"
EXPECTED_GEOMETRY_SHA256 = "f1e429a706faffd6ed6fd2b26a756cfca406e4ece440a9717e7e4b09f03db07d"
MASTER_BASENAME = "RealSaS_MASTER_CORPUS_1024_V3"


@dataclass
class ProbeReport:
    schema: str
    family_name: str
    asset_id: str
    python: str
    platform: str
    torch_importable: bool
    torch_version: str | None
    cuda_available: bool
    cuda_device_name: str | None
    cuda_total_memory_bytes: int | None
    master_root: str | None
    asset_root: str | None
    primary_geometry_path: str | None
    primary_geometry_sha256: str | None
    primary_geometry_sha_match: bool | None
    primary_geometry_keys: list[str]
    full_truth_fields_present: bool | None
    master_views_present: int | None
    source_textured_views_present: int | None
    current_model_imports_pass: bool
    notes: list[str]
    status: str


def _sha256_path(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _torch_probe() -> tuple[bool, str | None, bool, str | None, int | None, list[str]]:
    notes: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        return False, None, False, None, None, [f"TORCH_IMPORT_FAIL:{type(exc).__name__}:{exc}"]
    cuda = bool(torch.cuda.is_available())
    name = None
    total = None
    if cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            name = str(props.name)
            total = int(props.total_memory)
        except Exception as exc:  # pragma: no cover
            notes.append(f"CUDA_PROPERTY_FAIL:{type(exc).__name__}:{exc}")
    return True, str(torch.__version__), cuda, name, total, notes


def _candidate_master_roots() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("REALSAS_MASTER_ROOT")
    if env:
        out.append(Path(env).expanduser())
    home = Path.home()
    out.extend([
        home / MASTER_BASENAME,
        home / "data" / MASTER_BASENAME,
        home / "Downloads" / MASTER_BASENAME,
        Path("/content/drive/MyDrive") / MASTER_BASENAME,
    ])
    users = Path("/mnt/c/Users")
    if users.is_dir():
        for u in sorted(users.iterdir()):
            if not u.is_dir():
                continue
            out.extend([
                u / "Downloads" / MASTER_BASENAME,
                u / "Desktop" / MASTER_BASENAME,
                u / "Documents" / MASTER_BASENAME,
                u / "My Drive" / MASTER_BASENAME,
                u / "Google Drive" / "My Drive" / MASTER_BASENAME,
            ])
    # Stable de-duplication without recursively crawling user disks.
    seen: set[str] = set()
    dedup: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            dedup.append(p)
    return dedup


def _find_master() -> Path | None:
    for p in _candidate_master_roots():
        if (p / "master" / "assets" / ASSET_ID / "primary_geometry.npz").is_file():
            return p.resolve()
    return None


def _count_master_views(asset_root: Path) -> int:
    # Historical master layouts use V0..V7 folders. Count only folders carrying both
    # camera and raster authority; do not guess a visual style.
    count = 0
    for v in range(8):
        d = asset_root / f"V{v}"
        if (d / "camera.json").is_file() and (d / "raster_authority.npz").is_file():
            count += 1
    return count


def _count_source_textured_views(asset_root: Path) -> int:
    names = ("source_textured_rgba.png", "source_textured.png", "textured.png")
    count = 0
    for v in range(8):
        d = asset_root / f"V{v}"
        if any((d / name).is_file() for name in names):
            count += 1
    return count


def _probe_geometry(path: Path) -> tuple[list[str], bool, list[str]]:
    notes: list[str] = []
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(map(str, z.files))
    except Exception as exc:
        return [], False, [f"NPZ_OPEN_FAIL:{type(exc).__name__}:{exc}"]
    geppetto_any = {"bone_heads", "bone_tails"}.issubset(keys) and ("parents" in keys or "bone_parents" in keys)
    arachne = "skin" in keys
    iris = any(k in keys for k in ("vertices", "verts", "positions")) and any(k in keys for k in ("faces", "triangles"))
    full = bool(iris and geppetto_any and arachne)
    if not full:
        notes.append("FULL_TRUTH_FIELDS_NOT_ALL_PRESENT")
    return keys, full, notes


def _probe_current_imports() -> tuple[bool, list[str]]:
    notes: list[str] = []
    try:
        from models.iris.v2.q_domain_v2 import build_production_observation_ray_lattice_v2  # noqa:F401
        from models.iris.v2.observation_contract_v2 import ObservationContractV2  # noqa:F401
        from compiler.realsas_compiler_core.substrate.iris_v2 import compile_surface_v2, attach_dtb_nd1_from_evidence  # noqa:F401
        from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateV2  # noqa:F401
        from compiler.realsas_compiler_core.rig import qualify_skeleton_v2  # noqa:F401
        from models.skin_field_codec.v1.model_v1 import SkinFieldCodecV1  # noqa:F401
        from models.arachne.v2.arachne_candidate_v2 import ArachneCandidateV2  # noqa:F401
        from compiler.realsas_compiler_core.skin import qualify_skin  # noqa:F401
        from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh  # noqa:F401
        from compiler.realsas_compiler_core.mwb2_skin import bind_mwb2_mesh_skin  # noqa:F401
    except Exception as exc:
        notes.append(f"CURRENT_IMPORT_FAIL:{type(exc).__name__}:{exc}")
        return False, notes
    return True, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="first_family_local_preflight.json")
    args = ap.parse_args()

    torch_ok, torch_ver, cuda, cuda_name, cuda_mem, notes = _torch_probe()
    imports_ok, import_notes = _probe_current_imports()
    notes.extend(import_notes)

    master = _find_master()
    asset_root = master / "master" / "assets" / ASSET_ID if master else None
    geom = asset_root / "primary_geometry.npz" if asset_root else None
    geom_sha = _sha256_path(geom) if geom and geom.is_file() else None
    geom_match = (geom_sha == EXPECTED_GEOMETRY_SHA256) if geom_sha else None
    keys: list[str] = []
    full_truth: bool | None = None
    master_views: int | None = None
    textured_views: int | None = None
    if geom and geom.is_file():
        keys, full_truth, geom_notes = _probe_geometry(geom)
        notes.extend(geom_notes)
        master_views = _count_master_views(asset_root)
        textured_views = _count_source_textured_views(asset_root)
        if geom_match is False:
            notes.append("MASTER_GEOMETRY_SHA_DRIFT")
    else:
        notes.append("DATA_NOT_LOCAL: exact Master Mage asset not found in bounded known roots")

    if cuda and cuda_mem is not None and cuda_mem < 8 * (1 << 30):
        notes.append("LOCAL_GPU_IS_PREFLIGHT_ONLY_FOR_NATIVE1024_IRIS; use A100-class GPU for scientific fit")

    if master is None:
        status = "BLOCKED_DATA_NOT_LOCAL"
    elif not imports_ok:
        status = "BLOCKED_CURRENT_IMPORTS"
    elif geom_match is not True or full_truth is not True or master_views != 8:
        status = "BLOCKED_MASTER_AUTHORITY"
    else:
        status = "PASS_LOCAL_DATA_AND_CODE_PREFLIGHT"

    report = ProbeReport(
        schema="RealSaS.FirstFamilyLocalPreflight.v1",
        family_name=FAMILY_NAME,
        asset_id=ASSET_ID,
        python=platform.python_version(),
        platform=platform.platform(),
        torch_importable=torch_ok,
        torch_version=torch_ver,
        cuda_available=cuda,
        cuda_device_name=cuda_name,
        cuda_total_memory_bytes=cuda_mem,
        master_root=str(master) if master else None,
        asset_root=str(asset_root) if asset_root else None,
        primary_geometry_path=str(geom) if geom else None,
        primary_geometry_sha256=geom_sha,
        primary_geometry_sha_match=geom_match,
        primary_geometry_keys=keys,
        full_truth_fields_present=full_truth,
        master_views_present=master_views,
        source_textured_views_present=textured_views,
        current_model_imports_pass=imports_ok,
        notes=notes,
        status=status,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
