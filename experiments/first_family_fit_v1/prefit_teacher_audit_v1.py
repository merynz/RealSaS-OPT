from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.first_family_fit_v1.ray_surface_teacher_v2 import (
    build_exact_anchor_ray_surface_teacher_v2,
    teacher_depth_range_for_domain_v2,
)
from experiments.first_family_fit_v1.run_family_iris_fit_v1 import load_observation
from experiments.first_family_fit_v1.run_family_observation_v1 import load_family_manifest, stage_authority
from models.iris.v2.q_domain_v2 import build_production_observation_ray_lattice_v2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-manifest", required=True)
    ap.add_argument("--observation-root", required=True)
    authority = ap.add_mutually_exclusive_group(required=True)
    authority.add_argument("--master-root")
    authority.add_argument("--authority-zip")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    family_path = Path(args.family_manifest).resolve()
    family = load_family_manifest(family_path)
    observation_root = Path(args.observation_root).resolve()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    authority_root, authority_source = stage_authority(
        zip_path=Path(args.authority_zip).resolve() if args.authority_zip else None,
        master_root=Path(args.master_root).resolve() if args.master_root else None,
        work=work_root / "authority_stage",
        manifest=family,
    )

    _, contract, _ = load_observation(observation_root, family)
    iris = family.get("iris", {})
    stride = int(iris.get("anchor_stride_px", 16))
    depth_bins = int(iris.get("depth_bins", 48))
    model_max_modes = int(iris.get("max_modes", 3))
    depth_values = teacher_depth_range_for_domain_v2(
        authority_root=authority_root,
        contract=contract,
        depth_bins=depth_bins,
    )
    domain = build_production_observation_ray_lattice_v2(
        (contract,),
        anchor_view_index=0,
        anchor_stride_px=stride,
        depth_values=depth_values,
    )
    teacher = build_exact_anchor_ray_surface_teacher_v2(
        authority_root=authority_root,
        contract=contract,
        domain=domain,
    )

    mode_count = teacher.mode_count[0]
    supported = mode_count > 0
    exceeds = mode_count > model_max_modes
    payload = {
        "schema": "RealSaS.PrefitTeacherAudit.v1",
        "status": "PASS_AUDIT_COMPLETE",
        "asset_id": family["asset_id"],
        "family_label": family.get("family_label"),
        "authority_source": authority_source,
        "q_domain_authority": domain.construction_authority,
        "anchor_stride_px": stride,
        "depth_bins": depth_bins,
        "model_max_modes": model_max_modes,
        "teacher": teacher.telemetry,
        "supported_rays": int(supported.sum().item()),
        "rays_exceeding_model_max_modes": int(exceeds.sum().item()),
        "fraction_supported_rays_exceeding_model_max_modes": (
            float(exceeds[supported].float().mean().item()) if supported.any() else 0.0
        ),
        "fit_authorization": "BLOCKED_PENDING_REVIEW" if exceeds.any() else "TEACHER_MODE_CAPACITY_NOT_BLOCKING",
        "notes": [
            "Q rays are full-frame camera-only; teacher geometry never selects rays.",
            "Teacher retains all distinct canonical mesh intersections along each anchor ray.",
            "This audit does not authorize optimizer execution by itself.",
        ],
    }
    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
