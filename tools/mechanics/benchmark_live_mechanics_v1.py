from __future__ import annotations

"""CPU-only executable mechanics frame benchmark, NOT a 60 FPS certification.

Exercise the actual canonical multi-goal IK, contact-pin residuals, Motion edit
override, sparse LBS and full-vertex behavioral diagnostics. Uses only a
seeded synthetic branching rig. The Stage47 teacher is never a runtime input.

Timing is per live Mechanics evaluation only. Renderer, input driver, GPU
uploads, raster/Graphics and whole-product scheduling are NOT included.
"""

import argparse
from dataclasses import replace
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshVertexIR, canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import CanonicalSkinRowIRV2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.mechanics_prepared_skin_v1 import (
    prepare_mechanics_skin_deformer_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    evaluate_mechanics_intervention_frame_v1,
)
from tests.compiler.test_mechanics_live_multigoal_integration_v1 import _objects
from tests.compiler.test_mechanics_control_solver_v1 import _support


SCHEMA = "RealSaS.MechanicsFullFrameCPUResearchCourt.v1"


def fixture(vertex_count: int, seed: int):
    if not 5 <= vertex_count <= 50000:
        raise ValueError("BENCH_MECHANICS_VERTEX_POLICY")
    mesh,rig,skin,mechanics=_objects()
    rng=np.random.default_rng(seed)
    xyz=rng.normal(size=(vertex_count,3))*np.array((.15,.22,.14))
    xyz[:,0]+=np.where(np.arange(vertex_count)%2==0,1.5,-1.5)
    vertices=tuple(CanonicalMeshVertexIR(
        canonical_mesh_vertex_id=f"BENCH:{i}",
        support_binding=_support(i),component_id="SYNTHETIC_BRANCHED_CARRIER",
        P=tuple(map(float,xyz[i])),source_candidate_vertex_id=f"BENCH:SOURCE:{i}",
    ) for i in range(vertex_count))
    mesh=replace(mesh,vertices=vertices,faces=(),edges=(),mesh_lineage_hash="")
    mesh=replace(mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig=replace(rig,mesh_lineage_hash=mesh.mesh_lineage_hash)
    fractions=rng.beta(2.,2.,size=vertex_count)
    rows=tuple(CanonicalSkinRowIRV2(
        f"BENCH:{i}",
        ((("A1",float(fractions[i])),("A2",float(1.-fractions[i])))
         if i%2==0 else
         (("B1",float(fractions[i])),("B2",float(1.-fractions[i])))),
        f"BENCH:SOURCE:{i}",
    ) for i in range(vertex_count))
    skin=replace(skin,rows=rows,mesh_lineage_hash=mesh.mesh_lineage_hash)
    mechanics=replace(mechanics,mesh_lineage_hash=mesh.mesh_lineage_hash)
    return mesh,rig,skin,mechanics


def _summary(xs: list[float]) -> dict[str,float]:
    x=np.asarray(xs,dtype=np.float64)
    return {
        "min_ms":float(np.min(x)),
        "p50_ms":float(np.percentile(x,50)),
        "p95_ms":float(np.percentile(x,95)),
        "p99_ms":float(np.percentile(x,99)),
        "max_ms":float(np.max(x)),
    }


def measure(*, vertices: int, frames: int, seed: int=20261106, warmup: int=3) -> dict:
    if not 3 <= frames <= 100 or not 0 <= warmup <= 10:
        raise ValueError("BENCH_MECHANICS_FRAME_POLICY")
    mesh,rig,skin,mechanics=fixture(vertices,seed)
    start=time.perf_counter_ns()
    prepared=prepare_mechanics_skin_deformer_v1(mesh,rig,skin)
    prep_ms=(time.perf_counter_ns()-start)/1e6
    prepared_times=[]
    legacy_times=[]
    parity_errors=[]
    observed_residual=[]
    observed_strain=[]
    failed=[]
    prev=None
    warm_starts=0
    for i in range(frames+warmup):
        angle=0.08+0.18*(0.5+0.5*math.sin(2.*math.pi*i/max(frames,4)))
        transform=ControlTransformIRV1(
            "A_HANDLE",translation=(math.cos(angle)-1.,math.sin(angle),0.),
        )
        args=dict(
            overrides=(transform,),
            solver_mode="DECLARED_MULTI_GOAL_V1",
            pinned_contact_control_ids=("B_HANDLE",),
            max_allowed_constraint_residual=1e-3,
            initial_parameters_rad=prev,
        )
        t=time.perf_counter_ns()
        fr=evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,prepared_skin=prepared,**args,
        )
        elapsed=(time.perf_counter_ns()-t)/1e6
        ok=fr.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
        if i>=warmup:
            prepared_times.append(elapsed)
            observed_residual.append(fr.max_constraint_residual)
            observed_strain.append(fr.max_segment_length_relative_error)
            if not ok: failed.append({"frame":i-warmup,"reason":list(fr.failures)})
            warm_starts+=int(fr.metadata["warm_start_used"])
            if (i-warmup)%4==0:
                t=time.perf_counter_ns()
                legacy=evaluate_mechanics_intervention_frame_v1(
                    mesh,rig,skin,mechanics,prepared_skin=None,**args,
                )
                legacy_times.append((time.perf_counter_ns()-t)/1e6)
                delta=float(np.max(np.linalg.norm(
                    np.asarray(legacy.deformation.positions)
                    -np.asarray(fr.deformation.positions),axis=1,
                )))
                parity_errors.append(delta)
                if legacy.status!=fr.status or delta>1e-8:
                    raise AssertionError("MECHANICS_CPU_PARITY_DRIFT")
        prev=fr.joint_parameter_solution_rad if ok else None
    return {
        "schema":SCHEMA,
        "status":"PASS_FUNCTIONAL_CPU_COURT" if not failed else "FAIL_FUNCTIONAL_CPU_COURT",
        "research_only":True,
        "fixture":"SEEDED_5_JOINT_3_DOF_TWO_CHAIN_TWO_IK_SYNTHETIC",
        "vertex_count":vertices,
        "skin_influence_count":prepared.influence_count,
        "joint_count":len(rig.joints),
        "pinned_contact_control_ids":["B_HANDLE"],
        "frames":frames,
        "warmup":warmup,
        "random_seed":seed,
        "prepared_cache_build_ms":prep_ms,
        "full_mechanics_prepared":_summary(prepared_times),
        "full_mechanics_legacy_sampled":_summary(legacy_times),
        "legacy_parity_sample_count":len(parity_errors),
        "max_prepared_vs_legacy_vertex_l2":max(parity_errors),
        "max_constraint_residual":max(observed_residual),
        "max_segment_strain":max(observed_strain),
        "warm_started_frames":warm_starts,
        "failed_frames":failed,
        "sampled_p95_meets_16_67ms_budget":_summary(prepared_times)["p95_ms"]<1000./60.,
        "timing_environment":{
            "python":platform.python_version(),
            "platform":platform.platform(),
            "numpy":np.__version__,"scipy":scipy.__version__,
            "scope":"SEQUENTIAL_CPU_ONLY_NO_RENDER_OR_GRAPHICS",
        },
        "authority":{
            "source_truth_or_teacher_used":False,
            "model_inference_used":False,
            "source_to_canonical_binding_proven":False,
            "graphics_or_render_included":False,
            "production_p95_certified":False,
            "training_authorized":False,
        },
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--vertices",type=int,default=3266)
    parser.add_argument("--frames",type=int,default=15)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261106)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=measure(vertices=args.vertices,frames=args.frames,
                   warmup=args.warmup,seed=args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
    if result["failed_frames"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
