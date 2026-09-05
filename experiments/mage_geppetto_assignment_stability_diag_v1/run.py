from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.mage_geppetto_capacity_diag_v1 import run as base
from experiments.mage_scene_first_e2e_v1 import run_mage_geppetto_fit_v1 as mage
from experiments.mage_scene_first_e2e_v1.run_mage_geppetto_fit_v2q import _load_fixture_v2q
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from models.geppetto.v2.geppetto_loss_v2 import canonical_geometry_assignment_v2


LADDER = (4, 8, 16, 24, 31)
LADDER_STEPS = 256
LONG_RUNS = {8: 2048, 16: 2048}
CHECK_EVERY_SHORT = 32
CHECK_EVERY_LONG = 64


def _assignment_signature(out, target_np: np.ndarray, j: int) -> tuple[int, ...]:
    primary = out.positions_normalized[0, :j]
    q_np, t_np = canonical_geometry_assignment_v2(primary, target_np)
    signature = np.full(j, -1, dtype=np.int64)
    signature[np.asarray(q_np, np.int64)] = np.asarray(t_np, np.int64)
    if (signature < 0).any():
        raise RuntimeError("incomplete Hungarian assignment signature")
    return tuple(int(x) for x in signature.tolist())


def _run_one(conditioning, target_np: np.ndarray, *, steps: int, check_every: int, device, label: str) -> dict[str, object]:
    j = int(len(target_np))
    radius = base._positive_unique_radius(target_np)
    model, optimizer = base._fresh_model(device)
    f, p, m = base._tensors(conditioning, device)

    previous_signature: tuple[int, ...] | None = None
    unique_signatures: set[tuple[int, ...]] = set()
    flip_events = 0
    hamming_total = 0
    max_hamming = 0
    first_flip_step: int | None = None
    last_flip_step: int | None = None
    trace: list[dict[str, object]] = []
    best_p95 = float("inf")
    first_observed_pass_step: int | None = None

    for step in range(1, int(steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(f, p, m, decode_steps=j)

        signature = _assignment_signature(out, target_np, j)
        unique_signatures.add(signature)
        if previous_signature is not None:
            hamming = sum(int(a != b) for a, b in zip(previous_signature, signature))
            if hamming:
                flip_events += 1
                hamming_total += hamming
                max_hamming = max(max_hamming, hamming)
                if first_flip_step is None:
                    first_flip_step = step
                last_flip_step = step
        previous_signature = signature

        loss, parts = base._geometry_terms(out, target_np, j)
        loss.backward()
        optimizer.step()

        if step != 1 and step % int(check_every):
            continue

        model.eval()
        with torch.no_grad():
            check = model(f, p, m, decode_steps=j)
            metrics = base._direct_metrics(check, target_np, j)
        p95 = float(metrics["matched_p95"])
        best_p95 = min(best_p95, p95)
        geometry_pass = bool(p95 < radius)
        if geometry_pass and first_observed_pass_step is None:
            first_observed_pass_step = step
        row = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_parts": {k: float(v.detach().cpu()) for k, v in parts.items()},
            "matched_mae": float(metrics["matched_mae"]),
            "matched_p95": p95,
            "unique_radius": radius,
            "geometry_pass": geometry_pass,
            "flip_events_so_far": flip_events,
            "assignment_hamming_total_so_far": hamming_total,
            "unique_assignment_states_so_far": len(unique_signatures),
        }
        trace.append(row)
        print(
            "MAGE_GEPPETTO_ASSIGNMENT_STABILITY="
            + json.dumps({"label": label, "j": j, **row}, sort_keys=True),
            flush=True,
        )

    result = {
        "label": label,
        "target_count": j,
        "steps": int(steps),
        "check_every": int(check_every),
        "unique_radius": radius,
        "best_p95": best_p95,
        "ever_observed_geometry_pass": first_observed_pass_step is not None,
        "first_observed_pass_step": first_observed_pass_step,
        "assignment": {
            "flip_events": flip_events,
            "flip_event_rate": float(flip_events / max(steps - 1, 1)),
            "hamming_total": hamming_total,
            "hamming_per_transition": float(hamming_total / max(steps - 1, 1)),
            "max_hamming_single_transition": max_hamming,
            "unique_assignment_states": len(unique_signatures),
            "first_flip_step": first_flip_step,
            "last_flip_step": last_flip_step,
        },
        "final": trace[-1],
        "trace": trace,
    }
    print("MAGE_GEPPETTO_ASSIGNMENT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def run(output_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_MAGE_GEPPETTO_ASSIGNMENT_STABILITY_DIAGNOSTIC")
    device = torch.device("cuda")

    fx = _load_fixture_v2q()
    surface = mage._surface(fx)
    conditioning = GeppettoConditioningAdapterV2()([surface])
    teacher = mage._teacher(fx, conditioning)
    authored = np.asarray(teacher.positions_normalized, np.float32)
    unique = base._unique_loci(authored)
    if authored.shape != (41, 3):
        raise RuntimeError(f"Mage authored joint cardinality drift:{authored.shape}")
    if unique.shape != (31, 3):
        raise RuntimeError(f"Mage unique-locus cardinality drift:{unique.shape}")

    ladder = []
    subsets: dict[int, np.ndarray] = {}
    for j in LADDER:
        subset = base._farthest_subset(unique, j)
        subsets[j] = subset
        ladder.append(
            _run_one(
                conditioning,
                subset,
                steps=LADDER_STEPS,
                check_every=CHECK_EVERY_SHORT,
                device=device,
                label=f"ladder_{j}",
            )
        )

    long_runs = []
    for j, steps in LONG_RUNS.items():
        long_runs.append(
            _run_one(
                conditioning,
                subsets[j],
                steps=steps,
                check_every=CHECK_EVERY_LONG,
                device=device,
                label=f"long_{j}_{steps}",
            )
        )

    result = {
        "schema": "RealSaS.MageGeppettoAssignmentStabilityDiagnostic.v1",
        "status": "DIAGNOSTIC_ONLY",
        "promotion_authority": False,
        "generalization_claim": False,
        "source_fixture_sha256": "60b0b2788b7d37971dcd81886c165cd56eab3da767f9b555b08cc3f267ed1aa8",
        "surface_nodes": len(surface.surface_nodes),
        "surface_relations": len(surface.local_relations),
        "authored_joint_count": int(len(authored)),
        "unique_locus_count": int(len(unique)),
        "ladder": ladder,
        "long_runs": long_runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MAGE_GEPPETTO_ASSIGNMENT_STABILITY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/MAGE_GEPPETTO_ASSIGNMENT_STABILITY_DIAGNOSTIC_V1.json")
    run(out)
