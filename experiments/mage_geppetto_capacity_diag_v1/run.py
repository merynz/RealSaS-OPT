from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mage_scene_first_e2e_v1 import run_mage_geppetto_fit_v1 as mage
from experiments.mage_scene_first_e2e_v1.run_mage_geppetto_fit_v2q import _load_fixture_v2q
from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateV2
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from models.geppetto.v2.geppetto_eval_v2 import geppetto_metrics_v2
from models.geppetto.v2.geppetto_loss_v2 import canonical_geometry_assignment_v2, _first_hit_stop_loss


SEED = 20260905
LR = 3e-4
WEIGHT_DECAY = 1e-4
LADDER = (4, 8, 16, 24, 31)
LADDER_STEPS = 256
COUNT_STEPS = 256
AUTHORED_STEPS = 512
CHECK_EVERY = 32


def _tensors(conditioning, device):
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    return f, p, m


def _positive_unique_radius(points: np.ndarray) -> float:
    p = torch.as_tensor(points, dtype=torch.float32)
    d = torch.cdist(p, p).numpy()
    d[np.eye(len(d), dtype=bool)] = np.inf
    d[d <= 1e-5] = np.inf
    finite = d[np.isfinite(d)]
    if not len(finite):
        raise ValueError("no positive distinct target separation")
    return 0.5 * float(finite.min())


def _lex_key(row: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(x) for x in row.tolist())


def _unique_loci(points: np.ndarray) -> np.ndarray:
    by_key: dict[tuple[float, float, float], np.ndarray] = {}
    for row in np.asarray(points, np.float32):
        by_key.setdefault(_lex_key(row), row.copy())
    rows = [by_key[k] for k in sorted(by_key)]
    return np.stack(rows, axis=0).astype(np.float32)


def _farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if count > len(p):
        raise ValueError("subset exceeds target cardinality")
    order = sorted(range(len(p)), key=lambda i: _lex_key(p[i]))
    selected = [order[0]]
    min_d2 = np.sum((p - p[selected[0]][None]) ** 2, axis=1)
    min_d2[selected[0]] = -1.0
    while len(selected) < count:
        best = float(np.max(min_d2))
        ties = np.flatnonzero(np.isclose(min_d2, best, rtol=0.0, atol=1e-15)).tolist()
        nxt = min(ties, key=lambda i: _lex_key(p[i]))
        selected.append(nxt)
        d2 = np.sum((p - p[nxt][None]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected] = -1.0
    chosen = p[np.asarray(selected, np.int64)]
    return chosen.astype(np.float32)


def _geometry_terms(out, target_np: np.ndarray, j: int):
    target = torch.as_tensor(target_np, device=out.positions_normalized.device, dtype=out.positions_normalized.dtype)
    primary = out.positions_normalized[0, :j]
    q_np, t_np = canonical_geometry_assignment_v2(primary, target)
    q = torch.as_tensor(q_np, device=primary.device, dtype=torch.long)
    t = torch.as_tensor(t_np, device=primary.device, dtype=torch.long)
    tp = target[t]
    modes = out.position_modes_normalized[0, q]
    dist2 = (modes.detach() - tp[:, None, :]).square().sum(dim=-1)
    winner = torch.argsort(dist2, dim=-1, stable=True)[:, 0]
    gather = winner[:, None, None].expand(j, 1, 3)
    winner_positions = torch.gather(modes, 1, gather).squeeze(1)
    primary_loss = F.smooth_l1_loss(primary[q], tp, reduction="mean")
    winner_loss = F.smooth_l1_loss(winner_positions, tp, reduction="mean")
    mode_rank = F.cross_entropy(out.position_mode_logits[0, q], winner)
    total = 2.5 * primary_loss + 2.5 * winner_loss + mode_rank
    return total, {
        "position_primary": primary_loss,
        "position_winner": winner_loss,
        "mode_rank": mode_rank,
    }


def _direct_metrics(out, target_np: np.ndarray, j: int) -> dict[str, float]:
    return {k: float(v) for k, v in geppetto_metrics_v2(out.positions_normalized[0, :j], target_np).items()}


def _fresh_model(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = GeppettoCandidateV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def _geometry_ladder_rung(conditioning, target_np: np.ndarray, steps: int, device) -> dict[str, object]:
    j = int(len(target_np))
    radius = _positive_unique_radius(target_np)
    model, optimizer = _fresh_model(device)
    f, p, m = _tensors(conditioning, device)
    trace = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(f, p, m, decode_steps=j)
        loss, parts = _geometry_terms(out, target_np, j)
        loss.backward()
        optimizer.step()
        if step != 1 and step % CHECK_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            check = model(f, p, m, decode_steps=j)
            metrics = _direct_metrics(check, target_np, j)
        row = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_parts": {k: float(v.detach().cpu()) for k, v in parts.items()},
            "matched_mae": metrics["matched_mae"],
            "matched_p95": metrics["matched_p95"],
            "unique_radius": radius,
            "geometry_pass": bool(metrics["matched_p95"] < radius),
        }
        trace.append(row)
        print("MAGE_GEPPETTO_CAPACITY_LADDER=" + json.dumps({"j": j, **row}, sort_keys=True), flush=True)
    final = trace[-1]
    return {
        "target_count": j,
        "max_steps": steps,
        "unique_radius": radius,
        "final": final,
        "best_p95": min(float(x["matched_p95"]) for x in trace),
        "trace": trace,
    }


def _count_only(conditioning, authored_j: int, device) -> dict[str, object]:
    j = int(authored_j)
    model, optimizer = _fresh_model(device)
    f, p, m = _tensors(conditioning, device)
    trace = []
    for step in range(1, COUNT_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(f, p, m, decode_steps=j + 1)
        ex_target = torch.zeros(j + 1, device=device, dtype=out.existence_logits.dtype)
        ex_target[:j] = 1.0
        existence = F.binary_cross_entropy_with_logits(out.existence_logits[0], ex_target)
        stop = _first_hit_stop_loss(out.stop_logits[0, :j])
        loss = existence + stop
        loss.backward()
        optimizer.step()
        if step != 1 and step % CHECK_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            gout, counts = model.generate(f, p, m, resource_step_limit=128)
            stop_prob = torch.sigmoid(gout.stop_logits[0, :j]).detach().cpu()
            count = int(counts[0])
            terminal = float(stop_prob[-1])
            early_max = float(stop_prob[:-1].max()) if j > 1 else 0.0
        row = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "existence": float(existence.detach().cpu()),
            "stop": float(stop.detach().cpu()),
            "generated_count": count,
            "terminal_stop_probability": terminal,
            "max_early_stop_probability": early_max,
            "count_pass": count == j,
        }
        trace.append(row)
        print("MAGE_GEPPETTO_COUNT_ONLY=" + json.dumps(row, sort_keys=True), flush=True)
    return {
        "target_count": j,
        "max_steps": COUNT_STEPS,
        "final": trace[-1],
        "ever_count_pass": any(bool(x["count_pass"]) for x in trace),
        "trace": trace,
    }


def _authored_geometry_count(conditioning, target_np: np.ndarray, device) -> dict[str, object]:
    j = int(len(target_np))
    radius = _positive_unique_radius(target_np)
    model, optimizer = _fresh_model(device)
    f, p, m = _tensors(conditioning, device)
    trace = []
    for step in range(1, AUTHORED_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(f, p, m, decode_steps=j + 1)
        geometry, parts = _geometry_terms(out, target_np, j)
        ex_target = torch.zeros(j + 1, device=device, dtype=out.existence_logits.dtype)
        ex_target[:j] = 1.0
        existence = F.binary_cross_entropy_with_logits(out.existence_logits[0], ex_target)
        stop = _first_hit_stop_loss(out.stop_logits[0, :j])
        loss = geometry + existence + stop
        loss.backward()
        optimizer.step()
        if step != 1 and step % CHECK_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            direct = model(f, p, m, decode_steps=j)
            direct_metrics = _direct_metrics(direct, target_np, j)
            gout, counts = model.generate(f, p, m, resource_step_limit=128)
            count = int(counts[0])
            if count == j:
                shipping_metrics = {k: float(v) for k, v in geppetto_metrics_v2(gout.positions_normalized[0, :j], target_np).items()}
            else:
                shipping_metrics = {"matched_mae": float("inf"), "matched_p95": float("inf")}
        row = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_parts": {
                **{k: float(v.detach().cpu()) for k, v in parts.items()},
                "existence": float(existence.detach().cpu()),
                "stop": float(stop.detach().cpu()),
            },
            "generated_count": count,
            "direct_matched_mae": direct_metrics["matched_mae"],
            "direct_matched_p95": direct_metrics["matched_p95"],
            "shipping_matched_p95": shipping_metrics["matched_p95"],
            "unique_radius": radius,
            "direct_geometry_pass": bool(direct_metrics["matched_p95"] < radius),
            "shipping_geometry_count_pass": bool(count == j and shipping_metrics["matched_p95"] < radius),
        }
        trace.append(row)
        print("MAGE_GEPPETTO_AUTHORED_GEOMETRY_COUNT=" + json.dumps(row, sort_keys=True), flush=True)
    return {
        "target_count": j,
        "max_steps": AUTHORED_STEPS,
        "unique_radius": radius,
        "final": trace[-1],
        "best_direct_p95": min(float(x["direct_matched_p95"]) for x in trace),
        "ever_direct_geometry_pass": any(bool(x["direct_geometry_pass"]) for x in trace),
        "ever_shipping_geometry_count_pass": any(bool(x["shipping_geometry_count_pass"]) for x in trace),
        "trace": trace,
    }


def run(output_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_MAGE_GEPPETTO_CAPACITY_DIAGNOSTIC")
    device = torch.device("cuda")

    fx = _load_fixture_v2q()
    surface = mage._surface(fx)
    conditioning = GeppettoConditioningAdapterV2()([surface])
    teacher = mage._teacher(fx, conditioning)
    authored = np.asarray(teacher.positions_normalized, np.float32)
    unique = _unique_loci(authored)
    if authored.shape != (41, 3):
        raise RuntimeError(f"Mage authored joint cardinality drift:{authored.shape}")
    if unique.shape != (31, 3):
        raise RuntimeError(f"Mage unique-locus cardinality drift:{unique.shape}")

    ladder = []
    for j in LADDER:
        subset = _farthest_subset(unique, j)
        ladder.append(_geometry_ladder_rung(conditioning, subset, LADDER_STEPS, device))

    count_only = _count_only(conditioning, 41, device)
    authored_geometry_count = _authored_geometry_count(conditioning, authored, device)

    result = {
        "schema": "RealSaS.MageGeppettoCapacityDiagnostic.v1",
        "status": "DIAGNOSTIC_ONLY",
        "promotion_authority": False,
        "generalization_claim": False,
        "source_fixture_sha256": "60b0b2788b7d37971dcd81886c165cd56eab3da767f9b555b08cc3f267ed1aa8",
        "surface_nodes": len(surface.surface_nodes),
        "surface_relations": len(surface.local_relations),
        "authored_joint_count": int(len(authored)),
        "unique_locus_count": int(len(unique)),
        "full_authored_unique_radius": _positive_unique_radius(authored),
        "ladder": ladder,
        "count_only": count_only,
        "authored_geometry_count_no_topology_support": authored_geometry_count,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MAGE_GEPPETTO_CAPACITY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/MAGE_GEPPETTO_CAPACITY_DIAGNOSTIC_V1.json")
    run(out)
