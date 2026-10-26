#!/usr/bin/env python3
"""Execute SYNTHETIC_TESSA_FUSION_PLACEMENT_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.tessa.v2.model_v2 import _TypedMemoryStreamV2
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSATriStreamMemoryV3

SCHEMA = "RealSaS.TESSAV3FusionPlacementCourt.v1"
SEEDS = (731, 947, 1217, 1597, 2017)
TRAIN_STEPS = 1500
BATCH_SIZE = 128
EVAL_COUNT = 4096
LR = 1.0e-3
SCENE_MEMORY_TOKENS = 4
SURFACE_MEMORY_TOKENS = 6
PRESENTATION_MEMORY_TOKENS = 2


def tiny_config() -> TESSAStructuralizerConfigV3:
    return TESSAStructuralizerConfigV3(
        d_model=64,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=SCENE_MEMORY_TOKENS,
        surface_memory_tokens=SURFACE_MEMORY_TOKENS,
        presentation_memory_tokens=PRESENTATION_MEMORY_TOKENS,
        conditioning_token_budget=12,
        memory_source_width=32,
        memory_refinement_layers=1,
        local_attention_window=64,
        query_chunk_size=16,
        gradient_checkpointing=False,
    )


class TwoHopReader(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) / math.sqrt(d_model))
        self.presentation_read = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.surface_read = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.presentation_norm = nn.LayerNorm(d_model)
        self.surface_norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, presentation_memory: torch.Tensor, surface_memory: torch.Tensor) -> torch.Tensor:
        batch = presentation_memory.shape[0]
        q = self.query[None].expand(batch, -1, -1)
        p, _ = self.presentation_read(q, presentation_memory, presentation_memory, need_weights=False)
        q = self.presentation_norm(q + p)
        s, _ = self.surface_read(q, surface_memory, surface_memory, need_weights=False)
        q = self.surface_norm(q + s)
        return self.out(q[:, 0]).squeeze(-1)


class CurrentLateIndependentModel(nn.Module):
    def __init__(self, cfg: TESSAStructuralizerConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        self.memory = TESSATriStreamMemoryV3(
            scene_input_dim=3,
            surface_input_dim=3,
            presentation_input_dim=3,
            config=cfg,
        )
        self.reader = TwoHopReader(cfg.d_model, cfg.n_heads)

    def forward(self, scene: torch.Tensor, surface: torch.Tensor, presentation: torch.Tensor) -> torch.Tensor:
        memory = self.memory(
            scene_source=scene,
            surface_source=surface,
            presentation_source=presentation,
        )
        scene_end = self.cfg.scene_memory_tokens
        surface_end = scene_end + self.cfg.surface_memory_tokens
        surface_memory = memory[:, scene_end:surface_end]
        presentation_memory = memory[:, surface_end:]
        return self.reader(presentation_memory, surface_memory)


class JointTypedSPMemory(nn.Module):
    """Joint surface+presentation compression with typed sources and typed output queries."""

    def __init__(self, cfg: TESSAStructuralizerConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.memory_source_width
        d_model = cfg.d_model
        self.surface_projection = nn.Sequential(nn.Linear(3, width), nn.LayerNorm(width), nn.GELU())
        self.presentation_projection = nn.Sequential(nn.Linear(3, width), nn.LayerNorm(width), nn.GELU())
        self.surface_source_type = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.presentation_source_type = nn.Parameter(torch.randn(width) / math.sqrt(width))
        total = cfg.surface_memory_tokens + cfg.presentation_memory_tokens
        self.queries = nn.Parameter(torch.randn(total, d_model) / math.sqrt(d_model))
        self.surface_query_type = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.presentation_query_type = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.query_norm = nn.LayerNorm(d_model)
        self.source_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            kdim=width,
            vdim=width,
            batch_first=True,
        )
        if cfg.memory_refinement_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.feedforward_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.refinement = nn.TransformerEncoder(layer, num_layers=cfg.memory_refinement_layers)
        else:
            self.refinement = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, surface: torch.Tensor, presentation: torch.Tensor) -> torch.Tensor:
        s = self.surface_projection(surface) + self.surface_source_type[None, None, :]
        p = self.presentation_projection(presentation) + self.presentation_source_type[None, None, :]
        source = torch.cat((s, p), dim=1)
        batch = source.shape[0]
        q = self.queries.unsqueeze(0).expand(batch, -1, -1).clone()
        split = self.cfg.surface_memory_tokens
        q[:, :split] = q[:, :split] + self.surface_query_type[None, None, :]
        q[:, split:] = q[:, split:] + self.presentation_query_type[None, None, :]
        update, _ = self.cross(
            query=self.query_norm(q),
            key=self.source_norm(source),
            value=self.source_norm(source),
            need_weights=False,
        )
        memory = self.refinement(q + update)
        return self.out_norm(memory)


class JointTypedCompressionModel(nn.Module):
    def __init__(self, cfg: TESSAStructuralizerConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene = _TypedMemoryStreamV2(input_dim=3, token_count=cfg.scene_memory_tokens, type_index=0, config=cfg)
        self.joint = JointTypedSPMemory(cfg)
        self.fusion_norm = nn.LayerNorm(cfg.d_model)
        self.reader = TwoHopReader(cfg.d_model, cfg.n_heads)

    def forward(self, scene: torch.Tensor, surface: torch.Tensor, presentation: torch.Tensor) -> torch.Tensor:
        scene_memory = self.scene(scene)
        sp = self.joint(surface, presentation)
        memory = self.fusion_norm(torch.cat((scene_memory, sp), dim=1))
        scene_end = self.cfg.scene_memory_tokens
        surface_end = scene_end + self.cfg.surface_memory_tokens
        surface_memory = memory[:, scene_end:surface_end]
        presentation_memory = memory[:, surface_end:]
        return self.reader(presentation_memory, surface_memory)


class RawPositiveControlModel(nn.Module):
    def __init__(self, cfg: TESSAStructuralizerConfigV3) -> None:
        super().__init__()
        d_model = cfg.d_model
        self.surface = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.presentation = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.surface_type = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.presentation_type = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.reader = TwoHopReader(d_model, cfg.n_heads)

    def forward(self, scene: torch.Tensor, surface: torch.Tensor, presentation: torch.Tensor) -> torch.Tensor:
        del scene
        s = self.surface(surface) + self.surface_type[None, None, :]
        p = self.presentation(presentation) + self.presentation_type[None, None, :]
        return self.reader(p, s)


def lattice() -> torch.Tensor:
    axis = torch.linspace(-0.5, 0.5, 4, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(16, 2)


def input_batch(count: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = lattice()
    bits = torch.randint(0, 2, (count, 16), generator=generator, dtype=torch.int64)
    payload = bits.to(torch.float32).mul(2.0).sub(1.0)
    permutation = torch.rand((count, 16), generator=generator).argsort(dim=1)
    target_index = torch.randint(0, 16, (count,), generator=generator, dtype=torch.int64)
    expanded_coords = coords[None].expand(count, -1, -1)
    unpermuted = torch.cat((expanded_coords, payload[..., None]), dim=-1)
    surface = torch.gather(unpermuted, 1, permutation[..., None].expand(-1, -1, 3))
    target_xy = coords[target_index]
    presentation = torch.cat((target_xy, torch.ones(count, 1)), dim=-1)[:, None, :]
    scene = torch.zeros(count, 4, 3, dtype=torch.float32)
    target = bits[torch.arange(count), target_index].to(torch.float32)
    return scene, surface, presentation, target


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval(*, seed: int, arm: str, model_template: nn.Module, random_label: bool = False) -> dict:
    model = copy.deepcopy(model_template)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    input_gen = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    label_gen = torch.Generator(device="cpu").manual_seed(seed + 300_000)
    last_loss = None
    for _ in range(TRAIN_STEPS):
        scene, surface, presentation, true_target = input_batch(BATCH_SIZE, generator=input_gen)
        if random_label:
            target = torch.randint(0, 2, (BATCH_SIZE,), generator=label_gen, dtype=torch.int64).float()
        else:
            target = true_target
        optimizer.zero_grad(set_to_none=True)
        logits = model(scene, surface, presentation)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    model.eval()
    input_gen = torch.Generator(device="cpu").manual_seed(seed + 200_000)
    label_gen = torch.Generator(device="cpu").manual_seed(seed + 400_000)
    total_correct = 0
    total_bce = 0.0
    total_pred_positive = 0
    total_target_positive = 0
    seen = 0
    with torch.no_grad():
        while seen < EVAL_COUNT:
            n = min(BATCH_SIZE, EVAL_COUNT - seen)
            scene, surface, presentation, true_target = input_batch(n, generator=input_gen)
            if random_label:
                target = torch.randint(0, 2, (n,), generator=label_gen, dtype=torch.int64).float()
            else:
                target = true_target
            logits = model(scene, surface, presentation)
            total_bce += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            pred = logits >= 0
            truth = target >= 0.5
            total_correct += int((pred == truth).sum())
            total_pred_positive += int(pred.sum())
            total_target_positive += int(truth.sum())
            seen += n
    return {
        "seed": seed,
        "arm": arm,
        "train_final_bce": last_loss,
        "eval_accuracy": total_correct / float(EVAL_COUNT),
        "eval_bce": total_bce / float(EVAL_COUNT),
        "predicted_positive_fraction": total_pred_positive / float(EVAL_COUNT),
        "target_positive_fraction": total_target_positive / float(EVAL_COUNT),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    vals = [r["eval_accuracy"] for r in rows if r["arm"] == arm]
    return {
        "mean_accuracy": sum(vals) / len(vals),
        "min_accuracy": min(vals),
        "std_accuracy_population": statistics.pstdev(vals),
        "per_seed_accuracy": vals,
    }


def run() -> dict:
    torch.set_num_threads(1)
    cfg = tiny_config()
    cfg.validate()
    rows: list[dict] = []
    param_counts: dict[str, int] = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        current = CurrentLateIndependentModel(cfg)
        torch.manual_seed(seed)
        joint = JointTypedCompressionModel(cfg)
        torch.manual_seed(seed)
        raw = RawPositiveControlModel(cfg)
        if not param_counts:
            param_counts = {
                "CURRENT_LATE_INDEPENDENT": parameter_count(current),
                "JOINT_TYPED_COMPRESSION": parameter_count(joint),
                "RAW_TWO_HOP_POSITIVE_CONTROL": parameter_count(raw),
            }
        rows.append(train_eval(seed=seed, arm="CURRENT_LATE_INDEPENDENT", model_template=current))
        rows.append(train_eval(seed=seed, arm="JOINT_TYPED_COMPRESSION", model_template=joint))
        rows.append(train_eval(seed=seed, arm="RAW_TWO_HOP_POSITIVE_CONTROL", model_template=raw))
        rows.append(train_eval(seed=seed, arm="RANDOM_LABEL_FIREWALL", model_template=joint, random_label=True))

    current_s = summarize(rows, "CURRENT_LATE_INDEPENDENT")
    joint_s = summarize(rows, "JOINT_TYPED_COMPRESSION")
    raw_s = summarize(rows, "RAW_TWO_HOP_POSITIVE_CONTROL")
    firewall_s = summarize(rows, "RANDOM_LABEL_FIREWALL")
    by_arm_seed = {(r["arm"], r["seed"]): r for r in rows}
    paired_gain = [
        by_arm_seed[("JOINT_TYPED_COMPRESSION", s)]["eval_accuracy"]
        - by_arm_seed[("CURRENT_LATE_INDEPENDENT", s)]["eval_accuracy"]
        for s in SEEDS
    ]
    mean_gain = sum(paired_gain) / len(paired_gain)

    firewall_ok = all(0.45 <= by_arm_seed[("RANDOM_LABEL_FIREWALL", s)]["eval_accuracy"] <= 0.55 for s in SEEDS)
    raw_ok = raw_s["min_accuracy"] >= 0.90 and raw_s["mean_accuracy"] >= 0.95
    joint_accessible = joint_s["min_accuracy"] >= 0.90 and joint_s["mean_accuracy"] >= 0.95
    gain_positive_count = sum(g > 0 for g in paired_gain)
    gain_ok = gain_positive_count >= 4 and mean_gain >= 0.10
    capacity_ok = param_counts["JOINT_TYPED_COMPRESSION"] <= param_counts["CURRENT_LATE_INDEPENDENT"]

    if not firewall_ok:
        status = "FUSION_COURT_FIREWALL_INVALID"
    elif not raw_ok:
        status = "FUSION_COURT_READER_OR_TASK_INVALID"
    elif joint_accessible and gain_ok and capacity_ok:
        status = "JOINT_TYPED_COMPRESSION_SUPERIOR_TINY_V1"
    elif joint_accessible and gain_ok and not capacity_ok:
        status = "JOINT_FUSION_PROMISING_CAPACITY_CONFOUNDED"
    elif mean_gain > 0:
        status = "FUSION_PLACEMENT_SIGNAL_INCONCLUSIVE_TINY_V1"
    else:
        status = "JOINT_COMPRESSION_NOT_SUPPORTED_TINY_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "preregistration": "SYNTHETIC_TESSA_FUSION_PLACEMENT_PREREG_V1.md",
        "config": asdict(cfg),
        "seeds": list(SEEDS),
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "evaluation_count": EVAL_COUNT,
        "optimizer": {"name": "AdamW", "lr": LR, "betas": [0.9, 0.999], "weight_decay": 0.0},
        "parameter_counts": param_counts,
        "results": rows,
        "summary": {
            "current_late_independent": current_s,
            "joint_typed_compression": joint_s,
            "raw_positive_control": raw_s,
            "random_label_firewall": firewall_s,
            "paired_joint_minus_current_gain": paired_gain,
            "mean_paired_gain": mean_gain,
            "positive_gain_seed_count": gain_positive_count,
        },
        "gates": {
            "firewall_ok": firewall_ok,
            "raw_positive_control_ok": raw_ok,
            "joint_accessibility_ok": joint_accessible,
            "paired_gain_ok": gain_ok,
            "joint_parameter_count_le_current": capacity_ok,
        },
        "authority": {
            "reference_tessa_pass_claimed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "product_authority_changed": False,
            "architecture_replacement_authorized": False,
        },
        "execution": {"device": "cpu", "gpu_required": False, "product_inference_executed": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "mean_current": report["summary"]["current_late_independent"]["mean_accuracy"],
        "mean_joint": report["summary"]["joint_typed_compression"]["mean_accuracy"],
        "mean_gain": report["summary"]["mean_paired_gain"],
        "parameter_counts": report["parameter_counts"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
