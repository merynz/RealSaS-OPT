#!/usr/bin/env python3
"""Execute SYNTHETIC_TESSA_ADDRESSABILITY_PREREG_V1 exactly.

Tiny CPU learned probe only. No product inference, canonical IR, ModelRelease, or
scientific genericity claim is produced.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSATriStreamMemoryV3

SCHEMA = "RealSaS.TESSAV3SyntheticAddressabilityProbe.v1"
SEEDS = (731, 947, 1217)
TRAIN_STEPS = 1000
BATCH_SIZE = 128
EVAL_COUNT = 4096
LR = 1.0e-3


def config_v1() -> TESSAStructuralizerConfigV3:
    return TESSAStructuralizerConfigV3(
        d_model=64,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=4,
        surface_memory_tokens=6,
        presentation_memory_tokens=2,
        conditioning_token_budget=12,
        memory_source_width=32,
        memory_refinement_layers=1,
        local_attention_window=64,
        query_chunk_size=16,
        gradient_checkpointing=False,
    )


class AddressabilityModel(nn.Module):
    def __init__(self, cfg: TESSAStructuralizerConfigV3) -> None:
        super().__init__()
        self.memory = TESSATriStreamMemoryV3(
            scene_input_dim=3,
            surface_input_dim=3,
            presentation_input_dim=3,
            config=cfg,
        )
        self.query = nn.Parameter(torch.randn(1, cfg.d_model) / (cfg.d_model ** 0.5))
        self.read = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        scene: torch.Tensor,
        surface: torch.Tensor,
        presentation: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.memory(
            scene_source=scene,
            surface_source=surface,
            presentation_source=presentation,
        )
        q = self.query[None].expand(scene.shape[0], -1, -1)
        update, _ = self.read(q, memory, memory, need_weights=False)
        h = self.norm(q + update)
        return self.mlp(h[:, 0]).squeeze(-1)


def lattice() -> torch.Tensor:
    axis = torch.linspace(-0.5, 0.5, 4, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(16, 2)


def batch_v1(
    count: int,
    *,
    generator: torch.Generator,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if arm not in {"TRUE_ENDPOINT", "SHUFFLED_ENDPOINT"}:
        raise ValueError("TESSA_ADDRESSABILITY_ARM_INVALID")
    coords = lattice()
    bits = torch.randint(0, 2, (count, 16), generator=generator, dtype=torch.int64)
    payload = bits.to(torch.float32).mul(2.0).sub(1.0)
    permutation = torch.rand((count, 16), generator=generator).argsort(dim=1)
    target_index = torch.randint(0, 16, (count,), generator=generator, dtype=torch.int64)
    alternate_offset = torch.randint(1, 16, (count,), generator=generator, dtype=torch.int64)
    alternate_index = (target_index + alternate_offset) % 16

    expanded_coords = coords[None].expand(count, -1, -1)
    surface_unpermuted = torch.cat((expanded_coords, payload[..., None]), dim=-1)
    surface = torch.gather(
        surface_unpermuted,
        1,
        permutation[..., None].expand(-1, -1, 3),
    )

    chosen = target_index if arm == "TRUE_ENDPOINT" else alternate_index
    chosen_xy = coords[chosen]
    presentation = torch.cat(
        (chosen_xy, torch.ones(count, 1, dtype=torch.float32)), dim=-1
    )[:, None, :]
    scene = torch.zeros(count, 4, 3, dtype=torch.float32)
    target = bits[torch.arange(count), target_index].to(torch.float32)
    return scene, surface, presentation, target


def instantiate_from_state(
    cfg: TESSAStructuralizerConfigV3,
    state: dict[str, torch.Tensor],
) -> AddressabilityModel:
    model = AddressabilityModel(cfg)
    model.load_state_dict(copy.deepcopy(state), strict=True)
    return model


def run_arm(seed: int, arm: str, initial_state: dict[str, torch.Tensor]) -> dict:
    cfg = config_v1()
    model = instantiate_from_state(cfg, initial_state)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    train_gen = torch.Generator(device="cpu").manual_seed(seed + 100_000)

    last_loss = None
    for _ in range(TRAIN_STEPS):
        scene, surface, presentation, target = batch_v1(
            BATCH_SIZE, generator=train_gen, arm=arm
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(scene, surface, presentation)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    model.eval()
    eval_gen = torch.Generator(device="cpu").manual_seed(seed + 200_000)
    total_correct = 0
    total_bce = 0.0
    total_pred_positive = 0
    total_target_positive = 0
    seen = 0
    with torch.no_grad():
        while seen < EVAL_COUNT:
            n = min(BATCH_SIZE, EVAL_COUNT - seen)
            scene, surface, presentation, target = batch_v1(n, generator=eval_gen, arm=arm)
            logits = model(scene, surface, presentation)
            total_bce += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            pred = logits >= 0.0
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


def run() -> dict:
    torch.set_num_threads(1)
    cfg = config_v1()
    cfg.validate()
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        template = AddressabilityModel(cfg)
        initial_state = copy.deepcopy(template.state_dict())
        treatment = run_arm(seed, "TRUE_ENDPOINT", initial_state)
        control = run_arm(seed, "SHUFFLED_ENDPOINT", initial_state)
        results.extend((treatment, control))

    treatment_rows = [r for r in results if r["arm"] == "TRUE_ENDPOINT"]
    control_rows = [r for r in results if r["arm"] == "SHUFFLED_ENDPOINT"]
    by_seed = {r["seed"]: r for r in treatment_rows}
    control_by_seed = {r["seed"]: r for r in control_rows}
    treatment_mean = sum(r["eval_accuracy"] for r in treatment_rows) / len(treatment_rows)
    firewall_ok = all(0.45 <= r["eval_accuracy"] <= 0.55 for r in control_rows)
    treatment_ok = all(r["eval_accuracy"] >= 0.90 for r in treatment_rows) and treatment_mean >= 0.95
    paired_gain_ok = all(
        by_seed[s]["eval_accuracy"] - control_by_seed[s]["eval_accuracy"] >= 0.35
        for s in SEEDS
    )

    if not firewall_ok:
        status = "SYNTHETIC_FIREWALL_INVALID"
    elif treatment_ok and paired_gain_ok:
        status = "MECHANISM_ACCESSIBLE_TINY_V1"
    else:
        status = "TINY_LATE_FUSION_ADDRESSABILITY_NOT_ESTABLISHED"

    return {
        "schema": SCHEMA,
        "status": status,
        "preregistration": "SYNTHETIC_TESSA_ADDRESSABILITY_PREREG_V1.md",
        "config": asdict(cfg),
        "seeds": list(SEEDS),
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "evaluation_count": EVAL_COUNT,
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "schedule": None,
            "gradient_clipping": None,
        },
        "results": results,
        "treatment_mean_accuracy": treatment_mean,
        "firewall_ok": firewall_ok,
        "treatment_thresholds_ok": treatment_ok,
        "paired_gain_thresholds_ok": paired_gain_ok,
        "authority": {
            "reference_tessa_pass_claimed": False,
            "topology_pass_claimed": False,
            "truth139_training_authorized": False,
            "fit8_training_authorized": False,
            "product_authority_changed": False,
        },
        "execution": {
            "device": "cpu",
            "gpu_required": False,
            "product_inference_executed": False,
        },
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
        "treatment_mean_accuracy": report["treatment_mean_accuracy"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
