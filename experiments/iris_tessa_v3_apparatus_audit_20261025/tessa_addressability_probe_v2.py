#!/usr/bin/env python3
"""Execute SYNTHETIC_TESSA_ADDRESSABILITY_PREREG_V2 exactly.

Tiny CPU learned probe only. It isolates endpoint retrieval after current TESSA
independent typed-stream compression, with a random-label firewall and an
uncompressed two-hop positive control.
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

SCHEMA = "RealSaS.TESSAV3SyntheticAddressabilityProbe.v2"
SEEDS = (731, 947, 1217)
TRAIN_STEPS = 1000
BATCH_SIZE = 128
EVAL_COUNT = 4096
LR = 1.0e-3
SCENE_TOKENS = 4
SURFACE_MEMORY_TOKENS = 6
PRESENTATION_MEMORY_TOKENS = 2


def config_v2() -> TESSAStructuralizerConfigV3:
    return TESSAStructuralizerConfigV3(
        d_model=64,
        n_heads=4,
        decoder_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        coordinate_bins=128,
        max_faces=128,
        max_vertices=128,
        scene_memory_tokens=SCENE_TOKENS,
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
    def __init__(self, d_model: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) / (d_model ** 0.5))
        self.presentation_read = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.surface_read = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.presentation_norm = nn.LayerNorm(d_model)
        self.surface_norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, presentation_memory: torch.Tensor, surface_memory: torch.Tensor) -> torch.Tensor:
        batch = presentation_memory.shape[0]
        q = self.query[None].expand(batch, -1, -1)
        p, _ = self.presentation_read(q, presentation_memory, presentation_memory, need_weights=False)
        q = self.presentation_norm(q + p)
        s, _ = self.surface_read(q, surface_memory, surface_memory, need_weights=False)
        q = self.surface_norm(q + s)
        return self.out(q[:, 0]).squeeze(-1)


class CompressedAddressabilityModel(nn.Module):
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
        presentation_end = surface_end + self.cfg.presentation_memory_tokens
        surface_memory = memory[:, scene_end:surface_end]
        presentation_memory = memory[:, surface_end:presentation_end]
        if surface_memory.shape[1] != self.cfg.surface_memory_tokens:
            raise RuntimeError("TESSA_ADDR_V2_SURFACE_SLICE_DRIFT")
        if presentation_memory.shape[1] != self.cfg.presentation_memory_tokens:
            raise RuntimeError("TESSA_ADDR_V2_PRESENTATION_SLICE_DRIFT")
        return self.reader(presentation_memory, surface_memory)


class RawPositiveControlModel(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.surface = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.presentation = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.surface_type = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5))
        self.presentation_type = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5))
        self.reader = TwoHopReader(d_model, heads)

    def forward(self, scene: torch.Tensor, surface: torch.Tensor, presentation: torch.Tensor) -> torch.Tensor:
        del scene
        s = self.surface(surface) + self.surface_type[None, None, :]
        p = self.presentation(presentation) + self.presentation_type[None, None, :]
        return self.reader(p, s)


def lattice() -> torch.Tensor:
    axis = torch.linspace(-0.5, 0.5, 4, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(16, 2)


def input_batch(
    count: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = lattice()
    bits = torch.randint(0, 2, (count, 16), generator=generator, dtype=torch.int64)
    payload = bits.to(torch.float32).mul(2.0).sub(1.0)
    permutation = torch.rand((count, 16), generator=generator).argsort(dim=1)
    target_index = torch.randint(0, 16, (count,), generator=generator, dtype=torch.int64)

    expanded_coords = coords[None].expand(count, -1, -1)
    unpermuted = torch.cat((expanded_coords, payload[..., None]), dim=-1)
    surface = torch.gather(unpermuted, 1, permutation[..., None].expand(-1, -1, 3))
    target_xy = coords[target_index]
    presentation = torch.cat(
        (target_xy, torch.ones(count, 1, dtype=torch.float32)), dim=-1
    )[:, None, :]
    scene = torch.zeros(count, 4, 3, dtype=torch.float32)
    target = bits[torch.arange(count), target_index].to(torch.float32)
    return scene, surface, presentation, target


def instantiate_like(model: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    clone = copy.deepcopy(model)
    clone.load_state_dict(copy.deepcopy(state), strict=True)
    return clone


def train_eval(
    *,
    seed: int,
    arm: str,
    template: nn.Module,
    initial_state: dict[str, torch.Tensor],
) -> dict:
    if arm not in {
        "COMPRESSED_TRUE_ENDPOINT",
        "COMPRESSED_RANDOM_LABEL_FIREWALL",
        "RAW_TWO_HOP_POSITIVE_CONTROL",
    }:
        raise ValueError("TESSA_ADDR_V2_ARM_INVALID")
    model = instantiate_like(template, initial_state)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0
    )
    input_gen = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    firewall_label_gen = torch.Generator(device="cpu").manual_seed(seed + 300_000)
    last_loss = None

    for _ in range(TRAIN_STEPS):
        scene, surface, presentation, true_target = input_batch(
            BATCH_SIZE, generator=input_gen
        )
        if arm == "COMPRESSED_RANDOM_LABEL_FIREWALL":
            target = torch.randint(
                0, 2, (BATCH_SIZE,), generator=firewall_label_gen, dtype=torch.int64
            ).to(torch.float32)
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
    firewall_label_gen = torch.Generator(device="cpu").manual_seed(seed + 400_000)
    total_correct = 0
    total_bce = 0.0
    total_pred_positive = 0
    total_target_positive = 0
    seen = 0
    with torch.no_grad():
        while seen < EVAL_COUNT:
            n = min(BATCH_SIZE, EVAL_COUNT - seen)
            scene, surface, presentation, true_target = input_batch(n, generator=input_gen)
            if arm == "COMPRESSED_RANDOM_LABEL_FIREWALL":
                target = torch.randint(
                    0, 2, (n,), generator=firewall_label_gen, dtype=torch.int64
                ).to(torch.float32)
            else:
                target = true_target
            logits = model(scene, surface, presentation)
            total_bce += float(
                F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
            )
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
    cfg = config_v2()
    cfg.validate()
    results: list[dict] = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        compressed_template = CompressedAddressabilityModel(cfg)
        compressed_state = copy.deepcopy(compressed_template.state_dict())
        results.append(
            train_eval(
                seed=seed,
                arm="COMPRESSED_TRUE_ENDPOINT",
                template=compressed_template,
                initial_state=compressed_state,
            )
        )
        results.append(
            train_eval(
                seed=seed,
                arm="COMPRESSED_RANDOM_LABEL_FIREWALL",
                template=compressed_template,
                initial_state=compressed_state,
            )
        )

        torch.manual_seed(seed)
        raw_template = RawPositiveControlModel(cfg.d_model, cfg.n_heads)
        raw_state = copy.deepcopy(raw_template.state_dict())
        results.append(
            train_eval(
                seed=seed,
                arm="RAW_TWO_HOP_POSITIVE_CONTROL",
                template=raw_template,
                initial_state=raw_state,
            )
        )

    compressed = [r for r in results if r["arm"] == "COMPRESSED_TRUE_ENDPOINT"]
    firewall = [r for r in results if r["arm"] == "COMPRESSED_RANDOM_LABEL_FIREWALL"]
    raw = [r for r in results if r["arm"] == "RAW_TWO_HOP_POSITIVE_CONTROL"]

    mean_compressed = sum(r["eval_accuracy"] for r in compressed) / len(compressed)
    mean_raw = sum(r["eval_accuracy"] for r in raw) / len(raw)
    firewall_ok = all(0.45 <= r["eval_accuracy"] <= 0.55 for r in firewall)
    raw_ok = all(r["eval_accuracy"] >= 0.90 for r in raw) and mean_raw >= 0.95
    compressed_ok = (
        all(r["eval_accuracy"] >= 0.90 for r in compressed)
        and mean_compressed >= 0.95
    )

    if not firewall_ok:
        status = "V2_SYNTHETIC_FIREWALL_INVALID"
    elif not raw_ok:
        status = "V2_READER_OR_TASK_NOT_ESTABLISHED"
    elif compressed_ok:
        status = "MECHANISM_ACCESSIBLE_TINY_V2"
    else:
        status = "CURRENT_INDEPENDENT_COMPRESSION_ADDRESSABILITY_NOT_ESTABLISHED_TINY_V2"

    return {
        "schema": SCHEMA,
        "status": status,
        "preregistration": "SYNTHETIC_TESSA_ADDRESSABILITY_PREREG_V2.md",
        "v1_firewall_postmortem": {
            "v1_architecture_verdict": None,
            "bayes_optimal_control_accuracy_from_unintended_15_locus_majority_channel": 0.604736328125,
        },
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
        "mean_compressed_true_endpoint_accuracy": mean_compressed,
        "mean_raw_positive_control_accuracy": mean_raw,
        "firewall_ok": firewall_ok,
        "raw_positive_control_ok": raw_ok,
        "compressed_thresholds_ok": compressed_ok,
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
        "mean_compressed_true_endpoint_accuracy": report["mean_compressed_true_endpoint_accuracy"],
        "mean_raw_positive_control_accuracy": report["mean_raw_positive_control_accuracy"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
