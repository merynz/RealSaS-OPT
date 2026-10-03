from __future__ import annotations

"""Head-only D1 diffusion fit using frozen D0 anonymous matching.

The frozen D0 Geppetto output defines query<->teacher geometry assignment once.
D1 training cannot move that assignment. Control states are detached and only
ConditionalDiffusionLocusHeadV1 parameters are optimized.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np
import torch

from .geppetto_diffusion_challenger_v1 import (
    GeppettoCandidateDiffusionLocusV1,
)
from .geppetto_loss_v2 import canonical_geometry_assignment_v2
from .training_targets_v1 import GeppettoTeacherTargetV1


@dataclass(frozen=True)
class FrozenD0LocusPairsV1:
    conditions: torch.Tensor
    targets_xyz_normalized: torch.Tensor
    source_rows: tuple[tuple[int,int,int], ...]
    mapping_hash: str


def _mapping_hash(rows, targets) -> str:
    payload={
        "schema":"RealSaS.GeppettoFrozenD0LocusPairs.v1",
        "rows":[list(map(int,r)) for r in rows],
        "targets":[
            [float(x) for x in row]
            for row in targets.detach().cpu().double().tolist()
        ],
    }
    return sha256(
        json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def build_frozen_d0_locus_pairs_v1(
    base_output,
    targets: Sequence[GeppettoTeacherTargetV1],
) -> FrozenD0LocusPairsV1:
    B,K,D=base_output.control_states.shape
    if len(targets)!=B:
        raise ValueError("D1 pair builder batch mismatch")
    conditions=[]
    target_rows=[]
    source_rows=[]
    for b,target in enumerate(targets):
        if not bool(target.valid):
            continue
        truth=np.asarray(target.positions_normalized,dtype=np.float64)
        J=int(len(truth))
        if J<1:
            continue
        if J>K:
            raise ValueError("D1 teacher count exceeds frozen decoded queries")
        primary=base_output.positions_normalized[b,:J]
        q_np,t_np=canonical_geometry_assignment_v2(primary,truth)
        q=torch.as_tensor(
            q_np,device=base_output.control_states.device,dtype=torch.long
        )
        t=torch.as_tensor(
            t_np,device=base_output.control_states.device,dtype=torch.long
        )
        truth_t=torch.as_tensor(
            truth,
            device=base_output.control_states.device,
            dtype=base_output.control_states.dtype,
        )[t]
        conditions.append(base_output.control_states[b,q].detach())
        target_rows.append(truth_t.detach())
        source_rows.extend(
            (int(b),int(qq),int(tt))
            for qq,tt in zip(q_np.tolist(),t_np.tolist())
        )
    if not conditions:
        raise ValueError("D1 pair builder produced no valid training rows")
    cond=torch.cat(conditions,dim=0)
    truth=torch.cat(target_rows,dim=0)
    return FrozenD0LocusPairsV1(
        conditions=cond,
        targets_xyz_normalized=truth,
        source_rows=tuple(source_rows),
        mapping_hash=_mapping_hash(source_rows,truth),
    )


def fit_diffusion_locus_head_v1(
    challenger: GeppettoCandidateDiffusionLocusV1,
    pairs: FrozenD0LocusPairsV1,
    *,
    steps: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 77123,
) -> dict:
    if steps<1:
        raise ValueError("D1 fit steps must be positive")
    if learning_rate<=0 or weight_decay<0 or grad_clip<=0:
        raise ValueError("D1 optimizer contract invalid")
    scope=challenger.freeze_base_for_head_fit()
    params=[
        p for n,p in challenger.named_parameters()
        if n.startswith("diffusion_locus_head.") and p.requires_grad
    ]
    if not params:
        raise RuntimeError("D1_HEAD_ONLY_SCOPE_EMPTY")
    opt=torch.optim.AdamW(
        params,lr=float(learning_rate),weight_decay=float(weight_decay)
    )
    device=pairs.conditions.device
    gen=torch.Generator(device=device)
    gen.manual_seed(int(seed))
    losses=[]
    challenger.train()
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        out=challenger.diffusion_locus_head.training_loss(
            pairs.conditions,
            pairs.targets_xyz_normalized,
            generator=gen,
        )
        loss=out["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"D1_HEAD_ONLY_LOSS_NONFINITE:{step}")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,float(grad_clip))
        if not torch.isfinite(torch.as_tensor(norm)):
            raise RuntimeError(f"D1_HEAD_ONLY_GRAD_NONFINITE:{step}")
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "schema":"RealSaS.GeppettoDiffusionLocusHeadFit.v1",
        "status":"PASS_HEAD_ONLY_FIT__AWAIT_COURT",
        "mapping_hash":pairs.mapping_hash,
        "training_row_count":int(len(pairs.conditions)),
        "steps":int(steps),
        "learning_rate":float(learning_rate),
        "weight_decay":float(weight_decay),
        "grad_clip":float(grad_clip),
        "seed":int(seed),
        "initial_loss":float(losses[0]),
        "final_loss":float(losses[-1]),
        "minimum_loss":float(min(losses)),
        "trainable_parameter_names":scope["trainable"],
        "frozen_parameter_count":len(scope["frozen"]),
        "product_authority_claimed":False,
    }


__all__=[
    "FrozenD0LocusPairsV1",
    "build_frozen_d0_locus_pairs_v1",
    "fit_diffusion_locus_head_v1",
]
