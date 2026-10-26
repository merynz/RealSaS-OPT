"""Separate one scientific optimizer update from physical packing.

The loss callback returns one loss SUM per example. Each family must already be
normalized according to the sealed objective; raw families with unrelated units
cannot be summed here. Batch-coupled or stochastic operations need their own
equivalence court. There is exactly one clip/optimizer/scheduler per update.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SemanticExample:
    key: str
    loss_mass: float
    weight: float
    cost: int


def physical_batches(examples, *, max_padded_cost, max_examples):
    if not examples or min(max_padded_cost, max_examples) <= 0:
        raise ValueError("SEMANTIC_EMPTY_OR_INVALID_BUDGET")
    if len({x.key for x in examples}) != len(examples):
        raise ValueError("SEMANTIC_DUPLICATE_EXAMPLE")
    for x in examples:
        if (not x.key or not math.isfinite(x.loss_mass) or not math.isfinite(x.weight)
                or min(x.loss_mass, x.weight, x.cost) <= 0):
            raise ValueError("SEMANTIC_EXAMPLE_INVALID")
        if x.cost > max_padded_cost:
            raise ValueError(f"SEMANTIC_EXAMPLE_EXCEEDS_MEMORY_PLAN:{x.key}")
    batches, current = [], []
    for x in sorted(examples, key=lambda v: (v.cost, v.key)):
        if current and (len(current) >= max_examples or x.cost*(len(current)+1) > max_padded_cost):
            batches.append(tuple(current))
            current = []
        current.append(x)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def execute_semantic_step(examples, *, loss_sums, optimizer, scheduler, parameters,
                          max_padded_cost, max_examples, clip_norm):
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("SEMANTIC_CLIP_INVALID")
    batches = physical_batches(examples, max_padded_cost=max_padded_cost, max_examples=max_examples)
    total_weight = sum(x.weight for x in examples)
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    parameters = tuple(parameters)
    try:
        for batch in batches:
            sums = loss_sums(batch)
            if sums.shape != (len(batch),) or not torch.isfinite(sums).all():
                raise ValueError("SEMANTIC_LOSS_SUMS_INVALID")
            factors = sums.new_tensor([x.weight/(total_weight*x.loss_mass) for x in batch])
            loss = (sums*factors).sum()
            loss.backward()
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        # Failure during optimizer/scheduler commit is not a retryable microbatch.
        # Worker must reload the last durable checkpoint, including RNG/sampler.
        raise
    return {"loss": total, "gradient_norm": float(norm), "physical_batches": len(batches),
            "semantic_examples": len(examples), "optimizer_updates": 1}
