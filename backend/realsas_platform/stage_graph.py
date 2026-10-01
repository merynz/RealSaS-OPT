from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StageContract:
    ordinal: int
    stage_id: str
    title: str
    group: str
    depends_on: tuple[str, ...]
    adapter: str
    fail_closed: bool
    cacheable: bool
    output_hash_required: bool
    product_pass_authority: bool


class StageGraph:
    def __init__(self, stages: tuple[StageContract, ...]) -> None:
        if len(stages) != 46 or [s.ordinal for s in stages] != list(range(1, 47)):
            raise ValueError("canonical graph must contain ordinal stages 1..46")
        ids = [s.stage_id for s in stages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate stage id")
        seen: set[str] = set()
        for stage in stages:
            unknown = set(stage.depends_on) - seen
            if unknown:
                raise ValueError(f"{stage.stage_id} depends on future/unknown stages: {sorted(unknown)}")
            seen.add(stage.stage_id)
        owners = [s.stage_id for s in stages if s.product_pass_authority]
        if owners != ["46_PRODUCT_CLOSURE_SEAL"]:
            raise ValueError(f"single Stage46 product-pass authority required, got {owners}")
        self._stages = stages
        self._by_id = {s.stage_id: s for s in stages}
        self._children = {s.stage_id: set() for s in stages}
        for stage in stages:
            for parent in stage.depends_on:
                self._children[parent].add(stage.stage_id)

    @classmethod
    def from_canonical_plan(cls, path: Path) -> "StageGraph":
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("stage_count", -1)) != 46:
            raise ValueError("canonical plan stage_count must be 46")
        return cls(tuple(StageContract(
            ordinal=int(row["ordinal"]), stage_id=str(row["id"]), title=str(row["title"]), group=str(row["group"]),
            depends_on=tuple(map(str, row.get("depends_on") or ())), adapter=str(row["adapter"]),
            fail_closed=bool((row.get("policy") or {}).get("fail_closed")), cacheable=bool((row.get("policy") or {}).get("cacheable")),
            output_hash_required=bool((row.get("policy") or {}).get("output_hash_required")), product_pass_authority=bool((row.get("policy") or {}).get("product_pass_authority")),
        ) for row in tuple(payload.get("stages") or ())))

    @property
    def stages(self) -> tuple[StageContract, ...]:
        return self._stages

    def get(self, stage_id: str) -> StageContract:
        return self._by_id[stage_id]

    def ancestors_including(self, targets: Iterable[str]) -> tuple[str, ...]:
        pending = list(dict.fromkeys(targets))
        unknown = [x for x in pending if x not in self._by_id]
        if unknown:
            raise KeyError(f"unknown stage ids: {unknown}")
        required = set(pending)
        while pending:
            current = pending.pop()
            for parent in self._by_id[current].depends_on:
                if parent not in required:
                    required.add(parent)
                    pending.append(parent)
        return tuple(s.stage_id for s in self._stages if s.stage_id in required)

    def descendants_including(self, changed: Iterable[str]) -> tuple[str, ...]:
        pending = list(dict.fromkeys(changed))
        unknown = [x for x in pending if x not in self._by_id]
        if unknown:
            raise KeyError(f"unknown stage ids: {unknown}")
        affected = set(pending)
        while pending:
            current = pending.pop()
            for child in self._children[current]:
                if child not in affected:
                    affected.add(child); pending.append(child)
        return tuple(s.stage_id for s in self._stages if s.stage_id in affected)


PRODUCT_RENDER_ALLOWED_STAGE_IDS = frozenset({"42_RUNTIME_PROJECTION_AND_CAA_BINDING", "43_RSS_MATERIALIZE_COMPACT", "44_NATIVE_PACKAGE_OPEN_PLAYBACK"})


def assert_product_render_stage_subset(stage_ids: Iterable[str]) -> tuple[str, ...]:
    rows = tuple(stage_ids)
    forbidden = sorted(set(rows) - PRODUCT_RENDER_ALLOWED_STAGE_IDS)
    if forbidden:
        raise ValueError("product render requested forbidden compiler stages: " + ",".join(forbidden))
    return rows
