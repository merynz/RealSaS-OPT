from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "canonical" / "HISTORICAL_PRODUCT_POWER_REBIND_MANIFEST_V1_20260913.md"
JS = ROOT / "canonical" / "HISTORICAL_PRODUCT_POWER_REBIND_MANIFEST_V1_20260913.json"

STATUS = "P0_CLOSED_PASS__P1_P2_P3_OPEN__NO_MAIN_PROMOTION__NO_SCIENCE_GATE_CHANGE"
NEXT = "WAIT_FOR_EXISTING_CORRECTED_G_W_MESH_SCIENCE_GATES_THEN_BIND_EXACT_G_W_M_B_COMPONENTS_INTO_CLOSED_COMPILE_TRANSACTION"

CLOSURE_SECTION = """## P0 closure — 2026-09-13

P0 is now **CLOSED PASS**. The historical source reserve was physically reacquired and SHA-bound; the audited orchestration/reference-runtime files are byte-identical across v0.5, Aug-7, V19_30R1 and V19_30R2. The old monolithic orchestrator remains rejected as a current owner because it imports superseded authority families.

The reusable control semantics were rebound into the current typed transaction authority at `compiler/realsas_compiler_core/compile_transaction.py`: immutable exact artifact bindings, frozen stage policy/implementation identities, terminal `FAIL`/`ABSTAIN`, explicit child transactions for retry/repair, and exact product -> proof -> runtime identity continuity. Literal mutable `latest/current/newest` artifact aliases are forbidden.

Self-hosted contract run `34762089212` on `realsas-wsl-1660ti` closed green with **28/28 behavioral tests PASS**: compile transaction `9/9`, bounded repair loop `5/5`, V4 architecture `10/10`, Living Compile identity `4/4`.

Canonical closure evidence:

- `canonical/HISTORICAL_PRODUCT_POWER_P0_CLOSURE_20260913.md`
- `canonical/HISTORICAL_PRODUCT_POWER_P0_CLOSURE_20260913.json`
- `canonical/HISTORICAL_PRODUCT_POWER_P0A_SOURCE_LEDGER_V1_20260913.json`

This P0 closure does **not** claim corrected real G/W/M/B, professional motion, final runtime product closure or `PRODUCT_PASS`. P1 begins only when the already-existing corrected Geppetto/Arachne/mesh scientific and product gates close; those exact qualified artifacts then bind into the closed transaction contract without redesigning orchestration.

"""


def sync_markdown() -> None:
    text = MD.read_text(encoding="utf-8")
    old_status = "**Status:** `AUDIT_MANIFEST__NO_PROMOTION__NO_SGW_AUTHORITY_CHANGE`"
    new_status = f"**Status:** `{STATUS}`"
    if old_status in text:
        text = text.replace(old_status, new_status, 1)
    elif new_status not in text:
        raise SystemExit("unexpected Markdown status line")

    marker = "## P0 closure — 2026-09-13"
    if marker not in text:
        anchor = "**Purpose:** recover still-useful historical Compiler/runtime/product machinery behind the current typed authority without restoring obsolete ownership, parallel truth, or old front-brain semantics.\n\n"
        if anchor not in text:
            raise SystemExit("Markdown insertion anchor not found")
        text = text.replace(anchor, anchor + CLOSURE_SECTION, 1)

    old_next = """## 15. Immediate next engineering action after this manifest

Do **not** write a new 5k-line orchestrator.

Next task is P0A:

1. reacquire/materialize exact v0.5 orchestrator/proof/retry/feedback source by recorded SHA where accessible;
2. source-diff it against current `realsas_compiler_core`, proof/repair services, runtime/export and Living Compile;
3. produce a file-level dependency closure with `KEEP_CURRENT / REBIND / TEST_ONLY / REJECT` decisions;
4. then implement the smallest current typed transaction shell that reuses proven control behavior.

In parallel, do not alter the frozen active mesh experiment. Do not start/freeze a new Arachne product-consumer contract until P0B has fixed exact transaction identity semantics.
"""
    new_next = """## 15. Immediate next engineering action after P0 closure

Do **not** redesign orchestration and do **not** write a new 5k-line orchestrator. P0A/P0B/P0C are closed by the exact source ledger, current typed transaction implementation, self-hosted contract evidence and closure record.

Next legal transition is P1: wait for the already-authorized corrected G/W/mesh gates to close, then bind those exact qualified G/W/M/B/component identities into `CompileTransactionIR`. The transaction may consume them; it may not regenerate, substitute, retopologize or silently remap them.

The frozen active science remains unchanged. P2 professional motion and P3 runtime/editor productization remain open downstream work; P4 numerical backends remain blocker-triggered only.
"""
    if old_next in text:
        text = text.replace(old_next, new_next, 1)

    MD.write_text(text, encoding="utf-8")


def _close_p0(obj):
    if isinstance(obj, dict):
        ident = str(obj.get("id", obj.get("wave", obj.get("phase", "")))).upper()
        if ident == "P0":
            if "state" in obj:
                obj["state"] = "CLOSED_PASS"
            if "status" in obj:
                obj["status"] = "CLOSED_PASS"
        for value in obj.values():
            _close_p0(value)
    elif isinstance(obj, list):
        for value in obj:
            _close_p0(value)


def sync_json() -> None:
    data = json.loads(JS.read_text(encoding="utf-8"))
    data["status"] = STATUS
    data["p0_closure"] = {
        "status": "CLOSED_PASS",
        "closure_md": "canonical/HISTORICAL_PRODUCT_POWER_P0_CLOSURE_20260913.md",
        "closure_json": "canonical/HISTORICAL_PRODUCT_POWER_P0_CLOSURE_20260913.json",
        "source_ledger": "canonical/HISTORICAL_PRODUCT_POWER_P0A_SOURCE_LEDGER_V1_20260913.json",
        "workflow_run_id": 34762089212,
        "behavioral_tests": "28/28 PASS",
        "typed_transaction_module": "compiler/realsas_compiler_core/compile_transaction.py",
        "historical_owner_restored": False,
        "scientific_gate_changes": False,
        "main_promotion": False,
    }
    _close_p0(data)
    data["next_action"] = {
        "wave": "P1",
        "instruction": NEXT,
        "precondition": "EXISTING_CORRECTED_G_W_MESH_SCIENTIFIC_AND_PRODUCT_GATES_CLOSED",
        "transaction_redesign_allowed": False,
        "no_new_5k_orchestrator_from_scratch": True,
    }
    claims = data.setdefault("claims", {})
    claims["restored_product_capability"] = False
    claims["product_pass"] = False
    claims["scientific_gate_changes"] = False
    claims["historical_numerical_promotion"] = False
    claims["p0_transaction_contract_closed"] = True
    JS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sync_markdown()
    sync_json()
    print("P0 parent manifests synchronized")
