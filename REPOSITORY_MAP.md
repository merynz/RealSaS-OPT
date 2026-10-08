# Repository Map

Canonical code continuation is `main`. Start with [CURRENT_STATE.md](CURRENT_STATE.md), [AGENTS.md](AGENTS.md) and [Execution lanes](docs/platform/EXECUTION_LANES.md). This map is navigation, not scientific authority.

## Executable homes

| Path | Role |
|---|---|
| `platform/` | Go API/CLI, Registry, release DAGs, Attempts/ProductRevisions, scheduling and durable workflow control |
| `models/iris/` | Geometry/surface evidence |
| `models/tessa/` | Learned topology/geometry proposals |
| `models/axis/` | Required rig proposals |
| `models/mira/` | Carrier-native skin proposals |
| `compiler/realsas_compiler_core/` | Typed identity, binding, qualification, appearance, motion and product contracts |
| `compiler/realsas_compiler_services/platform_worker/` | Engine activities and verified CAS stage-input hydration |
| `compiler/realsas_compiler_services/orchestrator/` | Current 46-stage Compiler compatibility plan and scientific stage execution |
| `compiler/realsas_compiler_services/` | Subordinate deterministic proof, repair and export services |
| `runtime/realsas_cpp/` | Qualified native package consumption and rendering |
| `tools/platform_release_snapshot.py` | Source-only stage/DAG version snapshot for a release |
| `tests/` | Ownership, semantic, adversarial and integration contracts |

Legacy Geppetto/Arachne/ATLAS directories and identifiers preserve schema/checkpoint compatibility. Public architecture names are IRIS/TESSA/AXIS/MIRA; see [SYSTEM_INDEX.md](SYSTEM_INDEX.md) for ownership.

## Documentation and evidence

| Path | Read as |
|---|---|
| `docs/platform/` | Current operator contract and architecture decisions |
| `docs/repository/` | Repository structure and contribution rules |
| `canonical/` | Current contracts plus sealed scientific provenance; enter through its index |
| `experiments/` | Bounded research apparatus and records, without product authority |
| `prospective/` | Unpromoted hypotheses/plans |
| `historical/` | Preserved snapshots and catalog of older sealed records |
| `restoration/` | Restoration-era source/audit provenance |

Machine authority begins with [Canonical index](canonical/README.md). Generated `REHYDRATION_PACKET.md` and `LIVE_AUTHORITY_MAP.md` are navigation/cache views.

## Execution state

Go Attempts and Registry artifacts own ongoing research state. Every release binds an immutable DAG; research DAGs may add/remove/rewire nodes while preserving unrelated artifacts. Product state belongs to qualified ProductRevisions.

Compatibility Engine executions also keep run-local ledgers at `$REALSAS_AUTHORITY_ROOT/runs/<run_id>/ACTIVE_RUN_V2.json`. The repository's `canonical/ACTIVE_RUN_V2.json` is implementation governance, not a subject witness ledger.

Current witness authorization and qualification remain explicit. Historical branch names, frozen GIFs and green CI alone do not mint new scientific or product authority.
