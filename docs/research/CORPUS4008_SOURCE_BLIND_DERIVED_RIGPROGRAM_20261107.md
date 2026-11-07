# Source-blind derived RigProgram control — first real native-rig experiments

**2026-11-07.** Research-only branch; no main merge, model checkpoint changes, source truth admission or production Compiler modifications.

## Architectural boundary

- **Input evidence:** original source joint forest, rest/bone geometry, imported sparse vertex-group coverage and animation action-curve count *when available*.
- **Source native rig oracle:** original IK constraint owner, chain count, target and pole graph; kept outside source-only feature constructor and used only **after** the blind candidate array is SHA frozen.
- **AXIS V6 provider:** existing typed `RigProgramControlProposalIR`, `RigProgramChainProposalIR`, `RigProgramConstraintProposalIR`. Experimental source-only candidates are `INFERRED_IK_PRIOR` and can become future supervised/auxiliary model hypotheses only after separate rights and source-aligned courts.
- **Compiler:** validates declared proposal references/lineage; **MUST NOT** search for, add, or repair missing controls. A structurally accepted RigProgram is not a working IK proof.
- **MIRA:** tests exact source skin influence and deformation consequences. Merely observing deformed vertices after a CCD solve does not validate weight identity, anatomy, attachment correctness or source-native equivalence.
- **Runtime control:** requires separate solver and user-interaction product capability court; this experiment is not wired to the shipped puppet.

## Results already measured (development sources, pinned native CC0 files)

Source: Mesh2Motion/mesh2motion-assets, commit `dc2eb3cf06ba0aaeb0c651503d5667f773781522`. Blender 4.5.14 `--disable-autoexec`. Hash-locked source transport/decoded bytes. Workflows run on independent GitHub-hosted jobs, evidence stored on private Drive.

**Native oracle discovery**, [run 42401941730](https://github.com/merynz/RealSaS-OPT/actions/runs/42401941730), SHA of blind program frozen before oracle:
  
| Rig | Native IK constraints | Weighted distal source-only candidate chains | Top32 chain overlap >= 0.5 | Exact native owner effector |
|---|---:|---:|---:|---:|
| Spider | 8 | 48 | 8/8 | 0/8 |
| Dragon | 6 | 87 | 2/6 | 0/6 |
| Snake | 0 | 10 | N/A | N/A |

Inspecting the actual source-bound private receipts revealed that the **native IK owner bone is contained within** one or more candidate source-only chains in all 8 Spider and 6 Dragon cases, but it was not the candidate's proposed distal end. This is an inference/representation-policy distinction, NOT evidence of recovered native animator controls.

**Constraint-masked derived CCD trial**, [run 42402384752](https://github.com/merynz/RealSaS-OPT/actions/runs/42402384752):

| Rig | Independent world-relative target moves | >50% reduction of goal-distance error | Evaluated mesh response |
|---|---:|---:|---:|
| Spider | 15 | 14 | 15 |
| Dragon | 15 | 15 | 15 |
| Snake | 9 | 8 | 9 |
| **Total** | **39** | **37** | **39** |

All native bone/object constraints were muted **after source-only proposal freeze**, before CCD target solving. Mesh response is evaluated Blender skin/pose behavior, not source-human-approved deformation. The solver currently has no source-native pole target inference, authoring DOF/limits, contact, collision, full weight support validation, per-frame animation fit, locomotion or body-part semantics. These results cannot be presented as a full IK or editable puppet feature.

## Key methodological finding and next hypothesis

The initial candidate policy chooses **weighted deform distal bones** as prospective end effectors. Original IK constraints on native rig may act on **intermediate articulated bones** instead. Hence exact owner recovery was zero even while source joint-chain overlap was substantial.

A second *generic* hypothesis, `discover_articulation_frontiers`, proposes some intermediate ancestors of weighted distal joints as additional effectors, using only the same allowlisted bone/skin/motion data. It is an **ablation informed by the Spider/Dragon development set**, not an unseen result.

We will compare two frozen policies before opening oracle labels, report top32 and top128 chain Jaccard and exact owner ranking separately, then test **six previously uninspected native rigs**: Bird, Fox, Horse, Human, Kaiju and Shark. Snake is an existing negative-control example. Native IK prevalence is not presumed on heldout rigs. Their original source material may lack skin/motion evidence; those remain `UNKNOWN`, not failed subject evidence.

## Source truth/state matrix

| Evidence family | Current state | Valid claim |
|---|---|---|
| Joint parent forest and local rest geometry | IMPORTED_OBSERVATION | Source-skeleton structure measured |
| Sparse skin weights | IMPORTED_OBSERVATION | Weight coverage hint; no source bind parity |
| Action-curve counts | IMPORTED_OBSERVATION or UNKNOWN | Motion hint, not action semantics |
| Original native IK constraint presence | SOURCE_ORACLE_OBSERVATION | IK exists in native .blend |
| Native IK target→joint→deformation response | SOURCE_ORACLE_INTERVENTION_OBSERVATION | Earlier separate Spider/Dragon source experiment |
| Derived chain inference | INFERRED_IK_PRIOR | Source-only candidate control graph |
| Derived CCD target motion | BEHAVIORAL_OBSERVATION_ONLY | Goal-error reduction and skin movement |
| Source-native owner/control identity recovered | UNKNOWN/NOT_VALIDATED | Must compare in independent target cases |
| Anatomical constraints/contact/attachment consequences | UNKNOWN | No target authority |
| Model per-head training admitted | FALSE | No legal/semantic/mechanical court bundle |
| Materialization authorized | FALSE | Preflight remains blocked |

## Pre-materialization validation upgrades

For source to admit **derived IK supervision** instead of a research prior, require independent: (1) commercial ML rights, split and source lineage; (2) exact source rest/bind matrices and sparse weights; (3) source-frame and coordinate roundtrip; (4) multiple target goals not used in proposal selection; (5) endpoint error normalized by source extent, full solver trajectory and source joint limit plausibility; (6) contact/attachment retention and negative unaffected regions; (7) MIRA per-vertex deformation error and layer/visibility/Graphics art when consuming rendered truth; (8) source IK oracle equivalence when native controls are available, otherwise clearly `DERIVED` rather than `NATIVE`.

**No pass through all these gates currently.** No source-native IK controller label is inferred merely because a source-only CCD solver can move a weighted chain.
