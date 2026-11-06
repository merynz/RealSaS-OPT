# Drawable Capability Decision V2 — independent physical-source checkpoint

2026-11-06 · PR #354 (DRAFT), branch `research/drawable-capability-decision-v2` on PR #351 source head `e07d7dc6dd77e01f88e400c85e51b1e986923852`.

**Scientific status: FINITE CONTRACT DECISION + INDEPENDENT REAL-SOURCE COURT PASS.**  
**NOT: original artist ownership recovered, generic 370 closure, pose/attachment understanding or training admission.**

## Scope and actual execution

- Preregistered protocol: [DRAWABLE_CAPABILITY_DECISION_V2_PREREG_20261106.md](DRAWABLE_CAPABILITY_DECISION_V2_PREREG_20261106.md), decision record pins prereg commit `134b050191c7f2cc2007d6fbac551d2771420c74`.
- [Hosted original physical-evidence CPU decision run 42269086950](https://github.com/merynz/RealSaS-OPT/actions/runs/42269086950): **3/3 jobs PASS** (exact finite solver / 111 tests; actual SHA-pinned original source analysis and separate independently recomputing court; architecture discovery).
- [Authority contract regressions 42269095210](https://github.com/merynz/RealSaS-OPT/actions/runs/42269095210): **PASS**.
- **Zero NEW Blender renders**. This is a preregistered retrospective *new decision comparison* of the already original physically rendered PR #351 Cat and Triceratops datasets; it is not prospective held-out character validation.
- Raw private analysis SHA `500a074105fad1e6db8aef768e559b33f79faa569b97772b9c41521822249876`; separate independent court SHA `415f2a24c61b5f46668070da303f72758447b52c2e526094afd83aabb3a76dad`; execution receipt SHA `afc03715cdb08f85dae87f4e506b0aebefa5ce12c76ee8467dfead3ccba166b1`. These were independently SHA matched to the archived GitHub Actions output. Private Drive folder: https://drive.google.com/drive/folders/1WLbgr8uIfgxpieCPH4D9LYC7TQ3jgaBV.
- Parent **Cat V2** source array `d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114`; source manifest `8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1`; independent original physical court `90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08`.
- Parent **Triceratops** original K3 source array `10bd24f5f0f6bfc482cbcbe1169b46251ea4c7f8c8714022fc577816c2ded7ae`; source manifest `a93b777191a40f3c11faa9735b042058b9984436bf59f947093b799a83d11b37`; independent original physical court `021786e2bc9bc28d30d717c08eec3d92e64fca1ebc714a451ab85e2988b4882d`. Source evidence is teacher-private and the student does not read it.

## Research objects and decision states

- **Source truth E:** immutable original physical sample images, edit arms, material/depth evidence, samples/pose/cameras and validity. Correlation alone does not mint a part label.
- **Capability contract C:** specific controllable edit states, source compositing requirements and measurable error bounds, including explicit UNKNOWN for missing operations.
- **Representation policy P:** optional explicit cost axes and Pareto comparison; does not turn a cheap representation into source truth.
- A drawable is *not* just a connected pixel island: it includes source addresses, operators/state, deformation binding and compositing rule. Mechanical carriers, edit controls, graphic drawables and sampling/depth processing are distinct engineering dimensions.
- Decision outcome is per-capability **SATISFIED, VIOLATED, UNKNOWN**, with counterexamples. A single violated required operation refutes a representation **for this declared contract and representation class**. A missing operation is UNKNOWN, not implicitly PASS/FAIL. Approximate output closeness is a pairwise relation, **not necessarily transitive**; no false union-find equivalence.

## A. Original Cat linked-UV address-control capability

Capability requires **three independently toggleable texture RGB address edits** and all **eight binary combinations** under four original source-rest camera views, identical Cycles sample indices. Five set partitions of three source-address edit toggles were exhaustively enumerated. A tied-control partition can generate only the states for which all addresses under a control share the same bit.

| Grouping of source edit addresses | Independent controls | Exact 8-state edit-address contract | Source observational support — V3/V5 alone | Source observational support — V1/V3/V5/V7 |
| --- | ---: | --- | --- | --- |
| (0,0,0) | 1 | **VIOLATED** | VIOLATED | VIOLATED |
| (0,0,1) | 2 | **VIOLATED** | VIOLATED | VIOLATED |
| (0,1,0) | 2 | **VIOLATED** | SATISFIED (appearance indistinguishable for some requests) | **VIOLATED** |
| (0,1,1) | 2 | **VIOLATED** | SATISFIED (appearance indistinguishable for some requests) | **VIOLATED** |
| (0,1,2) | 3 | SATISFIED | SATISFIED | SATISFIED |

**Important distinction:** a two-control candidate may appear visually sufficient using just side source views because edit2 is hidden there; that does **not** satisfy the independent source-address edit contract. All four views expose the missing capability. On V3/V5, the 8 physical edit states contain **4 tolerance-compatible state pairs**; on all four camera views, **0 compatible state pairs** remain. The court computes the full source pairwise observation table independently, not from the producer summary.

The *minimum number of independent controls is three for these three required independently addressed toggles*. It is **NOT a proof of three drawable objects**. Multiple representations with three independent controls survive the finite edit contract:
- one addressed aggregate drawable with internal masks;
- several drawables sharing the same mechanical carrier;
- per-face packaging with UV masks;
- a hybrid mechanical-carrier / graphics-control representation.

They are logically equivalent **at this limited edit-operator interface**. Their physical runtime cost, texture memory, draw calls, interactions with deformation and attachment are NOT benchmarked. One drawable is not equivalent to one edit control, and one drawable is not necessarily zero mask-addressing cost.

## B. Original Triceratops source compositing capability

Requirement: original source FULL and all three independently physically edited FULL appearances, V1/V3/V5/V7, eight exact Cycles source samples per view, nonbinary alpha outside this scope, original tie abstention and inherited **0.025 max premultiplied RGBA** tolerance. Compared three source palette groups with ALL six global fixed orders, optimistic camera-specific selection of fixed order, and same-source-sample geometric local depth. Full/source physical targets only used for independent scoring.

| One global flat palette order (front-to-back) | Failed source sample-operation incidences |
| --- | ---: |
| 0 → 1 → 2 | 25,176 |
| 0 → 2 → 1 | 34,472 |
| 1 → 0 → 2 | **25,160** |
| 1 → 2 → 0 | **25,160** |
| 2 → 0 → 1 | 34,472 |
| 2 → 1 → 0 | 34,456 |
| Source sample-local depth ordering | **0** |

Even an **optimistic separately selected flat order for each camera** leaves `5,868 + 4,772 + 4,836 + 5,904 = 21,380` failed sample-operation incidences. Each camera contains source geometry winner-order reversals; the source-local precedence relation cannot be made a valid single flat order under the measured scope. Local depth sample ordering produces zero tolerance violations on qualified physical source samples across all 4 source arms/cameras. Exactly **2** source sample positions remain UNKNOWN from the original source depth gate; they were NOT forced into PASS.

**Capability-relative conclusion:** reject the class *three source palette surfaces + one fixed total compositing order*, even with one order per camera. Preserve the class using original measured source sample-local depth, as well as alternative addressed/merged structures capable of producing the same fragment-depth ordering. This **does not establish that the artist intended a particular cut or three drawables**, and it is not a universal 2D sprite layering impossibility result. A different drawable policy or richer compositor may pass.

## C. Independent decision court and costs

The independent physical court output is `RealSaS.IndependentDrawableDecisionCourt.v2 / PASS`. It independently recomputes **256 Cat physical state-pair comparisons** and **112 Triceratops compositor arm comparisons** from the sealed source samples, checks every reported source measurement/counterexample and preserves the authority boundary. The separate 111-test job covers finite control-state enumeration, flat-order mathematical properties, alpha/unknown gates, nontransitive empirical compatibility, cost-policy separation and tamper redteam. The source architecture impact job and canonical authority job separately passed.

Under a **policy example with cost axes** `logical_drawables, graphic_controls, source_edit_addresses`, the Pareto frontier contains tied-controls `012`, `addressed_aggregate`, `hybrid_carrier_graphics` (equal structural cost vector). These are **illustrative structural bookkeeping axes only**, not measured performance and not a generic product-optimality theorem. Runtime latency, edit UI complexity, mesh topology, source fidelity under pose and GPU cost remain unmeasured. With no declared cost ranking all five satisfying Cat edit representations remain viable.

## Identifiability limit, hard boundaries and next experiment

The source physical experiments observe source rendering/edit consequences, not an authored artist-part label. In a representation class with sufficiently expressive addressable masks and the same per-fragment mechanics and compositing, splitting and repacking drawables while preserving the fragment stream cannot be observationally distinguished under these limited operations. The correct output is a **set of feasible, potentially policy-ranked representations**, not invented artist ownership.

**Missing and not yet qualified:** original source pose2 edit identity/appearance persistence, hide/reveal on changed geometry, independent attachment/release, multi-axis deformation and secondary motion, arbitrary dynamic order override, nonbinary alpha, general new unseen subjects and IRIS one-image inferability. Full puppet contract surviving Cat candidates = **UNKNOWN**, not SATISFIED. Output original `same_drawable_pairs=[]`, `different_drawable_pairs=[]`, all model/owner/training/materialization permissions FALSE.

**Next discriminating trial, preregister before runtime:** select TWO representation classes that agree for today's RGB-edit/render contract but make different predictions for a source-backed **pose/attachment/hide-reveal operation**, including a deliberately limited one-graphic-control variant and a richer addressable-mask/control variant. Declare explicitly permitted operations (whether mask deformation, per-fragment visibility, dynamic relative order and attachment handoffs are allowed) so a failure of a restricted implementation is not misreported as a necessary drawable cut. Use same original mesh/rig/source authority and separately implemented independent physical source consequence court. If both classes remain observationally equivalent under the expanded capability contract, preserve the equivalence and report policy-only costs; if either fails, retain exact source intervention/camera/sample counterexamples. **No owner truth or training promotion** until a source-independent qualification contract authorizes that family.

## Handoff and execution

Files:
- `tools/corpus/drawable_capability_decision_v2.py` — generic finite decision core, policy Pareto and UNKNOWN.
- `tools/corpus/measure_drawable_capability_decision_v2.py` — source-private retrospective physical candidate comparisons, source SHA pins.
- `tools/corpus/court_drawable_capability_decision_v2.py` — independently recomputing court.
- `tests/corpus/test_drawable_capability_decision_v2.py`.
- `.github/workflows/drawable-capability-decision-v2.yml`.

Branch `research/drawable-capability-decision-v2`, PR #354 remains DRAFT on PR #351. Nothing here changes PR #351 data, `main`, the 370 final truth matrix or training authority.

**Operational caveat:** Hosted private transport emits a warning that rclone's shared Google Drive client ID will be retired during 2026. This did NOT block the PASSED run; credential/client configuration should be migrated to a user-owned Drive OAuth client before the shared ID is disabled, without committing secrets.
