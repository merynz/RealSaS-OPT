# Graphics Evidence Maturity Audit — v0 (6 Nov 2026)

**Research state:** PR [#356](https://github.com/merynz/RealSaS-OPT/pull/356), stacked on [#355](https://github.com/merynz/RealSaS-OPT/pull/355) → [#354](https://github.com/merynz/RealSaS-OPT/pull/354) → [#351](https://github.com/merynz/RealSaS-OPT/pull/351). Draft only. **Not a general Graphics solution and not training-authority promotion.**

## 1. Architecture evidence inspection (actual code, not design assumptions)

| Existing consumer / producer | What exists | Current scientific limitation |
| --- | --- | --- |
| `compiler/realsas_compiler_core/graphics_v1.py` | `GraphicsEvidenceV1` learned affinities, fragment union via thresholds, view order graph, candidate attachment | `_dag` can drop lowest-confidence cycle edges. This is a **policy-relative learned Graphics proposal**, not source-authored drawable truth and not proof that a total draw order exists. |
| `graphics_evidence_v2.py` + `graphics_v2.py` | Source-scoped canonical barycentric ray hits with Registry-admitted row hashes; ray-depth constraints and explicit local interleaving | Only observed scope is qualified. Alpha, missing hits, unsupported authored order and exact ties retain UNKNOWN. No artist owner/whole-puppet synthesis. Importantly, `edges_removed=0` in v2 scope records. |
| `graphics_runtime_v1.py` | Immutable face→fragment index and per-view topological order from an existing compiled graph | Materializes a graph; does not independently prove that learned fragments are physically needed or that a fixed order survives source motion. |
| `drawable_surface_v1.py` | Qualified geometric drawable surface + rigging support coefficients | A conditioned render/deformation surface **is not** an independently addressable artist or operational drawable partition. |
| `visibility_v2.py` / `appearance_render_v2.py` | Canonical posed 3D depth K4 and source-addressed CAA reference renderer (2×2 coverage samples), premultiplied source-over | A real **reference execution kernel**, not original Cat/Triceratops Cycles parity. Alpha support is limited by available teacher observations, exact-depth tie and layer overflow qualification. |
| `runtime_projection_v1.py` | Stage-36 Runtime V4 product projection with exact mesh/presentation/appearance/composition/motion binding and face-corner UV seam expansion | Product path exists; no generic source-evidence→qualified graphics owner synthesis proven. Enabling v0 as an authoritative stage would be premature. |
| `tools/corpus/compile_corpus370_graphics_evidence_ir_v1.py` | Original V18j ray/coverage/residual/observability evidence and independent source court | 180 source pixel observations across Cat/Triceratops; **no** source owner labels, edit consequence partition, or full camera/pose coverage. |
| `compile_corpus370_v18r_source_world_graph_evidence.py` | Source-world pose 0/1 normals, geometric gaps and pose2 court-only heldout constraints | UV/material continuity and actual drawable relationship explicitly unobserved. Vertex gap is not surface continuity or artist part identity. |
| `compile_corpus370_source_cat_uv_consequence_ledger_v1.py` + V3 pose court | Physically perturbed linked UV appearance and native authored pose edits | Finite source observation. Three edit addresses ≠ three drawables; source-native actor ≠ compiled BUSTED puppet. |
| Triceratops source K3 + V2 decision court | True local depth vs all six constant palette flat orders | Refutes **specified fixed global palette order** in measured domain, not sophisticated merged address/geometry alternatives. |

## 2. Source facts are not a single maturity Boolean

| Source operation requirement | Physically established scoped result | Still absent for generic puppet |
| --- | --- | --- |
| Independent atlas editing | Original Cat V2 [256 source renders](https://github.com/merynz/RealSaS-OPT/actions/runs/42263195370), three anonymous top-left UV rectangles; original joint effects independently courted | General editable control extraction, arbitrary source addresses, nonbinary alpha; independent control count is not drawable count |
| Pose-aware source UV | Original Cat V3 [64 source renders](https://github.com/merynz/RealSaS-OPT/actions/runs/42273572984), two authored poses (0, 14), four cameras, two sample offsets; frozen-screen hypothesis has 40,878 failures | Native BUSTED parity on source pixels; additional poses and morphologies; single image inference |
| Per-sample local depth compositing | Original Triceratops K3 [decision court](https://github.com/merynz/RealSaS-OPT/actions/runs/42269086950) gives local-depth 0 source threshold violations vs best flat global order 25,160 in observed scope; two source pixels UNKNOWN | Generic compositing/cost proof; all poses, complex alpha, independent actual graphics runtime |
| Hidden edit state & reveal | Reference puppet synthetic two-layer depth swap passes | **Original-source occlude/reveal with edit while fully hidden** has not been independently courted |
| Attachment/release | Source mechanical/contact constraints exist | Independent source attachment/release operation and graphics consequence UNKNOWN |
| Secondary motion | Authored source vertex movement may be observed | Isolated secondary control and its independent graphics consequence UNKNOWN |
| Nonbinary alpha parity | Cat source texture alpha entirely opaque, source alpha unchanged by edits | Nonbinary alpha edit, compositing and provenance truth UNKNOWN |

**Crucial:** the Cat V2, Cat V3 and Triceratops rows do **not** automatically form one source-character/scope complete contract. Different subjects, poses, camera sets, shaders and edit workloads cannot be silently pooled.

## 3. New machine-readable gate (PR #356)

The generic `graphics_source_readiness_v0.py` now records each operation's exact source-fact obligations, checks source+scope+artifact+record+independent-court digest binding, excludes learned/proxy/policy/reference-fixture claims from source admission, and prevents facts from *different* scopes from completing an operation.

- No claim/unknown/partial source record → `UNKNOWN` plus a list of missing facts.
- Verified original-source records from one coherent scope → `OBSERVED_IN_ADMITTED_SOURCE_SCOPE` **only for that observed operation**.
- An original-source observed fact never sets `SATISFIED` on the *compiled puppet*, never mints artist ownership, and never authorizes IRIS/TESSA/training.
- A caller must build the admitted-artifact map from **independently reopened source bytes and court receipts**. Merely instantiating `GraphicsAdmittedArtifactV0` is NOT that independent verification. Until the real adapter succeeds, v0 is a contract/tested boundary, not source qualification.
- Teacher-source claims are forbidden from `STUDENT_INFERRED_UNQUALIFIED` and `SYNTHETIC_FIXTURE` synthesis inputs. The source-specific Cat adapter is intentionally **outside** generic synthesis.

The synthesis output now distinguishes `reference_executor_capability_states`, source-observability, `capability_states` (still `UNKNOWN` absent independent source puppet parity), reference-executable candidates and `source_qualified_candidate_ids` (empty).

The original Cat V3 adapter `tools/corpus/admit_corpus370_original_cat_v3_graphics_readiness_v0.py` performs an independent re-court of the existing SHA-locked V3 source arrays, then admits only source pose/UV consequence **if** exact court bytes and physical source parent SHA matches. Its private hosted workflow is `graphics-source-readiness-original-cat-v0.yml`. **A successful source-readiness CI, if obtained, is not BUSTED physical parity**.

## 4. Genuine graphics closure: what remains

1. **Evidence source→canonical alignment:** source-original face, UV atlas, effective alpha and pose/camera scopes must have independently qualified canonical bindings. V18r geometry proximity alone is not adequate.
2. **Operational consequence identifiability:** original source edit-with-occlusion/reveal (at least one pose/view control), plus separate attachment/release and secondary-motion interventions where supported by the source. Pre-register explicit controls and matched counterfactuals; isolate real capability consequences.
3. **Strong candidate qualification:** compare source-addressed aggregate/packet/face-stream models **as actual distinct executors or render scheduling policies**. Today all four packings use the *same* CAA executor and therefore demonstrate an invariant/equivalence, not a winner.
4. **Source-private independent court:** test a compiled puppet's rendered output against original physical source targets at frozen pose/view/sample/edit protocol; declare renderer/color/alpha sampling mismatch instead of retuning without a new preregistration.
5. **Heldout morphology:** no source-case threshold or subject-specific split heuristic. Count observed vs masked vs unknown by family, not naive global PASS.

### Hard status
- Source artist drawable labels: **UNKNOWN**.
- Operational drawable split necessity beyond limited controls: **UNKNOWN**.
- Runtime physical source parity: **UNKNOWN**.
- BUSTED shipping Graphics and one-image inference: **UNKNOWN**.
- Corpus370 family training/materialization, IRIS/TESSA admission: **FALSE / unchanged**.
- Runtime GPU/CPU performance, memory and draw-call benchmark: **UNMEASURED_COST**.
- Main and Corpus370/IRIS/TESSA operational pipeline: **unchanged**.

## 5. Validation log

- [v0 reference initial CI](https://github.com/merynz/RealSaS-OPT/actions/runs/42278139489): 68 PASS (synthetic reference + existing V2/V3 regressions).
- [v0 source-readiness and nonleakage CI](https://github.com/merynz/RealSaS-OPT/actions/runs/42280497990): **86 PASS** (exact committed SHA and log).
- Cat private V3 source requalification workflow: **check latest result before claiming PASS**.
