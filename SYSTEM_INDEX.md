# RSS.vNext · one active production architecture

**Executable source of truth:** [`platform/internal/capability/contracts/product_capabilities_v1.json`](platform/internal/capability/contracts/product_capabilities_v1.json). This index describes that catalog and its code at the same commit; it is **not** a second manifest or approval authority. Legacy designs and experimental model variants may be stored for reproducibility but are **not** product providers. The one-product graph is enforced in [the producer/consumer regression](tests/repository/test_product_producer_consumer_audit_v1.py).

## Exact product path

```text
Two original uploaded image bytes (source CAS; pose/cameras untrusted)
  -> Input v3: preserve original files, normalize RGBA for model observations
  -> One pinned RSSVNEXT product ModelRelease / bundle
       -> Stage A: 2 predicted canonical RGB/A views
       -> Stage B: 6 predicted RGB/A + aligned normal views, virtual cameras
       -> One IRIS scene decoder / H(x):
            6 generated-view camera-aware tokens
          + 2 ORIGINAL view camera-free tokens (direct source conditioning)
            -> same shared three triplanes, not a second carrier
       -> IRIS: geometry, semantic strata/relations, graphics evidence
       -> TESSA: indexed mesh/topology candidate
       -> AXIS: typed rig/control/constraint/anchor candidate on that mesh
       -> MIRA: hard-FSQ skin and vertex influences on that rig/mesh
  -> Compiler: Mesh -> Rig -> Skin -> Mechanics -> Graphics + Motion -> Proof
  -> only independently admitted ProductRevision -> editable .rss / Studio
```

**Counting convention:** exactly **two observed source images**. The canonical and six-view images are **generated hypotheses**, not eight source observations, measured cameras or 3D truth. Product models do not synthesize cameras for uploaded source rays. The source upload slots remain untrusted labels, not verified front/back capture poses. The generated six-view route uses both observed source tokens **directly** inside the same scene memory as the generated views. A checkpoint lacking `RSSvNextConfig.original_source_direct_h_conditioning=True` is incompatible with this graph; the loader uses exact config keys. [Model](models/rss_vnext/unified_character_model.py), [scene fusion](models/rss_vnext/realsas_scene_conditioning_v1.py), [IRIS core](models/rss_vnext/experts/iris/vnext/scene_core_v1.py), [regression](tests/models/test_rss_vnext_direct_original_source_H_v1.py).

**One model vs three training stages:** `RSSVNEXT` means **one externally selected ModelRelease and one reconstruction carrier**, *not* one monolithic optimizer for everything. The immutable product bundle contains two separately learned diffusion construction stages plus the jointly represented IRIS/TESSA/AXIS/MIRA reconstruction network. Neither denoiser, nor AXIS, nor MIRA becomes an additional independently selected product ModelRelease. Source-to-stage dependencies and weights remain cryptographically bound. See [bundle](models/rss_vnext/realsas_product_neural_bundle_v1.py) and [shape worker](compiler/rss_vnext/workers/rss_vnext_shape_executor_v4.py).

## Single ownership boundary

| Active role | Real responsibility | What it must NOT claim |
| --- | --- | --- |
| `product.input` | Original image CAS and normalized neural observations | Calibrated source pose/cameras from uploads |
| `product.shape` | The one bundle inference; typed IRIS/TESSA/AXIS/MIRA evidence fanout | Canonical ID or independently qualified source truth |
| `product.mesh` | TESSA surface candidate -> source/static qualification -> `CanonicalMeshIR` | An arbitrary learned mesh already accepted as a puppet |
| `product.rig` | AXIS program anchored to that canonical mesh -> `CanonicalRigIR.v2` | Compiler-invented parents/controls/constraints; extra AXIS release |
| `product.skin` | MIRA hard-code replay on exactly that mesh/rig -> `CanonicalSkinIR.v2` | Hidden skin transfer, code search, extra MIRA release |
| `product.mechanics` | IK/FK, limits, deformation, declared sockets/contact candidates and evidence | Learned secondary/contact physics guaranteed from static inputs |
| `product.graphics` | **Sole authority for final 2D RGB/alpha, visibility, depth/order and presentation** | Semantic strata or mesh itself automatically proving edited drawables |
| `product.motion` | Control-space clips/retarget and canonical motion | Rest hold being called a learned Idle/Walk |
| `product.proof` | **Sole PASS decision owner** after independent runtime and source replay checks | Automatic ProductRevision promotion from candidate-stage success |

`Semantic` is StdGEN-style layered 3D surface/presentation evidence, **not** a cape/hair object-class ontology and **not** final Drawable ownership. `Drawable` is a separately addressable editable representation compiled using qualified evidence; it need not match a 3D component or XPBD domain. Graphics owns final visible 2D pixels; Mechanics executes admitted physical control; neither steals the other's authority.

The active catalog selects the nine roles above and their exact module entrypoints. The older calibrated source path and research S-LRM stages are compatibility/research interfaces, **not a second active uncalibrated product DAG**. Do not merge an old branch simply because its source code exists. Original expert package aliases point to the unified source tree, not four independent products. See [models index](models/README.md).

## Training is one product, with qualified staged supervision

- **Clean-room generator stages** learn from independently admissible source-native 3D renders. Generated views condition reconstruction but never become source ground-truth labels.
- **Shared reconstruction:** active [structural trainer](models/rss_vnext/training_engine.py) supports independently gated geometry, topology, appearance, semantics/relations, AXIS and MIRA-related loss families. Current ordinary optimizer remains a **fixed weighted sum with one backward**. The gradient-interference court measures selected owners; PCGrad/MGDA/CAGrad are **two-task research variants**, not an adopted full-unified optimizer.
- **MIRA codec-first phase** learns source-authorized skin compression/deformation with the relevant upstream frozen; after that AXIS predicts causal skin codes. Separate training stages share the **same eventual shipped decoder/weights**, not multiple inference authorities.
- **Source-posed consequence (opt-in):** on a source-reindexed TESSA mesh and AXIS joint domain, the student hard-FSQ MIRA output is actually posed under independently admitted source joint transforms and compared with source surface samples. Idle-only or unqualified motion cannot teach this. **Teacher-matched learning is not teacher-free performance.**
- **No accidental epistemic upgrade:** no GT from generated RGB/normal views, no open-mesh signed-solid label, no attachment inferred solely from proximity, no dynamic physics label from appearance class, and no missing-label-as-negative trick.
- **Curriculum and loss routing are not product-qualified.** A recorded stage, loss norm or synthetic gradient ablation is not evidence of real-corpus generalization.

Current source-code deep comparison: [trainer audit](docs/research/RSS_VNEXT_TRAINER_SOURCE_CODE_LOSS_AUDIT_20261115.md). Source research and old models are preserved but cannot override this production catalog.

## Product admission (still OPEN)

Engineering integration and CI passing **do not mean the product is proven or deployable**. The executable ProductProof presently fails closed. Outstanding independent qualifications include trained and pinned checkpoints on a rights-approved corpus, unseen two-image -> usable deformable mesh/rig/skin evidence, native 2D pixel/visibility and hidden-region behavior, Graphics renderer parity, motion/attachment/contact where claimed, and edit/save/reopen of the actual .rss puppet. Automatic secondary motion and full-body mouse control are **not prerequisite release claims**; their experimental solvers must not silently enable physics in a production puppet.

The product objective remains **two image inputs -> compiled editable puppet -> Studio -> edit/save/reopen**. Video, sprites and Spine exports are secondary outputs. The first scientifically meaningful milestone is animation-suitable mesh + rig + skin from previously unseen sources, with independent appearance replay.

Run `python tools/repository_clarity_audit_v1.py --json` for a read-only repository/current-readiness snapshot. Source-code presence and a green test suite cannot create `ProductRevision`.
