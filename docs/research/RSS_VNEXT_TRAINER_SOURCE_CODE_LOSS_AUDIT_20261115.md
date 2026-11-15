# RSS.vNext unified trainer — source-code comparison and staged loss contract (2026-11-15)

**Status:** source-code audit + one opt-in training-path change. **Not** a calibrated production training recipe or unseen validation. Architecture remains one RSS.vNext checkpoint, one IRIS H(x) carrier, TESSA/AXIS/MIRA expertise. No new image input or independent model release.

## Inspected *actual upstream implementations* (not papers alone)

- **StdGEN** `hyz317/StdGEN`, `slrm/models/lrm_mesh.py`: `forward_geometry()` first extracts a geometric mesh via FlexiCubes and reprojects exactly that geometry to render `img`, `semantic`, `mask`, `depth`, `normal`. `get_sdf_deformation_prediction()` restricts learned deformation with tanh, extracts semantic layers via SDF operations, and stops geometry gradients for full-positive/full-negative unqualified SDF. `refine/mesh_refine.py`: surface normals, rendered alpha and expansion losses are used in subsequent optimization. **Public repository is inference/refinement oriented; no full S-LRM original training loss implementation was verified.** Do not backfill imaginary trainer weights. Do not reinterpret StdGEN geometric semantic layers as BUSTED Drawable truth.
- **AniGen** `VAST-AI-Research/AniGen`, `anigen/trainers/vae/anigen_slat_mesh_vae.py`: `geometry_losses()` renders both source GT and predicted mesh and compares silhouette mask, foreground depth, normal perceptual losses; only successful extractions contribute to these losses. `skeleton_losses()` scores joint and parent fields with nearest/second-nearest distance weighting and confidence, not an unconditional hard skeleton token assignment. Skin matching can use BVH nearest-surface triangle barycentric interpolation to readdress source weights to predicted mesh vertices; invalid skin labels are not scored as true zeros. `anigen_skin_ae.py` area-weighted face sampling + KL skin loss. `anigen_flow_matching.py` jointly predicts shape/skeleton velocities in a multi-stage system; this is **not** an argument for replacing RSS.vNext's carrier or training method with flow matching.
- **SkinTokens** `VAST-AI-Research/SkinTokens`, `src/model/tokenrig.py`: pretrained Skin VAE frozen during TokenRig training, causal skeleton+skin token generation, explicit `skin_warmup_start_epoch/end_epoch` cosine ramp. `src/model/skin_vae_model.py` uses dense skin measurements. The prediction contract is skeleton/skin, *not* source-observed interactive mechanics. Use its codec-first sequencing lesson, not a false complete puppet-program claim.

Primary references: https://github.com/hyz317/StdGEN , https://github.com/VAST-AI-Research/AniGen , https://github.com/VAST-AI-Research/SkinTokens . Their research code and weights are **not** vendor-copied into RSS.vNext.

## Active RSS.vNext objective map (actual `models/rss_vnext/training_engine.py`)

| family | teacher / source authority | student pathway | unresolved risk |
| --- | --- | --- | --- |
| topology | source-native TESSA grammar tokens | TESSA causal token CE | sequence fit != animated mesh quality |
| motion_preference | source-qualified ranked topology candidate court | TESSA candidate log probability | discrete candidate population and surface consequence |
| source_semantics / semantic_risk | observed pixel patch known-mask source | IRIS early 2D semantic field | unknown must remain masked, not negative |
| surface_roles | source 3D role labels | H(x) presentation-role head | surface stratum != Drawable or mechanics |
| region_pairs | independently source-qualified region pairs | shared region assignments | co-membership != attachment |
| typed_relations | separately qualified typed pair targets | region relation head | no source truth -> abstain, not pseudo-label |
| graphics_source_rgb | original source first-hit opaque ray | Graphics RGB + direction decoder | color alone != alpha/draw order/hidden appearance |
| source_udf | source-native 3D unsigned distance | IRIS UDF | open mesh cannot teach signed solid |
| source_inside_sign | independent closed solid proof | IRIS sign | source sign MUST be qualified |
| axis_rig_skin_codes | source-native reindexed rig + pinned frozen MIRA FSQ code | AXIS causal grammar / code CE | teacher forcing and code CE != posed mesh correctness |
| **axis_posed_surface (opt-in)** | **qualified source-native LBS-only posed surface + joint transforms** | **actual matched TESSA mesh + AXIS hard-token MIRA skin -> LBS -> surface loss** | **teacher-mesh training; inference parity and real source eligibility remain unproven** |
| anchor_face_correspondence | exact source-native held/holder face reindex | AXIS/TESSA anchored face head | no inferred contact from proximity |
| flexicubes_surface | qualified signed source surface | differentiable FlexiCubes witness | surface witness not TESSA final topology |
| flexicubes_animation | qualified animated source | FlexiCubes posed loss | limited real admissible source coverage |
| novel_view | independently rendered GT views | constructed view RGB/normal | generated views are conditioning hypotheses, not GT |
| realsas_joint_slrm | source 3D rendered staged appearance | shared H(x) semantic/graphics renderer | appearance/semantic rendering must not issue Drawable or physical truth |

**Current optimizer fact:** ordinary qualified losses are summed with fixed weights, one backward; sampled gradient interference is a diagnostic. `training_gradient_routing_research_v1.py` only tests **two-task full-vector** PCGrad/MGDA/CAGrad on synthetic fixtures, and is not active. **Do not deploy a winner based on those tests.**

## Immediate functional addition: qualified AXIS posed surface consequence

New opt-in `axis_posed_surface` family reuses the existing `animation_ready_mesh.source_pose_surface_consequence_loss`; it does not invent a second solver.

- Requires an already source-qualified AXIS teacher sequence and `PRODUCT_MESH_MATCHED_REQUIRED` policy; exact teacher TESSA topology/face identity and embedded MIRA codec must pass existing qualification.
- Requires independently admitted **native**, **LBS-only**, **normalized**, **joint-reindexed** motion samples, with tensor hashes and observed source identity bound in the trusted training launcher registry.
- Rejects malformed/uninformative rigid-only source animation, unregistered or tampered targets, generated-view "motion teachers", and source tensors requiring gradients.
- Students use the **hard FSQ token forward** MIRA skin decoding; gradients flow to AXIS code logits, mesh appearance-independent H and TESSA surface vertex geometry, not through source GT.
- Source surface samples supply both rest and posed Chamfer; the objective does **not** certify exact source material parameters, contact, secondary dynamics or teacher-free topological generalization.
- Default 0.0 preserves prior training updates. Existing v2 checkpoint objective fingerprint stays unchanged while disabled; enabling it changes the digest, and the source registry hash becomes a resume constraint. No source teacher => objective remains unavailable on that batch (never fake a zero-labelled teacher).

**Remaining major gap:** Product uses **predicted** TESSA topology and free-running AXIS tokens. The new consequence is *source matched/teacher forced*, not that product condition. Future accepted training must compare exact source-conditioned versus teacher-free outputs, with independent source-only held-outs, missing data masks, actual mesh deformation and 2D Graphics Proof.

## Proposed future *measured* curriculum, not yet an authorized production recipe

1. Source-native 3D/ray/pose teacher eligibility with mask and lineage reports. No qualified truth => no gradients for that family.
2. Geometry/semantic/graphics joint representation and TESSA structural formation against source surface outcomes, with invalid extraction fail closed.
3. Frozen-upstream, separately learned MIRA codec (existing), with source-native skin and informative posed-LBS loss.
4. Mesh-matched AXIS + frozen codec, then gated `axis_posed_surface` for *qualified* source motion, with separate metrics for each family.
5. Measured shared-carrier conflict routing ablation **on real source truth**; controlled sampling/phase/optimizer/compute; only promote a strategy that improves held-out deformable geometry, mesh validity, source RGB and rig/skin results simultaneously.
6. Teacher-free **two image -> real mesh/rig/skin -> source-qualified pose -> Graphics** acceptance. Passing teacher-token CE or synthetic tiny-regression tests alone does not advance ProductProof.

**Safety/operational constraints:** Do not pretend that the current licensed corpus automatically contains LBS-qualified native posed examples. No actual model training, GPU benchmark, held-out metric, or ProductRevision was produced by this source change.
