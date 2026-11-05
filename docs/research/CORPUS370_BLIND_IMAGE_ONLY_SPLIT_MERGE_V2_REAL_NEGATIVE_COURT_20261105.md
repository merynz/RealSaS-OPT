# Corpus370 — blind image-only generic split/merge V2, genuine negative court

2026-11-05. Research code draft PR #334, based on #333, independent of active Corpus370 #327. First full original Spineboy/Owl/Raptor experiment: https://github.com/merynz/RealSaS-OPT/actions/runs/42070010839. Machine-readable primary evidence: canonical/CORPUS370_ARTIST_IMAGE_ONLY_SPLIT_MERGE_REAL_RESULT_V2.json. Three source jobs SUCCESS, each ran 35/35 tests successfully on the accepted run; all original art and private edit masks were ephemeral.

## Objective and strict model input

Infer candidate edit regions from only the original flattened 768x768 RGBA, not a source mesh, skeleton, actor rig name, artist drawing attachment, native source editor project, artist part count or private edit court mask.

The visual split phase applies two frozen RGB graph scales to the original single composite. It then explicitly divides any graph label spread across disconnected foreground islands. The generic merge phase greedily fuses only adjacent four-connected foreground regions with a fixed image-only cost, combining original pixel-boundary RGB difference, mean region RGB disparity and normalized Ward mean-color difference. The three conservative/adaptive/fine policies and thresholds were committed before observing their original-art court scores, with no target number of parts. SHA-locked V1 prediction and V2 candidate output are independently validated before private court evaluation.

## Result table — same three original author sources

| Source | Visible / masked hide edits | V1 graph labels | V1 support IoU / recall | Conservative V2 regions | V2 IoU / recall | Adaptive regions | Adaptive IoU / recall | Fine V2 regions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spineboy | 18 / 0 | 150 | .2787 / .2549 | 381 | .2178 / .3683 | 165 | .1809 / .3621 | 285 |
| Owl | 22 / 3 | 136 | .4178 / .3548 | 111 | .3787 / .5202 | 42 | .2391 / .4684 | 144 |
| Raptor | 33 / 2 | 166 | .2925 / .3217 | 461 | .2707 / .3885 | 219 | .1335 / .3554 | 457 |

No precommitted V2 method improved original artist-edit-support mean IoU on any of these three source examples. The conservative method improved matched changed-pixel recall in 3/3 examples, but IoU regressed in 3/3, and Spineboy/Raptor overfragmented compared with V1. This is a negative result, not drawable reconstruction or model quality PASS.

### The unexpectedly large region counts explain an important comparability trap

Original V1 image_graph used 150 / 136 / 166 **raw graph IDs**. When V2 enforced foreground four-connectedness on the same RGB graph, coarse seed regions became 1005 / 813 / 938 respectively. The original graph can connect two separated foreground islands through transparent background, so counting its labels as independent visible connected components is misleading. The V2 conservative policy made 624 / 702 / 477 visual merges, leaving 381 / 111 / 461 regions.

The exact parent graph recomputation assertion and explicitly SHA-verified connectivity audit added after the first result distinguish true source-mask topology from render variation or silent algorithm-parameter drift. Important: actual artist-authored drawables MAY legitimately contain disconnected visible islands; four-connectivity is only an image-segmentation proposal policy, not a canonical owner rule.

### Why static color merging failed

Image color adjacency is neither necessary nor sufficient for common editable ownership. One artist attachment can span multiple texture and shading regions; separate editables can share colors and even overlap. Hiding a source component often exposes lower layers, meaning its visible hide-effect mask is not identical to its own art membership mask.

This court is one-to-one matching to original visible hide effects, with observability explicitly recorded. It does not prove true artist layer membership, alpha compositing, depth/draw-order, IK or an editable puppet. Five original source attachments have masked or unobservable hide effects in the starting image. They must stay UNKNOWN; absence of visible image consequence is not absence of authored artwork.

No original editable native artist project file or training license verified; official original runtime exports were consumed transiently as research reference only.

### Next falsifiable stage

Freeze these three examples as a source-diagnostic set; do not tune threshold parameters against the hidden original artist edit references and relabel them heldout success. Obtain disjoint original authored artists/morphologies and distinct original edit poses, create genuine heldout creator/family partitions and physically isolated input-vs-court operation. Develop a generic behavior-conditioned ownership model incorporating occlusion, repeated edit response and source appearance alpha authority without leaking source bone/object/attachment labels into the student. V13b/V14 source causal mechanics is complementary court-only proof and does not equal image-derived drawable identity.

Authority remains FALSE for artist SAME/DIF pairs, image-to-3D source mechanics ownership, artist drawable recovery, native CAA/draw order, model training, materialization and product compile. Tests passing means our negative measurement is reliable, not that product-level ownership is solved.
