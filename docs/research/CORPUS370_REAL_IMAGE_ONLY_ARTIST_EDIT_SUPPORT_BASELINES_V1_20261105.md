# Corpus370 — real image-only artist edit-support baselines (2026-11-05)

**Status:** 3/3 GitHub-hosted original-artist export experiments SUCCESS, 14/14 focused adversarial tests PASS in each job, ownership UNKNOWN, training authority FALSE.

Experiment: https://github.com/merynz/RealSaS-OPT/actions/runs/42064435486
Machine-readable measured receipt: canonical/CORPUS370_ARTIST_IMAGE_ONLY_EDIT_SUPPORT_REAL_RESULT_V1.json
PR: https://github.com/merynz/RealSaS-OPT/pull/333 (draft, stacked on #327)

## Experimental design

Original Spineboy, Owl and Raptor export JSON, atlas and original PNG assets were SHA-256 verified and replayed through the official Spine CanvasKit runtime. Only each resulting 768x768 original flattened RGBA composite plus width, height and image SHA was passed to the prediction subprocess. There were no original slot names, counts, bone names, source mesh faces or original edit masks in the algorithm input.

Two **nonlearned image-only baselines** were frozen: foreground alpha four-connected components, and Felzenszwalb RGB graph regions with scale=180, sigma=0.8, min_size=140. Neither baseline knows the intended number of drawable parts or any semantic/body part class.

The separately invoked original-artist court opened the previously hidden SHA-bound native runtime hide/isolate RGBA files only after the prediction bytes had been sealed. It compared actual composite pixel differences when original source attachments were hidden with each baseline's predicted image regions. Hungarian one-to-one matching was used to calculate visible edit-support IoU and recall. This is **not original artist layer-mask IoU or true editable part accuracy**: a hide effect may expose a different underlying part, overlap other effects or be entirely invisible.

## Real measured results

| Source | Original active attachments | Visible hide edits | Masked hide edits | Alpha connected regions | Alpha edit-support IoU | Adaptive graph regions | Graph edit-support IoU | Graph edit-effect recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spineboy | 18 | 18 | 0 | 1 | 0.01697 | 150 | 0.27867 | 0.25486 |
| Owl | 25 | 22 | 3 | 2 | 0.01646 | 136 | 0.41776 | 0.35478 |
| Raptor | 35 | 33 | 2 | 2 | 0.00955 | 166 | 0.29251 | 0.32170 |
| Total (counts only) | 78 | 73 | 5 | 5 | — | 452 | — | — |

The alpha-only baseline undersegments: 1–2 visible connected regions cannot represent 18–35 original individually hideable attachments. Conversely, a naïve image graph finds 136–166 regions, but still gives weak edit-effect recall. Its moderate matched IoU can be rewarded by overfragmentation, so it is scientifically invalid to call Owl 0.4178 a 41.78% drawable ownership accuracy.

Five original hide edits are not observable in the composite (3 Owl, 2 Raptor); their original source artwork remains present when isolated. A single source image cannot be treated as negative truth for these masked parts.

## Scope and limitations

- These are original Spine **export** assets; the editable native project file and model-training/redistribution rights are not verified. All original third-party art pixels and private masks stayed ephemeral in hosted runners.
- Inference and court are separate CLIs with SHA-verified prediction output and restricted explicit readsets. However, they currently run sequentially on one filesystem. An adversarial-grade gold benchmark will require physical process/filesystem isolation or independently controlled input transfer to prevent malicious code reading nearby private files.
- This is not image-to-3D geometry, a learned IRIS/TESSA target, a mechanical ownership certificate or an editable puppet compiler. It cannot promote SAME_DRAWABLE / DIFFERENT_DRAWABLE pair labels, alpha CAA or draw-order authority.
- The single setup pose has no heldout action, deform, layer swap or source-edit compositing from RealSaS. Native rig controls measured by V13b/V14 remain teacher-only and must not be laundered into image-only student inference.

## Next measured scientific gate

Freeze an actual generic split/merge optimizer **before** opening independent artist labels, and measure against the same baseline with complexity regularization, region count, matched edit support and graph fragmentation. Evaluate on heldout original authors and source edit states, preserve opaque/masked regions as UNKNOWN and prove exact alpha/occlusion behavior. Join the valid causal geometry-to-RGBA evidence from V14 only as a separate teacher/court signal, never as a source object identity prior in the image-only algorithm.

The desired decisive measurement is not that we can manufacture layers; it is that a precommitted partitioning method recovers independent editable consequences without artist-authored boundaries leaking through mesh topology or semantic labels. Qualified geometry-only training remains independently permissible when its exact family court and roster are ready.
