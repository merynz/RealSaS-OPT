# Corpus370 original Cat UV edit V1 FAIL and glTF top-left V2 amendment

2026-11-06 · PR #351 · research source address correction. Original source physical V1 run [42259366611](https://github.com/merynz/RealSaS-OPT/actions/runs/42259366611) remains **FAIL** and is NOT retrospectively recertified.

## Immutable V1 failure

256 real original Cat source Cycles frames rendered at V1/3/5/7, eight sample subsets per camera, eight separate source-texture edit arms. Real original source Cat was visible (each source view >145,600 alpha-covered sample-pixels). Every separately edited original atlas image caused **zero actual visible source RGB change** in all four camera views. Independent V1 court rejected all three edit operation non-vacuity gates. The predicted pair/joint algebra having zero errors was VACUOUS, not a success.

[Private original V1 independent FAIL court](https://drive.google.com/file/d/1SlP82GBWjxtTlXPeQHl5kLVrU-A_BMUF/view) SHA 6cdd432ce3a3ceb64d9c88117847f99336d33d9891b483a95c8bbb2d0fa8b186. Full original 256 source frames and all original atlas edit bytes were privately archived and SHA verified. All owner and training flags remain FALSE.

## Independent root cause: source glTF V-axis convention

Normative external reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html (Texture Coordinates section). glTF 2.0 uses **image TOP-LEFT (U0,V0) origin**. V increases DOWN in the PNG. The archived original GLB has TEXCOORD_0 in glTF coordinates. Blender internally may invert to match Blender UV convention, but the pixel edit tool operates on ORIGINAL PNG bytes.

Original V1 plan wrongly mapped source UV V tile to PNG Y tile (7-V), editing untouched/black PNG regions rather than addresses actually sampled. Original SHA-locked source PNG directly confirms:

| Source GLB UV tile chosen before rendering | Correct glTF top-left PNG source region nonblack texels | Incorrect V1 vertically mirrored PNG region nonblack texels |
| --- | ---: | ---: |
| U6,V6 | 4096 | 0 |
| U6,V7 | 4096 | 0 |
| U7,V6 | 4096 | 0 |

Source UV selections remain original-only triangle-centroid top three: (6,6) supports 292 source face centroids, (6,7) 246, (7,6) 63. Original source GLB SHA 1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850, PNG SHA 2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d, source policy/cameras unchanged.

## Fresh V2 preregistration before rerender

Version a new UV plan, physical source atlas edit asset manifest, original Cat same-Cycles physical render and independent court. Only correct raw glTF UV image Y address to Y=V*64 rather than old (7-V)*64, and bind a new plan SHA. Same three selected UV tiles, same original source, alpha, 64x64 texel edit, source unlit-alpha policy, V1/3/5/7, 8 Cycles sample offsets, eight single/pair/joint physical arms, same 0.025 max premultiplied RGBA threshold, same >=32 visible edited source pixel sample nonvacuity gate. All original scene and image assets immutable.

V2 scientific PASS would certify **source Cat linked atlas edit superposition under SAME Cycles sample**, not source art drawable owners, IRIS single-view inferability, arbitrary texture edits, multi-pose consistency or Corpus370 final training authority.

The V1 physical FAIL remains independently accessible forever.
