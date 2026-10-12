# Knight native motion inspector and source-frame correction

The old sealed SLASH witness genuinely contains Run_swordAttack; it was not a mislabeled RUN. Its local source quaternions were copied into different target rest frames, and omitted animated ancestors such as Shoulder.R lost their motion. The review adapter now samples individual source channels at 30 FPS, evaluates source FK, folds skipped ancestors and transports proper relative skin rotations and orbital translations through explicit source/target rest frames and object alignment. Source-FK orientation residuals are below 5e-15 on the seven native clips. This proves those rotations, not artistic attack semantics or universal retarget correspondence.

## Exact native source

KnightCharacter.fbx SHA256 is `41f2616fb4a42f9d35e242bbed1450ededc7537def4bf38cbb11b4e54ff2bf21`. Its 12 native takes are Death, Idle, Idle_swordLeft, Idle_swordRight, Jump, Roll, Roll_sword, Run, Run_swordAttack, Run_swordRight, swordAttackJump and Walking. The extracted source specification selects Idle, Run, Run_swordAttack, Jump, Death, Roll_sword and swordAttackJump. Blender extraction receipts pin the FBX, selected exact actions and complete inventory. No external animation library is used.

The correspondence is recovered once from the old sealed SLASH by unique equality of complete copied rotation/translation sequences, then reused only after native skeleton topology and rest positions match. This is the bounded existing witness correspondence, not a new universal automatic mapping solve. The old posed vertices never supply the new animation. All clips retain identical rest M, faces, hierarchy, control positions and canonical W.

## Attachment and overlap candidates

The rigid presentation operator finds held and holder triangle addresses with barycentric coordinates. Holder faces are restricted to the two immediate ancestor skin domains of the equipment slot; an unrelated nearer surface cannot become the grip. Slot rotation is retained, while translation transports the held surface anchor to the posed holder surface. Equipment remains rigid and residual contact distance is below 1e-14 world units. Primary mechanical poses and W are retained separately. This contact/rigidity proof does not establish grip orientation, body clearance or all-frame artistic qualification.

An optional kinematic-region policy groups disconnected components sharing a dominant skin control. Sword and shield each remain one region. It sorts region centers per view and pose, then uses geometric triangle depth within each region. It avoids the rejected 19-component order that placed face skin over its helmet. It is an explicit presentation heuristic; there is no temporal, occlusion-semantic or product PASS. Raw geometric ties remain measured rather than disappearing behind presentation ranks. The existing approved RUN default renderer is unchanged.

## Reproduction and artifacts

```bash
blender --background --python tools/motion/blender_extract_motion_source_v2.py -- \
  --source-fbx assets/external/quaternius/knight/KnightCharacter.fbx \
  --spec canonical/MOTION_PRESET_SOURCE_QUATERNIUS_KNIGHT_NATIVE_V1.json \
  --out-dir /absolute/native-source
python tools/demo/render_knight_source_motion_v3.py \
  --source /absolute/native-source/demo_death_v1.motion.json \
  --bind-source /absolute/sealed-original-slash.motion.json \
  --witness /absolute/sealed_witness_replay.npz \
  --material /absolute/single_m_caa_material.npz \
  --output /absolute/renders/demo_death_v1 --workers 8 \
  --fps 30 --duration 4.6 --resolution 512 \
  --attachment-policy slot_anchor --overlap-policy kinematic
python tools/demo/build_knight_investor_package_v1.py \
  --renders /absolute/renders --output /absolute/investor-demo
python -m unittest tests.compiler.test_motion_retarget_frames_v3 \
  tests.compiler.test_presentation_attachment_order_v1 -v
```

The bind source must be SHA256 `6422209f1291dbb3dbfdffb6b238e63214f8aeddfc31e12f1b105ecc1da85298`; the old witness must be SHA256 `bb8cfb075b5ec3df93cc1fac306d65c88c27f5a6a7fc1bbf74c05ec79bcdba35`. Source-owned RGB/alpha and sealed CAA addressing cameras stay fixed. A separate fixed output viewport fits every sampled pose and all eight directions, including Death and Jump. Its pixel resolution is passed explicitly so source-texture resolution cannot rescale the overlay. No character geometry is shifted to achieve framing.

At 30 FPS, Idle shows two source loops (6.667 s), Run six (5 s); Slash, Jump, Sword Roll and Jump Attack use 4.2 s; Death uses 4.6 s. Nonlooping takes preserve native speed and hold the terminal pose; the player stops there and Play restarts it. No artificial loop closure is required. MP4 frame counts and full decoding are checked. GIF is a centisecond-timed review fallback; MP4 and the player provide exact 30 FPS.

START_DEMO.html is a read-only, offline consumer. It computes live LBS from the actual constant W and compiled matrices, applies the compiled rigid surface transforms, and synchronizes source artwork, selectable bone glyphs, depth-tested mesh and weight heatmap on one frame. Live deformation contribution is separately labeled. It loads only a bounded image cache. It is not the native product runtime or an authoring editor.

## Authority and continuation

These are source-review/demo artifacts. Stage40 automatic retarget is not silently cut over. No Go Attempt, scientific qualification or ProductRevision is minted. Existing qualified artifacts and mainline authority remain unchanged. The next canonical Stage40 intervention must declare its actual correspondence, rest-frame and camera dependencies through a released-DAG Attempt. Grip orientation/clearance, terminal and temporal behavior, semantic region granularity, diagonal pixel/owner parity and linear-light composition remain qualification work. This engineering demo cannot close FIT8/unseen/generalization.
