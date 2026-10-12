# Knight SLASH source-frame correction

The sealed `demo_slash_v1.motion.json` is the distinct artist take
`HumanArmature|HumanArmature|Run_swordAttack`, SHA256
`6422209f1291dbb3dbfdffb6b238e63214f8aeddfc31e12f1b105ecc1da85298`.
The previous witness was genuinely this clip, rather than a mislabeled RUN.
However, its per-joint local quaternions were copied into different target rest
frames. Source ancestors omitted by the correspondence, including Shoulder.R,
also lost their animation.

`motion_retarget_frames_v3.compile_retargeted_keys` samples artist source FK,
folds skipped source ancestors, applies an explicit proper object-basis rotation,
and rebases rotations/translations into each exact target rest frame. Target M,
hierarchy, control positions and W stay byte-identical. Unknown correspondence,
wrong ancestry, animated scale and reflected object bases fail closed.

The review renderer first identifies the original correspondence by equality of
the entire copied quaternion sequence (or animated body translation). Every
nonidentity match must be exact and unique. The previous witness supplies M/G/W
and this correspondence certificate; its posed vertices are **never** the new
animation input. Identity/unmapped controls retain inherited parent motion.
This is the bounded existing correspondence, not a new universal retarget solve.

The two sealed apparatus root frames establish the explicit proper 180-degree
object-basis alignment. A separate source FK comparison checks the resulting
world-space skin rotation at every mapped control and every authored time.
The largest matrix residual is 2.03e-15 over 21 frames. The old right forearm
orientation differs from the corrected source orientation by up to 160.64 degrees.

Run from current source:

```bash
python tools/demo/render_knight_source_motion_v3.py \
  --source /absolute/path/demo_slash_v1.motion.json \
  --witness /absolute/path/sealed_witness_replay.npz \
  --output /absolute/path/slash-output --workers 4
python -m unittest tests.compiler.test_motion_retarget_frames_v3 -v
```

The witness must match the approved baseline's exact parent SHA256
`bb8cfb075b5ec3df93cc1fac306d65c88c27f5a6a7fc1bbf74c05ec79bcdba35`.
The renderer checks baseline bytes, reuses the existing source-wins + compiled
CAA material on M, and generates 168 fresh frames, an eight-view GIF/MP4 and
individual V0-V7 GIFs. Authored timing is retained; GIF timing uses cumulative
centisecond rounding. Its repeat flag is for visual inspection and does not
impose loop closure on the nonlooping source clip.

This is source-review/demo evidence. Stage40 automatic retarget and existing
qualified artifacts are not replaced by this review consumer. No Go Attempt,
scientific qualification, ProductRevision, attachment PASS or depth/order PASS
is created. The old approved RUN regression is unchanged. Camera-twist attachment
correction is not applied to these frames; they show actual source motion on M.

Three analytic unit cases verify an omitted animated shoulder's rotation and
orbital translation, a different target rest axis, object-basis rotation and
reflection rejection. The next canonical Stage40 intervention must carry its
actual camera/rest-frame and correspondence dependencies rather than silently
substituting this bounded witness map.
