# Connected presentation pose intervention — 2026-10-09

Scope: PR #66, downstream research only. PR #65 remains merged. Frozen M/G/W,
rig, skin, IRIS, AXIS and MIRA are unchanged. Visual closure is **pending**.

## Implemented intervention

`visual_presentation_pose_v1.py` derives one connected 2D palette from exact
frozen G parents and the existing sealed motion matrices. It retains proper
camera-twist rotations, transports projected setup offsets through the parent
palette, preserves projected root transport and explicit local translations,
and fails on unobservable rotations. It does not infer another rig or fit W.
Body coefficients are unchanged. Equipped art consumes the same palette's
qualified target slot; body-only source motion does not require sword/shield.

The scoped V4 adapter compiles this field under a new V6 operator identity,
retains the independently verified legacy field as a control, and freezes the
witness-wide body safety selection. RSS/native and Python reference readers
consume the new identity and compiled XY/depth. Runtime performs no pose solve.
The previous V5 provisional surface/slot depth remains unchanged in this
intervention; it is **not** claimed as qualified semantic drawing ownership.

## Independent checks and limits

The Stage45 V4 gate independently measures actual palette parent/child origins,
root transport, rotation style, palette consumption, setup identity, final
triangle health, rigid equipped-art consumption and source-cut candidate
separation. It replays safety selection and final fields from sealed inputs.
Frame0 and loop endpoints are reported separately; setup is not frame0.

Coincident source coordinates only generate diagnostic cut candidates, scoped
to the same body/prop owner. They do not authorize welding unrelated surfaces.
The existing reduced graph has no qualified cross-chart contact/support,
dynamic material exposure or semantic overlap interval/transition inputs.
These predicates are explicitly **unavailable (`null`)**, with named blockers;
the gate cannot turn healthy triangles and byte parity into visual PASS.
Final contact constraints, contact-preserving repair, qualified dynamic coverage,
and semantic body/prop drawing order still require implementation/qualification.

The receipt [connected palette Knight diagnostic](evidence/connected_palette_knight_local_v1.json)
pins the exact exported witness/projection and implementation hashes. Across
984 frame-views, legacy joint-origin residual reaches 108.698119 source px;
the connected palette reaches 3.46e-13 source px, with no failed palette relation.
Source images are 1024 px; these are not final raster gap sizes. This local
diagnostic is not a scientific Attempt, Stage45 PASS or causal attribution of
all visible gaps.

An additional [support/alpha diagnostic](evidence/visual_support_alpha_knight_local_v1.json)
compares opaque-texture triangle support with original-source artwork support
at RUN frame10 in V0/V4/V6, using the engineering candidate fields. The candidate
has zero covered-but-art-alpha-zero pixels in those three frames. This rules
out source texture alpha alone as the explanation for their remaining empty
support; it does not qualify an intended silhouette or establish that every
enclosed empty component is erroneous. Reproduce the numerical ablation with
`tools/probe_visual_coverage_vs_alpha_v1.py`; the receipt binds its source and
all supplied arrays/textures. It remains an unqualified local probe.

Tests exercise a broken relation with proper rigid triangles and real C++/Python
byte-identical nonempty renders; the shared Stage45 relational gate rejects it.
An adapter-level fixture also rejects the old palette while its old replay proof
is assumed healthy. Missing material/order qualification is tested independently.

## Matched Go execution

`run_matched_presentation_attempts.py` executes V5 baseline and V6 child under
the same Stage45 V4 measurement gate, exact SubjectInput and native player.
Release snapshots differ directly only at Stage42. Before running the child,
the driver inspects Go's impact. Afterward it requires Stage37 Registry reuse
and identical output byte hashes; a failed baseline Stage45 cannot suppress
retention of successful upstream artifacts.

Measured joint-origin signatures feed the existing controlled owner-attribution
service, guarded against protected metric regression. The broad visual-gap
signature receives no invented intervention evidence and must abstain.
Research projection state hashes do not confer ProductRevision authority.

Both child outputs export fresh native-derived IDLE/RUN/SLASH across V0–V7
as 768 px previews of 256 px native frames. A FAIL retains diagnostic visuals
and Attempt receipts. Accepted visual quality, product promotion and optimization
remain later boundaries. No new hosted verdict is claimed by this source commit.
