# Presentation attachment motion ownership

Stage37 qualifies target-art membership, canonical slot identity and the explicit
presentation motion model. Texture charts and safe surface regions are addressing
boundaries; they must not silently become independent motion owners.
Stage42 owns compilation of the resulting source-owned visual motion. Stage45
owns independent replay, ownership, slot fit, deformation quality and depth proof.
M/G/W and motion retargeting remain the sealed mechanical authorities.

`CANONICAL_SOURCE_CHART_FIELD` retains the continuous source-chart deformation
field for deformable artwork. `TARGET_SLOT_LOCAL_RIGID_2D_CAMERA_TWIST_V1` is an
explicit opt-in for rigid 2D target attachments. All its visual vertices, including
disconnected texture charts, use one proper 2D rotation and translation around
the projected canonical target slot. Source art stays in the target rest frame;
the animated first frame is not substituted for that rest frame. Source motion
equipment is optional.

For a view screen basis P and the sealed slot rotation R, the proper view-normal
twist has cosine/sine proportional to `(PRPᵀ)00 + (PRPᵀ)11` and
`(PRPᵀ)10 - (PRPᵀ)01`. An unobservable twist fails closed. This is a 2D art motion
contract: out-of-plane rotation does not flatten, reflect or shred a rigid 2D
image. It is not a claim of exact 3D plate projection. Slot translation follows
the exact posed canonical slot; every source vertex retains its rest offset.
The independently bound canonical surface-depth field is unchanged, including
its tie/overflow rejection. There is no painter-order or depth bias override.

Stage45 rederives ownership and all XY/Z samples from the sealed witness. It also
checks that selected target carrier vertices follow the declared slot within the
versioned relative fit tolerance. A wrong slot cannot pass merely because the
rendered image is rigid. Attachment-specific shape evidence is reported separately
from the complete presentation verdict; body failures keep the overall verdict
FAIL. The area/condition/edge thresholds are unchanged.

The design was checked against Esoteric Software's official `RegionAttachment`
and unweighted `VertexAttachment` implementations: all local attachment vertices
share the owning slot bone's transform. RealSaS implements its own camera-twist
mapping and canonical proof; no Spine runtime implementation is incorporated.

Reference: https://github.com/EsotericSoftware/spine-runtimes/blob/4.3/spine-libgdx/spine-libgdx/src/com/esotericsoftware/spine/attachments/VertexAttachment.java

## Canonical body motion blend

V4 also transports scalar motion coefficients from the sealed canonical skin
field through the same safe barycentric samples and positive harmonic chart
operator. Every visual vertex evaluates the shared 2D canonical pose palette
with those coefficients. A single-anchor chart follows rotation as well as
translation; it no longer becomes an independently translated shard. The rigid
attachment declaration still overrides body blending for its owned vertices.

The coefficient field is presentation addressing derived from exact W_M; there
is no mechanical skin inference, fitting, dominance selection or sparsification.
The canonical 3D skin matrices and W_M must reconstruct the exact sealed posed M
to 1e-8. Stage45 rederives every coefficient from the upstream W_M and verifies
every XY/Z frame. This is a versioned 2D expression of the existing 3D witness,
not exact orthographic projection of a 3D body or a new mechanical qualification.
