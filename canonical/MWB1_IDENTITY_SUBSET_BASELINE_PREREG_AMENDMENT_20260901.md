# RealSaS — MWB-1 Identity Baseline Prereg Amendment — Camera-Independent Binding — 2026-09-01

**Status:** `FROZEN_BEFORE_MWB1_OUTPUTS`

The consumer-interlock transport fixture is an already reconstructed qualified D2 surface witness and does not expose a byte-level `camera.json` authority to MWB-1.

MWB-1 does not perform projection, backprojection, vertex insertion, triangulation from raster coordinates, or any camera-dependent geometry operation. Every mesh rest position is an exact existing admitted `SurfaceNode.P`.

Therefore MWB-1 freezes the following explicit null-camera-dependency policy:

- `view_index` is the smallest support view common to the three selected identity-bound surface nodes;
- `camera_binding_hash` is a deterministic lineage sentinel `SHA256({fixture_transport_sha256, shared_support_view, authority_class="CAMERA_NOT_CONSUMED_BY_MWB1_IDENTITY_BASELINE"})`;
- this sentinel proves which already-admitted witness/view context was used but makes **no claim** that camera bytes were restored in this fixture;
- `qualification_report.camera_geometry_consumed = false`;
- this exception is limited to MWB-1 because no mesh position or support coefficient is derived from camera geometry.

Any later MWB-2 view-local interpolation/CDT candidate that uses projection/raster geometry must bind the exact authoritative camera bytes/hash and may not inherit this sentinel policy.
