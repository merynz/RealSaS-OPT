# Production evidence and runtime ownership

This is an implementation rule for the existing Go / Compiler / Runtime product
line, not a new research backend.

- Input owns immutable observed front/back RGBA and cameras.
- One RSS checkpoint owns the shared H(x), semantic, graphics, TESSA, AXIS/MIRA
  candidates; all seven children remain independently addressable in CAS.
- Compiler may create new, versioned decisions but must never destructively
  replace neural evidence with a surface or infer source facts from predictions.
- Graphics preserves independent same, cut, directed-attachment and UNKNOWN.
- Mechanics uses CanonicalMesh, CanonicalRig, CanonicalSkin and solver policy;
  articulated appearance consumes those exact posed vertices, not a visual mesh.
- Runtime pose-depth inspection queries the same deformed canonical triangles,
  using camera-forward Z; it exposes ties and never claims source-observed
  visibility, RGBA alpha, material, order, or drawable certainty.
- Go Platform keeps EngineRelease / ModelRelease / ProductRevision identity;
  UI compilation stays unavailable until an exact-parent complete product
  chain is admitted.

Engineering adoption from references:
- AniGen S3 Fields: shared spatial representation must preserve shape,
  skeleton and skin cross-consistency; no topology-only joint promotion.
- SkinTokens: FSQ IDs only have meaning with exactly bound trained MIRA
  decoder, joint order, hard-code parity; soft training weights are not proof.
- ArtMesh: topology and motion consistency are different supervision axes;
  dynamic mesh cannot be assumed motion-ready from static triangles.
- StdGEN: semantic source-role fields and first-hit source witnesses require
  distinct admission; predicted color is not source pixel ownership.
- NVIDIA FlexiCubes/nvdiffrast: differentiated extraction and pose-local depth
  raster do not by themselves establish source-visibility authority.

Next production gates: active Go RSS five-role catalog, source-qualified
canonical mesh, embedded MIRA hard-code parity, pose/alpha compositing,
runtime ProductRevision binding and native publish E2E.
