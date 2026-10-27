# Graphics Consumer Local Order V2

Status: ORACLE_CONSUMER_REPAIR__NO_PRODUCT_ACTIVATION__NO_TRAINING_AUTHORITY

## Why this exists

`CompiledRayGraphicsV2` correctly permits ray-local interleaving. Two different
rays may support opposite precedence for the same pair of surfaces without any
contradiction. `RuntimeGraphicsV1`, however, stores one view-global fragment DAG.
Unioning ray-local constraints into that DAG can create a false cycle and loses
the spatial support on which the evidence was measured.

A second V1 problem is equal-depth fallback. Native V1 eventually uses face index
to break an otherwise unqualified tie. Face index is canonical topology identity,
not presentation truth, and must not silently become draw-order authority.

## Additive native primitive

`graphics_local_order_v2.hpp` adds a court-level native consumer primitive with
these rules:

- precedence edges are supplied for one pixel/ray only;
- opposite precedence on another ray is legal;
- local contradictory precedence fails closed;
- unrelated layers fall back to measured physical depth;
- an unresolved depth tie fails closed instead of using face id;
- no edge is unioned into a view-global DAG.

This does not mutate the existing `RSGFX001` transport or activate a new product
runtime contract.

## Remaining blocker: executable support

Current `RayStackEvidenceV2` carries `scope_hash` and `ray_key`, but the native
runtime cannot evaluate an opaque hash as a spatial predicate. A production
Graphics V2 carrier therefore still requires an explicit, content-addressed,
consumer-evaluable support domain, for example a qualified raster/ray address or
another declared predicate that can answer:

> does this local precedence rule apply to this runtime overlap sample?

The exact support representation is not minted by this court. It must preserve
camera/view binding, scope provenance, coverage/unknown status, ties and motion
conditioning where applicable. Sparse oracle samples must not be silently
extrapolated to unobserved pixels.

Until that support carrier exists and is exercised by the native renderer:

- Graphics V2 product activation is forbidden;
- source oracle PASS does not imply runtime coverage PASS;
- IRIS/TESSA training must not treat opaque `scope_hash` as a fully executable
  consumer target;
- V1 remains a historical/global-DAG consumer only.

## Court consequence

This repair is still useful before training: it proves the native consumer can
represent the correct local ordering semantics once an executable support binding
is supplied. The next Graphics court is therefore not another ordering heuristic;
it is the support-binding + native raster consequence court.
