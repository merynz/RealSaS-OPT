# RealSaS IRIS G0/G1 Starter V1

Canonical start of the single-pose geometry line, with explicit preservation of prior work.

Active path: `A×8 -> encoder -> multiview fusion -> dense decoder -> P/N/V/U -> SurfaceEvidenceSet`.

Surface correspondence/persistence is **required as a capability**. An explicit learned `Z` descriptor is optional as a representation. D1/D2 sources are retained under `legacy_reserve/` and may be reactivated for G5/local refinement if geometry evidence requires them.

`migrate_n1d_state_dict()` reports every tensor retained from N1D and every source tensor that remains archive-only, so no lineage component disappears silently.
