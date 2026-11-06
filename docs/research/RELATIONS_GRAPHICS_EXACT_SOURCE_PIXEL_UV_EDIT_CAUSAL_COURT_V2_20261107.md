# Original-source Relations × Graphics exact local face–pixel–UV intervention truth V2

2026-11-07 · draft PR #367 on source-courted V1 PR #366 · **source-only**. No artist drawable identity, no IRIS/TESSA materialization or training authority.

## Scientific problem closed by this method (in a bounded source subset)

We already independently proved 608 original Cat source face→pixel→UV-edit witnesses with a very conservative whole-other-triangle minimum-depth bound. This excluded otherwise valid surface pixels when a distant part of an occluding triangle happened to have a small depth, even though that near region never projected into the relevant pixel.

V2 solves this specific mathematical over-rejection **without weakening the first-hit criterion**:

Let source original indexed face \(f\) cover a guarded entire camera pixel square \(Q\), with all four pixel corners strictly inside the original projected source triangle and original near/far clipping. For any OTHER indexed source triangle \(g\), let

\[
P_{gQ} = Q \cap \Pi(g).
\]

If \(P_{gQ}\) is empty then \(g\) cannot contribute to ANY source ray within \(Q\). If nonempty, we ask whether at every point of \(P_{gQ}\), original camera depth of \(g\) is strictly farther than \(f\).

Both depths are affine on original projected triangles under our **admitted orthographic camera policy**. Thus \(z_g(q)-z_f(q)\) is affine on the convex intersection polygon \(P_{gQ}\), and its minimum occurs at one of that polygon's vertices. Therefore

\[
\boxed{
Q\subseteq\Pi(f),\quad
\forall g\ne f\ \forall v\in\operatorname{vertices}(P_{gQ}):
 z_g(v)-z_f(v) > 10^{-5}
\quad\Longrightarrow\quad
\forall q\in Q:\operatorname{firstHit}(q)=f
}
\]

under the originally qualified opaque-source, fixed camera/rest geometry regime.

**Proof:** Let \(q\in Q\). Coverage guarantees a source intersection on face \(f\). Any other source triangle \(g\) that intersects the ray through \(q\) must project over \(q\), hence \(q\in P_{gQ}\). \(P_{gQ}\) is convex (intersection of two convex polygons). A linear function on a bounded convex polygon attains its minimum at a vertex. The strict vertex inequalities imply \(z_g(q)>z_f(q)\); thus \(g\) cannot be a nearer first hit. Repeated for all original source triangles, \(f\) wins every ray in \(Q\). QED.

Degenerate original source triangle projections are handled **conservatively**. When a degenerate projected triangle's AABB intersects \(Q\) and it could be near, the candidate is excluded; no zero-area triangle is silently ignored. This can cause false negatives, never new false original face-positive proofs. The pixel is enlarged by 0.05 pixels in each direction, and source depth/UV barycentric validity carry fixed numerical margins.

The **original glTF UV-address** condition is unchanged from V1: the ORIGINAL source face-corner top-left UV map, barycentrically interpolated to the FOUR guarded pixel corners, must lie strictly inside the same precommitted 64×64 atlas RGB-edit tile with a whole 1-original-texel margin. The UV mapping is affine in orthographic screen coordinates, so convexity proves interior tile address for the entire pixel.

The theorem is geometric **and conditional on the physical Cycles camera sample rays lying inside the guarded pixel footprint**; individual stochastic Cycles offsets are not fabricated. Physical observability is tested separately against the ORIGINAL 256 Cycles sample render frames.

## A fully independent implementation/court pair

- Producer: \`measure_original_source_face_pixel_uv_exact_depth_v2.py\`. Uses Shapely 2.1 STRtree and projected polygon∩pixel geometry, then original SHA-locked source samples with the existing V1 physically courted PM-RGBA source effect checker.
- Independent court: \`court_original_source_face_pixel_uv_exact_depth_v2.py\`. **Does not import the V2 producer or Shapely.** It independently projects original indexed faces, clips every possible pixel-intersecting source triangle with a hand-written Sutherland–Hodgman half-plane polygon algorithm, recomputes original face/competitor affine depths directly from determinants, validates UV-address and first-hit proof for *every* published source witness, and rechecks all eight physical Cycles original edit effects and other-edit negative constraints.
- Adversarial tests: \`test_original_source_face_pixel_uv_exact_depth_v2.py\`, including a synthetic triangle whose global minimum depth is closer than the candidate but whose exact pixel-intersection region is safely behind. This tests why the V2 method is less restrictive than V1 while remaining geometrically valid.
- Source V1 receipt pins: producer ad26d6f23baa25cb8ebcf38b9d8256344b83c9bed7c342f7a1688cc659208a65, independent physical court 2b77b43ec09afe37fb1f277887c9e946133c034d3d736f4b8ed0f5d40f2f5bc0. All **608** previously physically court-accepted source face/pixel/edit addresses must be a literal subset of V2's new addresses. If one disappears, V2 fails.
- Source original Cat GLB, 684 native face indices, source rest XYZ, raw glTF UV, camera, source unlit material policy, UV operation plan, original source Cycles 256 frames and independent source edit court are exactly the same SHA-locked original artifact bytes as V1. No new source render; no threshold, pixel mask or source-camera fitting.

## Current measured local pilot — independently heldout until GitHub court PASS

The new original-source pixel/mask logic was first evaluated locally against exact original Cat source bytes. It found **49,494** source face/pixel/UV edit candidate witnesses across four original cameras:

| Original Cat source view | Exact local-depth potential edited face pixels |
|---|---:|
| V1 | 11,929 |
| V3 | 12,813 |
| V5 | 12,781 |
| V7 | 11,971 |
| **Total** | **49,494** |

Edit counts 24,851 / 24,400 / **243** by original atlas edit indices 0/1/2 respectively. All eight exact original Cycles sample offset frames for all candidate addresses had physically observed target edit effects above inherited original 0.025 PM-RGBA tolerance, with no observed other-edit effect above that tolerance and original BASE alpha opaque.

These numbers are PRELIMINARY at the time of writing. Only an independently successful GitHub real source court with immutable Drive SHA readback can promote them to an admitted physical source truth subset. In particular the edit2 result is deliberately bounded (243 source pixel addresses), not 6,048 entire source sample-incidences or source drawable ownership.

Expected original source physical *sample obligations* if independent court qualifies every candidate: 395,952 intended edited sample effects and 791,904 other-edit absence checks.

## What this does NOT establish

- No original author layer/part ownership or 42 drawable count.
- No **complete** source screen ray→face atlas map; near coverage boundaries, physically invisible surfaces and UV tile boundaries are excluded.
- No exact original Cycles stochastic per-sample ray origin reconstruction or camera pattern estimation.
- No new original pose14 experiment. The source here is original bind/rest pose, cameras 1,3,5,7, original 8 source sample indices.
- No nonbinary source-alpha compositing contract and no cross-morphology genericity demonstration. Triceratops source shader/shape is a DIFFERENT original experiment.
- No source geometry from student IRIS prediction; all exact original source indices are private teacher-only evidence.
- No inference/training-authority, unseen validity or product compiler decomposition policy. All flags remain false.

**Research consequence if PASS:** we can now identify far more **typed source causal edit-consequence observations at original indexed face/UV address**, while the original artist's drawable partition remains mathematically underdetermined. That is the necessary bridge from V18s Relations/Graphics geometry to independently measured original physical UV interventions in the declared same-source domain, not a post-hoc segmentation workaround.

Scientific next gate: validate generality with another source morphology, with its own independent source camera/material/shader/edit/alpha policy; compare pose-transported source event loci under controlled articulation rather than assume Cat rest equivalence everywhere.
