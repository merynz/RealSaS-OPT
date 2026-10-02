# Appearance → Runtime Dataflow Audit — 2026-10-02

## Scope

This audit maps the corrected Knight R&D lineage together with the recovered Oct-1 engine/runtime lineage. It distinguishes the **frozen mechanical-CAA witness path** that produced the accepted smear render from the later **source-owned visual runtime path** added during recovery.

## Accepted frozen witness

- scientific lineage: `audit/knight-rest-visual-owner-20260930`
- frozen source anchor: `70cbd671a08f3d13460a04a55e7d65ad6b4640d6`
- accepted recovered witness:
  - IDLE `4c64437ba1f2ecaca8302a52c598f3ba7750a923fd50b90f7cafa3f8473386c3`
  - RUN `64ad50eeb9ea55ce202ddc46728ffea9074419dbf855938840f69d06661806e3`
  - SLASH `adc8e8cd0e2272a836e50f7a73585208074703ea5f3c9a549583d7a659fa2b85`

The accepted frozen witness explicitly reports `source_owned_visual_mesh_arap_used=false`. It is therefore a **mechanical CAA render**, not the recovered source-owned visual runtime consumer.

## Producer / consumer map

| Stage | Produced authority | Producer | Consumed by | What is carried |
|---|---|---|---|---|
| 07 | `QualifiedObservationSetIR.v1` | observation qualification | 18, 23, recovered 37 | exact source raster hashes, foreground masks, dimensions |
| 18 | `VisualMeshSetIR.v1` | `v2_architecture.py` | 23, recovered 37 | per-view source-silhouette visual mesh: 2D positions, faces, fixed source-raster UV; metadata explicitly says mechanical candidate is not render authority |
| 18 | mechanical candidate / surface addressing | canonical mesh build | 19→35, mechanical CAA | canonical 3D carrier used for mechanics |
| 23 source-owned mode | `CompleteAppearanceAssetIR.v2` | `appearance_v2.bake_complete_appearance_stage` | 24, 37, 38, 42 | direct source RGBA textures; visual-set binding hash; `mechanical_mesh_render_authority=false`; `visual_uv_binding.npz` contains visual mesh hashes/cardinality, not mechanical face UV |
| 23 mechanical-CAA mode | `CompleteAppearanceAssetIR.v2` | CAA bake | frozen renderer / historical 42 | mechanical-face UV atlas + CAA provenance; this is the lane that produced the accepted smear witness |
| 35 | `QualifiedMeshIR.v1` + `SkinTopologyCompatibilityReport.v1` | dynamic mechanical qualification | recovered 37, 41, 42 | corrected mechanical carrier, corrected skin/topology safety evidence |
| 37 historical | `PresentationPartitionEvidenceIR.v2` + `QualifiedPresentationStructureIR.v2` | old `product_state_v2.py` | 38 | metadata says source-owned visual mode and `mechanical_mesh_render_authority=false`, but no typed qualified visual-presentation payload is emitted |
| 37 recovered | `QualifiedVisualPresentationSetIR.v1` | recovered `_qualified_visual_presentation_stage37` | 38, recovered 42 | source visual topology rebuilt/partitioned using source mask + Stage35 unsafe faces + mechanical z-buffer; region labels, seed regions, fixed source UV, Stage35 binding |
| 38 | `CompletePuppetStateIR.v2` | puppet seal | 40–42 | seals mechanical state + presentation graph + visual-presentation binding hash |
| 41 | `QualifiedDynamicMotionIR.v2` | dynamic motion proof | 42 | exact posed mechanical XYZ frames |
| 42 historical | `RuntimeProjectionIR.v2` | old `runtime_v2.build_runtime_projection_stage` | 43–45 | reopens Stage35 mechanical mesh, reads CAA face UV, packs posed 3D mechanical mesh; renderer = mechanical canonical-depth CAA |
| 42 recovered | `SourceOwnedVisualRuntimeProjectionIR.v1` | recovered `_build_source_owned_visual_runtime_projection` | 43–45 | qualified visual faces/UV + per-frame 2D visual positions; visual vertices bound to mechanical faces with region-local affine/barycentric binding |
| 43 | source-owned RSS | `build_source_owned_visual_rss_v2_entries` | 44 native player | visual mesh, fixed UV, source RGBA, per-frame 2D positions |
| 44 source-owned | native pixels | `render_source_owned_visual` | 45 | rasterizes visual triangles in face-index iteration order and alpha-composites samples |
| 45 source-owned | `SourceOwnedVisualDynamicIntegrityIR.v1` | `_prove_source_owned_visual_dynamic_integrity` | closure | checks native/reference parity, direct-source provenance, non-empty frames, flipped triangles, edge stretch |

## Proven discontinuities / unproven edges

### F1 — Historical Stage37 → Stage42 carrier discontinuity: PROVEN

The historical Knight tree can state at Stage37 that source-owned visual presentation is authoritative and that the mechanical mesh is not render authority, while the historical Stage42 implementation unconditionally reopens `35_DYNAMIC_MECHANICAL_MESH_QUALIFIED`, loads mechanical faces and mechanical CAA `face_uv`, and builds `RuntimeProjectionIR.v2`.

This is a real authority/consumer discontinuity. The Oct-1 recovery code adds a typed source-owned consumer and closes this particular gap.

### F2 — Accepted smear witness does not exercise the recovered source-owned consumer: PROVEN

The frozen accepted witness is mechanical CAA and explicitly reports `source_owned_visual_mesh_arap_used=false`. Therefore the smear render cannot be used as evidence that the recovered Stage37→42 source-owned transport itself is bad; it is evidence about the corrected mechanical CAA lane.

### F3 — Recovered source-owned runtime has no explicit dynamic occlusion/draw-order authority: PROVEN

`SourceOwnedVisualRuntimeProjectionIR.v1` carries visual faces, fixed UV and per-frame 2D positions. The RSS manifest explicitly forbids runtime tint/order/visibility authority. The native `render_source_owned_visual` loop iterates faces by index and alpha-composites each sample over the accumulated pixel. It does not consume:
- per-region dynamic depth,
- presentation draw order,
- occlusion-owner priority,
- clipping relations,
- a per-pixel arbitration authority.

The final owner buffer becomes the last nonzero-alpha triangle written at the pixel.

### F4 — Stage45 does not prove presentation arbitration correctness: PROVEN

Source-owned Stage45 gates:
- native/reference byte parity,
- direct-source provenance,
- non-empty frame views,
- zero flipped triangles,
- zero edges above catastrophic ratio 4.

It does **not** gate semantic/presentation occlusion correctness, region draw-order correctness, overlap ownership, or perceptual correctness. Native/reference parity only proves the C++ renderer matches the Python reference implementing the same policy.

## Current working diagnosis

The old `Stage37 → Stage42` carrier cut was real and is already implemented in the recovered engine.

The recovered path still has a narrower unresolved presentation problem: **dynamic 2D overlap arbitration is not represented as an authority in the Stage42 package and is not proven by Stage45.** Whether this is the dominant remaining visual defect on the corrected Knight lineage must be measured by replaying the corrected topology/weights state through the recovered source-owned path.

## Next executable audit

1. Rebind the corrected Knight Stage35 mechanical state to recovered Stage37.
2. Build `QualifiedVisualPresentationSetIR.v1`.
3. Run recovered Stage42→45 without falling back to mechanical CAA.
4. Capture per-frame:
   - region overlap pixels,
   - number of contributing visual faces per pixel,
   - cross-region overlap count,
   - owner changes under face-order permutation,
   - occlusion disagreement against a mechanical-depth-derived region ordering oracle.
5. Render idle/run/slash and compare to frozen mechanical-CAA witness.

No RGB head, generated texture, or new appearance architecture should be selected before this test distinguishes missing arbitration from representation failure.


## Promoted behavioral evidence — 2026-10-02

### F5 — Triangle array order is an implicit overlap authority: PROVEN

Hosted audit run `36940982276` held geometry, posed positions, texture bytes and per-vertex UVs constant and reversed only two fully overlapping visual triangle rows. **45 visible pixels changed**. The representative pixel changed from blue `[0,0,255,255]` to red `[255,0,0,255]`; framebuffer hashes changed from `41a177be...` to `a4c87ff6...`.

Canonical evidence: `canonical/SOURCE_OWNED_VISUAL_FACE_ORDER_SENSITIVITY_AUDIT_V1_20261002.json`.

Therefore Stage44 face iteration order is behaviorally acting as an undeclared overlap/presentation authority.

### F6 — Visual triangle single-affine-domain coherence is not guaranteed: PROVEN CONTRACT GAP

`bind_region_visual_vertices_to_mechanical_affine_v1` selects a mechanical face independently per visual vertex. Stage37 guarantees visual-region coherence, but it does not guarantee that all three corners of one visual triangle bind to one qualified affine deformation domain or to a continuous transfer field. The recovered Stage42 source-owned path still consumes this per-vertex binding.

### Corrected solved-lineage source-owned replay: FAIL-CLOSED

The attempted replay on `SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929` did **not** reach a successful source-owned runtime render. Workflow run `36943075707` failed during visual affine binding with:

`VISUAL_AFFINE_BIND_SEED_DISTANCE_EXCEEDS_BUDGET:109:10.124228365658293`

This failed attempt is preserved as evidence for A (deformation-domain/admissibility ownership); its workflow change is not promoted to canonical main.

## Current closure order

A. Visual triangle deformation-domain coherence.  
B. Canonical depth → visual overlap ownership.  
C. Stage45 fail-closed proof of A + B.
