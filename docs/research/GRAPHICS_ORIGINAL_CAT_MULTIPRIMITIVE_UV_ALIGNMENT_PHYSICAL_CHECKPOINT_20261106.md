# Graphics original-source multi-primitive UV→teacher binding: physical-input checkpoint

Date: 2026-11-06. Stacked graphics research PR #357 and child #358. **This is a narrow, reproducible original Cat source-address binding result, not a generic Graphics decomposition, runtime parity or student inference PASS.**

## Verified original-source facts

- [Original source-alignment hosted CI 42284306947](https://github.com/merynz/RealSaS-OPT/actions/runs/42284306947): **PASS** with exact private source-byte/readback and Blender 4.5.14 imported-mesh verification.
- Original source Cat GLB SHA `1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850`, original teacher v4 SHA `c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603`, same original linked texture bytes/SHAs as previous V3 physical experiment. Original private teacher/GLB bytes remain out of GitHub.
- The source glTF has **7 source mesh primitives** in the active source scene and **684 UV triangles**. Blender imported teacher rest has **684 frozen source triangles**.
- In the native-teacher source-V4 import, frozen Blender face indices, anonymous mesh owner indices and source-world rest vertex positions were checked against the separately SHA-locked teacher arrays, with the frozen exporter tolerance.
- Independently reading raw glTF binary `TEXCOORD_0` triangles for all active scene mesh instances produced the same **684/684 top-left UV triangle multiset** as the Blender-native face-corner UV array. **0 absent, 0 excess** UV triangles. The comparison uses orientation-independent per-triangle UV keys at a declared **1e-5 normalized UV quantization grid**.
- The face-ordered native Blender triangle UV is produced only after both teacher snapshot identity and raw glTF UV multiset checks; private `PRIVATE_CAT_NATIVE_FACE_UV_V0.npz` and signed JSON report were uploaded to the dedicated private Drive run artifact location.

## Scientific scope, independence and limits

This proves a teacher-only original-import **source face/corner UV transport witness** and blocks the previous single-primitive indexing assumption. A UV-only multiset check is *not* a global invertible direct raw-glTF-primitive-to-canonical-vertex map independent of Blender. The per-face correspondence specifically relies on the immutable v4 Blender import and exact teacher face/rest replay; importer or teacher lineage drift must fail closed. Quantization is diagnostic provenance, not authorization to mint artist ownership.

It does **not** prove a BUSTED student can infer UV, topology, mechanical controls, drawable parts or source identity from a single image. It does not prove `CompiledGraphicsV1`, `graphics_v2` or product Runtime V4 uses this witness in a shipping pipeline. Nor does it prove original Cycles vs CAA reference-puppet pixel parity. All materialization/training authority remains false.

## Failures retained

- [Single-primitive research gate 42282768473](https://github.com/merynz/RealSaS-OPT/actions/runs/42282768473) legitimately returned `UNKNOWN: MULTIPRIMITIVE_VERTEX_MAP_UNQUALIFIED`; successful CI means correct abstention, not source binding.
- [First native Blender probe 42284017476](https://github.com/merynz/RealSaS-OPT/actions/runs/42284017476) failed before scientific comparison because Blender Python lacked Pillow. A dependency-free strict glTF GLB accessor plus tests replaced the transitive image import. Subsequent operational probe failure(s) were retained.
- [Current source UV inventory CI 42284381283](https://github.com/merynz/RealSaS-OPT/actions/runs/42284381283) passed **16/16** strict binary/scene/UV and fail-closed tests.
- No ad hoc subject-specific primitive offset mapping, no silently guessed merge, no retuned face tolerance.

## Next independent result

[Cat V3 compiled-puppet physical comparison workflow 42285528834](https://github.com/merynz/RealSaS-OPT/actions/runs/42285528834) was launched with predictions archived before opening private original Cycles target arrays. Its measured outcome must be read from the independent comparison JSON, not assumed from the workflow state. Original source pose/view/edit scope and .025 PM RGBA threshold are frozen in the preregistration. No new training or product authority is minted.
