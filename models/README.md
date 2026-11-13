# Models · unified RSS.vNext

The current product target is **one RSSVNEXT release, one inference**, with independently addressable specialist evidence. **Model proposals are not canonical compiler output.** The exact runtime model selection comes from `../platform/internal/capability/contracts/product_capabilities_v1.json` on the checked-out branch.

| Specialist | Current source | Role |
| --- | --- | --- |
| Shared model / fanout | `rss_vnext/` | Unified backbone and typed evidence |
| IRIS | `iris/vnext/` plus supporting modules | Geometry and shared 3D scene evidence |
| TESSA | `tessa/` | Surface/topology proposal |
| AXIS | `axis/` | Learned control and articulation structure |
| MIRA | `mira/` | Influence/skin evidence tied to the same mesh and rig |

The nine-role RSS.vNext candidate is in [PR #485](https://github.com/merynz/RealSaS-OPT/pull/485), not yet the `main` execution graph. Checkpoint training and unseen qualification are **not** established by source availability.

## Historical code and compatibility

`geppetto/`, `arachne/`, `skin_field_codec/` and earlier numbered IRIS/TESSA branches preserve research/checkpoint behavior. Their presence does **not** imply a second active product model. **AXIS and MIRA still import some older model modules**; before relocating, audit direct + transitive Python imports, checkpoint keys, serialized paths, workflow callers and training scripts.

Run `python tools/repository_clarity_audit_v1.py --json` to inspect direct static imports; this does not prove dynamic-load safety. Compiler validates and qualifies evidence; model files do not mint ProductRevision.
