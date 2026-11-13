> **One physical product model:** [`rss_vnext/`](rss_vnext/) contains the unified model and [`experts/`](rss_vnext/experts/) houses IRIS, TESSA, AXIS, MIRA source. The original `models.iris`, `models.tessa`, `models.axis` and `models.mira` Python module names remain stable through the declared package search path; there are not four ModelReleases. See [relocation manifest](rss_vnext/EXPERT_SOURCE_RELOCATION_V1.json).

# Models · unified RSS.vNext

The current product target is **one RSSVNEXT release, one inference**, with independently addressable specialist evidence. **Model proposals are not canonical compiler output.** The exact runtime model selection comes from `../platform/internal/capability/contracts/product_capabilities_v1.json` on the checked-out branch.

| Specialist | Current source | Role |
| --- | --- | --- |
| Shared model / fanout | `rss_vnext/` | Unified backbone and typed evidence |
| IRIS | `rss_vnext/experts/iris/vnext/` plus supporting modules | Geometry and shared 3D scene evidence |
| TESSA | `rss_vnext/experts/tessa/` | Surface/topology proposal |
| AXIS | `rss_vnext/experts/axis/` | Learned control and articulation structure |
| MIRA | `rss_vnext/experts/mira/` | Influence/skin evidence tied to the same mesh and rig |

This integrated source tree selects the nine-role RSS.vNext graph through the Go capability catalog. Checkpoint training and unseen qualification are **not** established by source availability.

## Historical code and compatibility

`../historical/model_sources/geppetto/`, `../historical/model_sources/arachne/`, `../historical/model_sources/skin_field_codec/` and earlier numbered IRIS/TESSA branches preserve research/checkpoint behavior. Their presence does **not** imply a second active product model. **AXIS and MIRA retain compatibility imports of archived old model modules**; before relocating, audit direct + transitive Python imports, checkpoint keys, serialized paths, workflow callers and training scripts.

Run `python tools/repository_clarity_audit_v1.py --json` to inspect direct static imports; this does not prove dynamic-load safety. Compiler validates and qualifies evidence; model files do not mint ProductRevision.
