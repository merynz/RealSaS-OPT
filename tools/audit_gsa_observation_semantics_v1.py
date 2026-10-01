from __future__ import annotations

import inspect, json, subprocess
from pathlib import Path

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)

ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/"compiler/realsas_compiler_core/substrate/scene_first_signed.py").read_text()
tensor=(ROOT/"models/geppetto/reference_strength_v1/rigging_surface_tensorization_v1.py").read_text()

params=tuple(inspect.signature(rigging_surface_from_scene_first_zero_mesh_v1).parameters)
mask_inputs=[p for p in params if "mask" in p.lower() or "observation" in p.lower()]
payload={
  "schema":"RealSaS.GSAObservationSemanticsAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "builder_parameters":params,
  "source_mask_or_observation_parameters":mask_inputs,
  "self_zbuffer_support_used":"_self_zbuffer_support(" in src,
  "observed_flag_from_any_self_zbuffer_support":(
      'flags = ("OBSERVED_SIGNED_ZERO_SURFACE",) if views else' in src
  ),
  "source_observation_ids_for_scene_first_empty":"source_observation_ids=()," in src,
  "geppetto_consumes_observed_completed":(
      '"OBSERVED_SIGNED_ZERO_SURFACE" in flags' in tensor
      and '"MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in flags' in tensor
  ),
  "finding":{
    "id":"GSA_OBSERVED_FLAG_MEANS_GEOMETRICALLY_EXPOSED_NOT_SOURCE_OBSERVED",
    "severity":"P1",
    "confirmed":(
      not mask_inputs
      and "_self_zbuffer_support(" in src
      and 'flags = ("OBSERVED_SIGNED_ZERO_SURFACE",) if views else' in src
      and "source_observation_ids=()," in src
    ),
    "class":"EPISTEMIC_SEMANTICS_DRIFT",
    "consequence":"A compact node can be labelled observed because the predicted surface is camera-visible even though no exact source-observation identity is attached. Geppetto treats this label as an input feature.",
    "design_before_code":"Freeze whether this bit means camera-exposed geometry or source-supported observation. If source-supported, derive it from qualified observation evidence/masks with exact provenance rather than renaming self-visibility as observation.",
    "claim_boundary":"Stage13 global silhouette qualification can limit gross false positives but does not make per-node self-visibility identical to exact source observation support."
  }
}
out=ROOT/"canonical"/"GSA_OBSERVATION_SEMANTICS_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
