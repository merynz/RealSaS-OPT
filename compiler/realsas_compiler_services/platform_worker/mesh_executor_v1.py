from __future__ import annotations

from typing import Any, Callable

MESH_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductMeshCompile.v1"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) != 1:
        raise RuntimeError(f"MESH_INPUT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}")
    return rows[0]


def execute_mesh_capability_v1(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != MESH_EXECUTOR_CONTRACT_V1:
        raise RuntimeError("MESH_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("MESH_CAPABILITY_KIND_DRIFT")
    if not str(request.get("execution_id") or ""):
        raise RuntimeError("MESH_EXECUTION_ID_REQUIRED")

    source_ref = _one(request, "RealSaS.SourceObservation", "v1")
    shape_ref = _one(request, "RealSaS.ShapeProposal", "v2")

    from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import decode_source_observation_v1
    from compiler.realsas_compiler_services.platform_worker.mesh_capability_v1 import compile_mesh_v1
    from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import decode_shape_proposal_v2

    source = decode_source_observation_v1(read_object(source_ref))
    shape = decode_shape_proposal_v2(read_object(shape_ref))
    result = compile_mesh_v1(
        shape,
        source,
        source_shape_content_sha256=str(shape_ref.get("content_sha256") or ""),
        source_observation_content_sha256=str(source_ref.get("content_sha256") or ""),
    )
    sealed = put_bytes(result.payload)
    diagnostics = dict(result.diagnostics)
    diagnostics["source_observation_content_sha256"] = str(source_ref.get("content_sha256") or "")
    diagnostics["shape_proposal_content_sha256"] = str(shape_ref.get("content_sha256") or "")
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [
            {
                "role": "canonical_mesh",
                "artifact_type": "RealSaS.CanonicalMesh",
                "schema_version": "v1",
                "storage_key": sealed["storage_key"],
                "content_sha256": sealed["content_sha256"],
                "size_bytes": sealed["size_bytes"],
                "authority_class": "COMPILER_STATIC_CANONICAL_MESH",
            }
        ],
        "diagnostics": diagnostics,
        "failure": None,
    }
