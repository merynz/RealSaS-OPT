"""AST-visible contact/composition presentation closure; no product authority."""
from __future__ import annotations

import json
from pathlib import Path

from compiler.realsas_compiler_core.semantic_runtime_package_v1 import (
    SEMANTIC_EFFECTIVE_DEPTH_CONTRACT,
    seal_semantic_runtime_entries,
)
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v5_impl as impl


def compile_source_domains_stage(ctx):
    return impl.compile_source_domains_stage(ctx)


def compile_projection_stage(ctx):
    return impl.compile_projection_stage(ctx)


def package_stage(ctx):
    """Seal V7's effective-depth bytes under their semantic, not physical, contract."""
    result = impl.package_stage(ctx)
    archive_output = next(
        row for row in result["outputs"]
        if row.get("schema") == "application/x-realsas-rss-v2"
    )
    package_output_index = next(
        i for i, row in enumerate(result["outputs"])
        if row.get("schema") == impl.PACKAGE_SCHEMA
    )
    archive = Path(archive_output["path"])
    entries = impl.base.read_rss_v2(archive)
    rewritten = seal_semantic_runtime_entries(entries)
    receipt = impl.base.write_rss_v2(archive, rewritten)
    archive_output["sha256"] = receipt["archive_sha256"]

    package_path = Path(result["outputs"][package_output_index]["path"])
    package = json.loads(package_path.read_text())
    package["archive"]["sha256"] = receipt["archive_sha256"]
    package["semantic_depth_sort_key_contract"] = SEMANTIC_EFFECTIVE_DEPTH_CONTRACT
    result["outputs"][package_output_index] = impl.base.write_json(
        package_path,
        package,
        authority_class="SCOPED_RESEARCH_PRESENTATION_PACKAGE",
        schema=impl.PACKAGE_SCHEMA,
    )
    result["diagnostics"] = {
        **dict(result.get("diagnostics") or {}),
        **receipt,
        "semantic_depth_sort_key_contract": SEMANTIC_EFFECTIVE_DEPTH_CONTRACT,
        "physical_depth_final_authority": False,
    }
    return result


def playback_stage(ctx):
    return impl.playback_stage(ctx)


def prove_presentation_stage(ctx):
    return impl.prove_presentation_stage(ctx)
