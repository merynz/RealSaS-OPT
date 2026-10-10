"""Research-only producer. Product 46-stage DAG is unchanged."""
from pathlib import Path
from compiler.realsas_compiler_core.visual_oracle_reference_v1 import produce
from compiler.realsas_compiler_core.visual_oracle_io_v1 import write_json, file_ref


def produce_reference(ctx):
    root = ctx["run_root"] / "artifacts" / "47_AUTHORED_VISUAL_ORACLE"
    evidence = produce(ctx["run_manifest"]["oracle_source"], root)
    outputs = [write_json(root / "evidence.json", evidence)]
    outputs.extend(file_ref(Path(a["ref"]["path"]), "image/png") for a in evidence["assets"].values())
    return {"status": "PASS", "outputs": outputs, "diagnostics": {
        "scope": evidence["scope"], "frame_count": len(evidence["frames"]),
        "owner_count": len(evidence["owners"]), "asset_count": len(evidence["assets"]),
        "product_qualification": False}}
