from backend.realsas_platform.api import create_app
from backend.realsas_platform.workflows import CompileSubjectCommand, RenderCommand


def test_api_exposes_only_control_plane_contract_surface():
    app=create_app()
    paths={route.path for route in app.routes}
    assert "/health/live" in paths
    assert "/v1/contracts/product-invariants" in paths


def test_workflow_commands_bind_exact_product_or_attempt_identity():
    compile_cmd=CompileSubjectCommand("cmd-1","subject-1","attempt-1","release-1","a"*64,"46_PRODUCT_CLOSURE_SEAL")
    render_cmd=RenderCommand("cmd-2","subject-1","revision-7","render-1","b"*64)
    assert compile_cmd.attempt_id=="attempt-1"
    assert render_cmd.product_revision_id=="revision-7"
