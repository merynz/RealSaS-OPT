"""AST-visible connected-pose research adapter; no product authority."""
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v4_impl as impl


def compile_source_domains_stage(ctx):
    return impl.compile_source_domains_stage(ctx)

def compile_projection_stage(ctx):
    return impl.compile_projection_stage(ctx)

def package_stage(ctx):
    return impl.package_stage(ctx)

def playback_stage(ctx):
    return impl.playback_stage(ctx)

def prove_presentation_stage(ctx):
    return impl.prove_presentation_stage(ctx)
