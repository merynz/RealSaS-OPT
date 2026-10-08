"""AST-visible public adapter surface for scoped presentation V2 research.

The implementation lives in ``presentation_research_v2_impl``. These explicit
wrappers are intentional: EngineRelease snapshotting validates adapter callables
from source/AST without importing runtime-heavy modules, so each DAG entrypoint
must be visible in this module's source while its import closure still binds the
complete implementation bytes.
"""
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v2_impl as impl


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
