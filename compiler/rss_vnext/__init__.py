"""RSS.vNext Compiler source home. Import through the stable canonical IR API.

core/ and workers/ store the single product candidate's specialized modules.
Existing `compiler.realsas_compiler_core.rss_vnext_*` and
`compiler.realsas_compiler_services.platform_worker.rss_vnext_*` module IDs
are retained by their package search paths to avoid serialization drift.
"""
