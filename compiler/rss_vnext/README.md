# RSS.vNext compiler · current product path

This directory contains the RSS.vNext-specific **Compiler core courts** (`core/`) and **Platform worker producers/consumers** (`workers/`). The modules were moved byte-for-byte from historical flat locations.

The stable import identities remain `compiler.realsas_compiler_core.rss_vnext_*` and `compiler.realsas_compiler_services.platform_worker.rss_vnext_*`, resolved through their respective package search paths. They are **not** independent copies. See [RELOCATION_MANIFEST_V1.json](RELOCATION_MANIFEST_V1.json) for exact original SHA-1 identities.

Shared canonical mechanics, Graphics, Motion, Proof and artifact primitives still live in `compiler/realsas_compiler_core/` and `compiler/realsas_compiler_services/`; they are dependencies of this unified product candidate, **not obsolete until the remaining consumers and physical path contracts are migrated**. Product selection is owned by the Go capability catalog, not this README.
