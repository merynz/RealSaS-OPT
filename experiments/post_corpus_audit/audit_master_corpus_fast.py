#!/usr/bin/env python3
"""DEPRECATED RealSaS fast post-corpus auditor.

The first fast implementation applied a render-era `modifiedTime` cutoff to
`primary_geometry.npz`, even though primary geometry had been copied from
normalized variants created before that cutoff. This produced the false
`missing=2398` metadata result despite deep 64/64 geometry loading successfully.

Use `post_corpus_stage_a_closure_v2.py` instead. The corrected runner performs
primary-geometry census without the render timestamp filter and adds exhaustive
observable-coverage, selected provenance/source-container and Gate-2/3 subset
qualification.
"""
raise SystemExit(
    "DEPRECATED: audit_master_corpus_fast.py had a primary_geometry modifiedTime census bug. "
    "Run post_corpus_stage_a_closure_v2.py instead."
)
