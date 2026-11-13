# Historical model implementation sources

These **byte-identical** Geppetto, Arachne and SkinFieldCodec modules are retained for scientific replay and old checkpoint import identity, not as current product models. Their previous locations and exact Git blob SHA-1 are recorded in [RELOCATION_MANIFEST_V1.json](RELOCATION_MANIFEST_V1.json).

`models/__init__.py` appends this directory to the `models` package search path so historical imports such as `models.geppetto.reference_strength_v1` remain resolvable without a second model release. This does **not** authorize old checkpoint results for RSS.vNext. Prefer `models/rss_vnext/` for all new product code.
