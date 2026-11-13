# RSS.vNext expert implementations

All four learned specialist families are physically co-located here as components of **one unified RSSVNEXT ModelRelease**. Production inference selects the single backbone under `models/rss_vnext/`. The folders `iris/`, `tessa/`, `axis/`, `mira/` are internal experts, not independently registered ModelReleases.

Compatibility: `models/__init__.py` appends this directory to the package search path, preserving the original Python names `models.iris.*`, `models.tessa.*`, `models.axis.*`, `models.mira.*`. The [manifest](../EXPERT_SOURCE_RELOCATION_V1.json) records original and current content identities. Loading and implementation-closure hashing must resolve the **real** file under `experts/`, not a synthetic shim.
