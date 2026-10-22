"""IRIS vNext current package boundary.

Authority-bearing consumers import explicit versioned submodules.  The package
initializer intentionally performs no eager model, teacher, cohort, probe, or
experimental-surface imports so stale research implementations cannot become
current merely by importing ``models.iris.vnext``.
"""

__all__: list[str] = []
