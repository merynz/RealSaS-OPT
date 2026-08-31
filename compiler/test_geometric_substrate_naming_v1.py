from __future__ import annotations

import unittest

from realsas_compiler_core.bundle_routes import route_for
from realsas_compiler_core.types import ObservationEvidenceIR, RiggingSurfaceIR


class GeometricSubstrateNamingTests(unittest.TestCase):
    def test_new_surface_default_uses_geometric_substrate_assembler(self):
        surface = RiggingSurfaceIR(())
        self.assertEqual(surface.builder_id, "RealSaS.GeometricSubstrateAssembler.current")
        self.assertEqual(route_for(surface).producer, "GeometricSubstrateAssembler")

    def test_observation_route_targets_geometric_substrate_assembler(self):
        evidence = ObservationEvidenceIR(())
        self.assertEqual(route_for(evidence).allowed_consumers, ("GeometricSubstrateAssembler",))

    def test_explicit_historical_builder_identity_is_preserved(self):
        historical = RiggingSurfaceIR((), builder_id="RealSaS.SurfaceBuilder.current")
        self.assertEqual(historical.builder_id, "RealSaS.SurfaceBuilder.current")


if __name__ == "__main__":
    unittest.main()
