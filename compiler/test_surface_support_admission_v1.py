import unittest
from realsas_compiler_core.types import ObservationSample,ObservationEvidenceIR,PersistenceGroup
from realsas_compiler_core.surface import build_surface_from_persistence

class SurfaceSupportTests(unittest.TestCase):
    def test_support_false_cannot_move_fused_point_or_enter_raster_lineage(self):
        a=ObservationSample("a",0,(1.,2.),(0.,0.,0.),(0.,0.,1.),1.,True,"PA")
        b=ObservationSample("b",1,(9.,9.),(0.,0.,0.),(0.,0.,1.),99.,False,"PB")
        s=build_surface_from_persistence(ObservationEvidenceIR((a,b)),(PersistenceGroup("g",("a","b")),))
        n=s.surface_nodes[0]
        self.assertEqual(n.P,(0.,0.,1.))
        self.assertEqual(n.source_observation_ids,("a",))
        self.assertEqual(n.raster_bindings,((0,(1.,2.)),))
        self.assertEqual(n.provenance_refs,("PA",))
        self.assertEqual(n.metadata["support_false_excluded_observation_ids"],("b",))

if __name__=="__main__":
    unittest.main()
