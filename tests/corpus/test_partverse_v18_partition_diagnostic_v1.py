import json
import unittest
import numpy as np
import trimesh

from tools.corpus.truth_foundry.partverse_v18_partition_diagnostic_v1 import analyze, analyze_bytes

class PartVerseV18PartitionDiagnosticTest(unittest.TestCase):
    def test_same_topology_two_part_labels_not_same_drawable(self):
        m=trimesh.creation.box()
        n=len(m.faces)
        labels={str(i): (0 if i<n//2 else 1) for i in range(n)}
        r=analyze([m],labels)
        self.assertEqual(r['processed_topology_islands'],1)
        self.assertEqual(r['partverse_part_ids_including_zero'],2)
        self.assertGreater(r['adjacent_pairs_different_part_label'],0)
        self.assertEqual(r['native_artist_drawable_ownership'],'UNKNOWN')
        self.assertFalse(r['training_authorized'])

    def test_one_part_can_span_disconnected_islands(self):
        a=trimesh.creation.box()
        b=trimesh.creation.box()
        face_map={str(i):0 for i in range(len(a.faces)+len(b.faces))}
        r=analyze([a,b],face_map)
        self.assertEqual(r['processed_topology_islands'],2)
        self.assertEqual(r['partverse_part_ids_including_zero'],1)
        self.assertEqual(r['part_labels_spanning_multiple_topo_islands'],1)

    def test_face_map_gap_fails_closed(self):
        m=trimesh.creation.box()
        d={str(i):1 for i in range(len(m.faces)) if i!=3}
        with self.assertRaisesRegex(ValueError,'CONTIGUITY'):
            analyze([m],d)

    def test_boolean_part_id_fails_closed(self):
        m=trimesh.creation.box()
        d={str(i):0 for i in range(len(m.faces))};d['2']=True
        with self.assertRaisesRegex(ValueError,'INVALID_PART_ID'):
            analyze([m],d)

    def test_real_glb_like_roundtrip_stays_source_unknown(self):
        m=trimesh.creation.box()
        encoded=m.export(file_type='glb')
        data=json.dumps({str(i):i%2 for i in range(len(m.faces))}).encode()
        r=analyze_bytes(encoded,data)
        self.assertEqual(r['glb_faces'],len(m.faces))
        self.assertEqual(r['processed_glb_to_original_source_face_order'],'UNKNOWN')
        self.assertEqual(r['semantic_concept_court'],'NOT_RUN')

if __name__=='__main__':unittest.main()