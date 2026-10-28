from __future__ import annotations
import json, math
from pathlib import Path
import unittest
from experiments.iris_tessa_v3_apparatus_audit_20261025.fit8_extraction_resolution_selection_v1 import choose_resolution

ROOT=Path(__file__).resolve().parents[2]

class V31ExtractionResolutionSelectionTests(unittest.TestCase):
    def test_prereg_is_sealed_and_fail_closed(self):
        p=json.loads((ROOT/'canonical/V31_EXTRACTION_RESOLUTION_SELECTION_PREREG_V1.json').read_text())
        self.assertEqual(p['status'],'SEALED_BEFORE_SELECTION_EXECUTION')
        self.assertEqual(p['candidate_uniform_resolutions'],[24,32,48,64,96,128,192,256])
        self.assertEqual(p['source_observation_policy']['source_uncertainty_floor_px'],2.0)
        self.assertEqual(p['measurement_binding']['resolved_lower_quantile'],0.05)
        self.assertEqual(p['selection_rule']['policy_form'],'OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION')
        self.assertEqual(p['selection_rule']['ceiling_failure_action'],'LOCAL_SPARSE_OR_HIERARCHICAL_REFINEMENT_REQUIRED__DO_NOT_SILENTLY_FORCE_256')
        self.assertFalse(p['claims']['training_authorized'])
        self.assertFalse(p['claims']['product_shape_activation_authorized'])

    def test_smallest_candidate_satisfying_both_guarantees_is_selected(self):
        candidates=[24,32,48,64,96,128,192,256]
        # R=128 passes: sqrt(2)/128 ~= .01105 and 1/128=.0078125.
        selected=choose_resolution(0.012,0.008,candidates)
        self.assertEqual(selected,128)
        self.assertGreater(math.sqrt(2.0)/96,0.012)

    def test_uniform_ceiling_failure_returns_none(self):
        self.assertIsNone(choose_resolution(0.004,0.003,[24,32,48,64,96,128,192,256]))

    def test_parent_census_is_exactly_pinned_before_execution(self):
        p=json.loads((ROOT/'canonical/V31_EXTRACTION_RESOLUTION_SELECTION_PREREG_V1.json').read_text())
        parent=p['parent_census']
        self.assertEqual(parent['source_code_sha'],'9c7c93b89c54093e1ab7f7c570268cbf9eaadaa3')
        self.assertEqual(parent['sha256'],'4c34dbd6b1118c5d747a8470dcb1e3de604aa5f74adbc5553a133ceb56d297a7')
        self.assertEqual(parent['status'],'FIT8_REAL_MORPHOLOGY_SCALE_CENSUS_COMPLETE_V1')

if __name__=='__main__': unittest.main()
