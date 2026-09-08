from __future__ import annotations
import json, re
from pathlib import Path

ROOT=Path.cwd()
EXP=ROOT/'experiments/arachne_skintokens_fit1_20260908'
summary=json.loads((EXP/'ARACHNE_MAGE_A0_FS1_FAILURE_DIAGNOSTIC_SUMMARY_V1.json').read_text())
kin=json.loads((EXP/'ARACHNE_MAGE_A0_FS1_KINEMATIC_ERROR_ROUTING_DIAGNOSTIC_V1.json').read_text())
loss_src=(ROOT/'models/skin_field_codec/v1/skin_field_codec_v1.py').read_text()

# Frozen sealed-evidence guards.
g=summary['global']; bc=summary['bad_complexity_vs_good']
assert g['rows_gt_0p05']==165
assert g['confidence_counts_bad']['0']==158
assert abs(g['row_l1_p95']-0.11468102738268096)<2e-9
assert abs(g['row_l1_mean']-0.026040060552825844)<2e-9
assert kin['bad_rows']==165
assert kin['top_pair_tree_distance_counts']=={'1':129,'2':33,'3':3}
assert kin['abs_error_mass_within_tree_distance_1_of_teacher_active']>0.978
assert kin['abs_error_mass_within_tree_distance_2_of_teacher_active']>0.997
assert bc['bad_active_count_mean']>bc['good_active_count_mean']+0.8
assert bc['bad_entropy_mean']>bc['good_entropy_mean']+0.3

# Inspect the actual loss implementation, not prose.
m=re.search(r'def skin_field_codec_loss_v1\(.*?\):(.+?)(?:\n\s*def |\Z)',loss_src,re.S)
assert m, 'LOSS_FUNCTION_NOT_FOUND'
body=m.group(1)
pair_mean_ce=('weighted.sum()/pair_mask.sum' in body.replace(' ',''))
pair_mean_l1=('sum()/pair_mask.sum' in body.replace(' ','') and 'torch.abs(decoded_weights-teacher_weights)' in body.replace(' ',''))
tail_term=any(tok in body.lower() for tok in ('topk','quantile','percentile','cvar','hard_row','tail'))
assert pair_mean_ce and pair_mean_l1
assert not tail_term

# Root-cause logic is intentionally categorical and non-authorizing.
classification='BLEND_RATIO_TAIL_UNDERFIT__OBJECTIVE_GATE_MISALIGNMENT_PRIMARY__CAPACITY_STILL_UNRESOLVED'
reasons=[
  'mean row-L1 is low while p95 fails by >2x, so failure is tail-concentrated rather than global',
  'bad rows have materially higher teacher entropy and active-joint count than good rows',
  '97.86% of absolute error mass stays within one skeleton edge of teacher-active joints',
  'only 8/165 bad rows change dominant joint, so gross anatomical assignment is not the failure mode',
  'the frozen training loss averages CE and L1 over all valid surface-joint pairs and contains no p95/top-k/CVaR hard-row term',
  'deformation passed, consistent with local blend-ratio error between adjacent joints rather than wrong-body-part routing',
]
report={
 'schema':'RealSaS.ArachneMageA0FS1HostedRootCauseDiagnostic.v1',
 'status':'PASS_HOSTED_DIAGNOSTIC__NON_AUTHORIZING',
 'classification':classification,
 'a0_verdict_unchanged':'NO_A0_TERMINAL_CLOSURE',
 'a1_authorized':False,
 'evidence':{
   'row_l1_mean':g['row_l1_mean'],'row_l1_p95':g['row_l1_p95'],'rows_gt_0p05':g['rows_gt_0p05'],
   'high_confidence_bad_rows':g['confidence_counts_bad']['0'],
   'bad_active_count_mean':bc['bad_active_count_mean'],'good_active_count_mean':bc['good_active_count_mean'],
   'bad_entropy_mean':bc['bad_entropy_mean'],'good_entropy_mean':bc['good_entropy_mean'],
   'dominant_joint_mismatch_bad_rows':g['dominant_joint_mismatch_bad_rows'],
   'error_mass_within_tree_distance_1':kin['abs_error_mass_within_tree_distance_1_of_teacher_active'],
   'error_mass_within_tree_distance_2':kin['abs_error_mass_within_tree_distance_2_of_teacher_active'],
   'loss_pair_mean_ce':pair_mean_ce,'loss_pair_mean_l1':pair_mean_l1,'loss_has_tail_term':tail_term,
 },
 'reasons':reasons,
 'remaining_uncertainty':'Frozen-decoder capacity versus encoder/latent inference cannot be cleanly separated until the preregistered oracle-latent diagnostic completes. The present evidence already rules against low-confidence truth and gross anatomical confusion as primary causes.',
 'next_test':'Complete the preregistered frozen-decoder oracle-latent diagnostic before changing architecture, thresholds, or A1 authorization.'
}
out=Path('/tmp/ARACHNE_MAGE_A0_FS1_HOSTED_ROOT_CAUSE_DIAGNOSTIC_V1.json')
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('A0_HARDROW_ROOT_CAUSE='+classification,flush=True)
for r in reasons: print(' - '+r,flush=True)
print('CAPACITY_STATUS=UNRESOLVED_PENDING_ORACLE_LATENT',flush=True)
print(json.dumps(report,indent=2,sort_keys=True),flush=True)
