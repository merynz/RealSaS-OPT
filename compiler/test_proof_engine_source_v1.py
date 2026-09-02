import numpy as np
from compiler.realsas_compiler_core.proof_engine import mutation_worsens_measurement
from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1, verified_lbs_report_v1, mutate_weights_swap_mass_v1

def test_verified_lbs_weight_mutation_is_causal():
    p=np.array([[0.,0.,0.],[1.,0.,0.]],np.float64); w=np.array([[1.,0.],[0.,1.]],np.float64); T=np.tile(np.eye(4),(1,2,1,1)); T[0,1,0,3]=1.0
    expected=apply_verified_lbs_v1(p,w,T); base=verified_lbs_report_v1(apply_verified_lbs_v1(p,w,T),expected); wm=mutate_weights_swap_mass_v1(w,joint_a=0,joint_b=1,fraction=.5); bad=verified_lbs_report_v1(apply_verified_lbs_v1(p,wm,T),expected)
    assert base.rms==0.0 and bad.rms>base.rms and bad.p95>base.p95
    assert mutation_worsens_measurement('DEFORMATION',{'rms':base.rms,'p95':base.p95},{'rms':bad.rms,'p95':bad.p95})

def test_all_domain_mutation_comparators_are_causal():
    cases={
      'MECHANICAL_STRUCTURE':({'illegal_parent_count':0,'deform_root_count':1},{'illegal_parent_count':1,'deform_root_count':1}),
      'MESH_QUALITY':({'degenerate_faces':0,'min_area':.1},{'degenerate_faces':1,'min_area':0.}),
      'DIRECTIONAL_VISUAL':({'direction_count':8,'corner_binding_count':24},{'direction_count':7,'corner_binding_count':21}),
      'MOTION':({'effective_joint_track_count':1},{'effective_joint_track_count':0}),
      'RUNTIME_CONSUMPTION':({'representation_class':'DIRECTIONAL_2D_2P5D_PUPPET','full_3d_reconstruction_authority':False},{'representation_class':'FULL_3D','full_3d_reconstruction_authority':True}),
    }
    assert all(mutation_worsens_measurement(d,a,b) for d,(a,b) in cases.items())
