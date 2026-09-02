from types import SimpleNamespace
from dataclasses import replace
import pytest
from compiler.realsas_compiler_core.v4_types import ProductProofBundleIR
from compiler.realsas_compiler_core.v4 import proof_bundle_hash
from experiments.single_family_e2e_v1.runtime_bridge_v1 import build_and_consume_reference_runtime_v1

def product():
    dirs=tuple(SimpleNamespace(view_index=i,components=(SimpleNamespace(),)) for i in range(8))
    joint=SimpleNamespace(canonical_joint_id='J:0'); track=SimpleNamespace(canonical_joint_id='J:0',transform_space='PUPPET_LOCAL_2D_2P5D')
    return SimpleNamespace(schema_version='RealSaS.CanonicalPuppetGraph.v3',representation_class='DIRECTIONAL_2D_2P5D_PUPPET',mechanical_equivalence_class='THREE_D_EQUIVALENT_MECHANICS',full_3d_reconstruction_authority=False,product_state_hash='P',directional_renderables=SimpleNamespace(directions=dirs),mechanical_state=SimpleNamespace(skeleton=SimpleNamespace(joints=(joint,))),motion_state=SimpleNamespace(joint_tracks=(track,)))

def proof(state='P',status='PASS'):
    p=ProductProofBundleIR(state,(),(),status,'',metadata={}); return replace(p,proof_bundle_hash=proof_bundle_hash(p))

def test_reference_runtime_requires_current_pass_proof():
    p=product(); runtime,report=build_and_consume_reference_runtime_v1(product=p,proof_bundle=proof())
    assert report['status']=='PASS_REFERENCE_V4_CONSUMPTION' and runtime.source_product_state_hash=='P'
    with pytest.raises(Exception): build_and_consume_reference_runtime_v1(product=p,proof_bundle=proof('OLD'))
    with pytest.raises(Exception): build_and_consume_reference_runtime_v1(product=p,proof_bundle=proof(status='FAIL'))
