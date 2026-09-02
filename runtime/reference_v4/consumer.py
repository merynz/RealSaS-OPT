from __future__ import annotations


def consume_product_v4_reference(product, proof_bundle, runtime_package):
    if product.schema_version!='RealSaS.CanonicalPuppetGraph.v3': raise ValueError('REFERENCE_RUNTIME_PRODUCT_SCHEMA')
    if product.representation_class!='DIRECTIONAL_2D_2P5D_PUPPET' or product.full_3d_reconstruction_authority: raise ValueError('REFERENCE_RUNTIME_PRODUCT_ONTOLOGY')
    if proof_bundle.source_product_state_hash!=product.product_state_hash or proof_bundle.overall_status!='PASS': raise ValueError('REFERENCE_RUNTIME_REQUIRES_CURRENT_PASS_PROOF')
    if runtime_package.source_product_state_hash!=product.product_state_hash or runtime_package.source_proof_hash!=proof_bundle.proof_bundle_hash: raise ValueError('REFERENCE_RUNTIME_PACKAGE_LINEAGE')
    if len(product.directional_renderables.directions)!=8 or tuple(d.view_index for d in product.directional_renderables.directions)!=tuple(range(8)): raise ValueError('REFERENCE_RUNTIME_DIRECTION_SET')
    if runtime_package.manifest.get('representation_class')!='DIRECTIONAL_2D_2P5D_PUPPET' or runtime_package.manifest.get('full_3d_reconstruction_authority') is not False: raise ValueError('REFERENCE_RUNTIME_MANIFEST_ONTOLOGY')
    joint_ids={j.canonical_joint_id for j in product.mechanical_state.skeleton.joints}
    for track in product.motion_state.joint_tracks:
        if track.canonical_joint_id not in joint_ids or track.transform_space!='PUPPET_LOCAL_2D_2P5D': raise ValueError('REFERENCE_RUNTIME_MOTION_BINDING')
    return {'status':'PASS_REFERENCE_V4_CONSUMPTION','product_state_hash':product.product_state_hash,'proof_hash':proof_bundle.proof_bundle_hash,'direction_count':8,'joint_track_count':len(product.motion_state.joint_tracks),'component_count':sum(len(d.components) for d in product.directional_renderables.directions)}
