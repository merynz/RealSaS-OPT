"""V18a reactive source stain and variable-K graphics candidate redteam.

Synthetic fixtures are NEVER real source art or artist drawable teacher labels.
"""
from __future__ import annotations
import numpy as np
import pytest

from tools.corpus import compile_corpus370_v18a_reactive_staining as stain
from tools.corpus.court_corpus370_v18a_reactive_staining import (
    independent_heldout_fraction,verify_component_connectivity
)


def geometry():
    # Two disjoint source patches, each two triangles sharing one full edge;
    # triangles meeting only at one vertex must not be merged.
    verts=np.array([
        [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
        [4.,0.,0.],[5.,0.,0.],[5.,1.,0.],[4.,1.,0.]
    ],np.float64)
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],np.int64)
    return verts,faces


def original_controls():
    rest,faces=geometry()
    base=np.stack([rest,rest+np.array([0,0,.15]),
                   rest+np.array([0,0,.30])])
    controls=np.broadcast_to(
        base[:,None,None,None,:,:],(3,3,3,2,len(rest),3)
    ).copy()
    for p in range(3):
        # Two mechanically distinguishable source regions; SIGNED effects.
        for a in (0,1):
            s=1. if a==0 else -1.
            controls[p,0,0,a,:4,0]+=s*(.12+.02*p)
            controls[p,1,0,a,4:,1]+=s*(.12+.03*p)
            controls[p,2,0,a,:,2]+=s*(.05+.01*p)
    joint=np.broadcast_to(
        base[:,None,None,:,:],(3,4,4,len(rest),3)
    ).copy()
    for p in range(3):
        for c in range(4):
            joint[p,:,c,:4,0]+=.08*(1 if c in (0,1) else -1)
            joint[p,:,c,4:,1]+=.1*(1 if c in (0,2) else -1)
            joint[p,:,c,0,2]+=.01
    return rest,faces,base,controls,joint


def test_true_shared_edges_never_connect_vertex_only_meshes():
    rest,faces=geometry()
    edges=stain.face_edges(faces)
    assert edges.tolist()==[[0,1],[2,3]]
    labels=stain.component_labels(np.zeros((4,3),np.float32),edges,.1)
    assert labels.tolist()==[0,0,1,1]
    assert verify_component_connectivity(labels,edges)=={0:2,1:2}


def test_reactive_stain_distinguishes_independent_controls_without_semantics():
    rest,faces,base,controls,joint=original_controls()
    picks=stain.select_blind_control_axes(base,controls,faces,6.,budget=4)
    assert len(picks)>=2
    stain.joint_selection_indices=((0,0),(1,0))
    f=stain.face_stain_features(rest,faces,base,controls,joint,picks)
    assert f.shape[0]==4 and f.shape[1]>4
    assert np.isfinite(f).all()
    labels=stain.component_labels(f,stain.face_edges(faces),.1)
    assert len(np.unique(labels))>=2
    assert labels[0]==labels[1] or labels[2]==labels[3]
    assert not stain.heldout_response_error(
        faces,base,controls,picks,labels,6.) is None


def test_fit_channels_are_strictly_blind_to_heldout_source_pose():
    rest,faces,base,controls,joint=original_controls()
    stain.joint_selection_indices=((0,0),(1,0))
    picks=[(0,0),(1,0)]
    original=stain.face_stain_features(rest,faces,base,controls,joint,picks)
    controls_bad=controls.copy()
    joint_bad=joint.copy()
    # Hidden-pose mutation must change *within-component* response variance.
    # A uniform translation of every vertex cannot do that when each group
    # is mechanically homogeneous; that is not a valid heldout falsifier.
    controls_bad[2,0,0,0,1,0]+=.7
    controls_bad[2,1,0,1,5,1]-=.4
    joint_bad[2]+=np.array([-90.,10.,400.])
    after=stain.face_stain_features(rest,faces,base,controls_bad,
                                    joint_bad,picks)
    assert np.array_equal(original,after)
    # Heldout *evaluation* notices the source change, but clustering didn't.
    # Two genuinely multi-face candidate regions: singleton components
    # trivially have zero within-group variance and cannot falsify this gate.
    labels=np.array([0,0,1,1],np.int32)
    a=independent_heldout_fraction(faces,base,controls,picks,labels,6.)
    b=independent_heldout_fraction(faces,base,controls_bad,picks,labels,6.)
    assert a is not None and b is not None and abs(a-b)>1e-7


def test_independent_heldout_bincount_matches_original_evaluator():
    rest,faces,base,controls,joint=original_controls()
    labels=np.array([0,0,1,1],np.int32)
    picks=[(0,0),(1,0)]
    a=stain.heldout_response_error(faces,base,controls,picks,labels,6.)
    b=independent_heldout_fraction(faces,base,controls,picks,labels,6.)
    assert a is not None and b is not None and abs(a-b)<1e-11


def test_source_control_channel_order_does_not_change_graph_partition():
    rest,faces,base,controls,joint=original_controls()
    stain.joint_selection_indices=((0,0),(1,0))
    a=stain.face_stain_features(rest,faces,base,controls,joint,
                                [(0,0),(1,0)])
    b=stain.face_stain_features(rest,faces,base,controls,joint,
                                [(1,0),(0,0)])
    assert np.array_equal(
        stain.component_labels(a,stain.face_edges(faces),.1),
        stain.component_labels(b,stain.face_edges(faces),.1)
    )


def test_reactive_color_count_is_not_artist_ownership():
    rest,faces,base,controls,joint=original_controls()
    stain.joint_selection_indices=((0,0),(1,0))
    f=stain.face_stain_features(rest,faces,base,controls,joint,[(0,0),(1,0)])
    counts=[len(np.unique(stain.component_labels(f,stain.face_edges(faces),t)))
            for t in stain.THRESHOLDS]
    assert len(counts)==3 and all(1<=x<=len(faces) for x in counts)
    assert tuple(sorted(stain.THRESHOLDS))==stain.THRESHOLDS


def test_rigid_root_effect_has_zero_local_rigid_residual():
    rest,faces=geometry()
    shift=np.array([.5,-.2,.4])
    residual=stain.intrinsic_rigid_residual(rest,rest+shift)
    assert np.max(np.abs(residual))<1e-12


def test_nonlinear_local_source_effect_survives_global_rigid_subtraction():
    rest,faces=geometry()
    moved=rest.copy();moved[:4,0]+=.2
    residual=stain.intrinsic_rigid_residual(rest,moved)
    assert np.max(np.linalg.norm(residual,axis=-1))>.01


@pytest.mark.parametrize("mutant",["unconnected_label","cross_edge","invalid_shape"])
def test_false_graph_layer_connection_is_rejected(mutant):
    rest,faces=geometry()
    edges=stain.face_edges(faces)
    if mutant=="unconnected_label":
        labels=np.array([0,0,0,0])
        with pytest.raises(ValueError,match="DISCONNECTED"):
            verify_component_connectivity(labels,edges)
    elif mutant=="cross_edge":
        altered=np.array([[0,2],[1,3]],np.int64)
        assert not np.array_equal(edges,altered)
    else:
        with pytest.raises(ValueError):
            stain.component_labels(np.ones((4,3)),np.zeros((5,3),dtype=np.int64),.1)


@pytest.mark.parametrize("case",["nan","zero_budget","bad_source_shape","all_noop"])
def test_no_hallucinated_source_stains(case):
    rest,faces,base,controls,joint=original_controls()
    if case=="nan":
        control=controls.copy();control[0,0,0,0,1,1]=np.nan
        with pytest.raises(ValueError):
            stain.shape_check(rest,faces,base,control,joint)
    elif case=="zero_budget":
        with pytest.raises(ValueError):
            stain.select_blind_control_axes(base,controls,faces,6.,budget=1)
    elif case=="bad_source_shape":
        with pytest.raises(ValueError):
            stain.shape_check(rest,faces,base,controls[:2],joint)
    else:
        all_noop=np.broadcast_to(base[:,None,None,None,:,:],controls.shape)
        with pytest.raises(ValueError,match="ABSTAIN"):
            stain.select_blind_control_axes(base,all_noop,faces,6.)
