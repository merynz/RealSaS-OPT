"""Geometric local response under an explicit, transported action measure.

J[locus,xyz,control] is a measured local kinematic derivative, not a bone label
or physical stiffness. B[control,measure_axis] declares Sigma=B B^T, retaining
the response magnitude. The primary pair tensor J_a Sigma J_b^T contains no
teacher control index. Under J'=J A and B'=A^-1 B it is unchanged. A duplicate
control without transporting the law legitimately changes the object; it must
not be normalized away silently. Native clips alone do not provide this J.
"""
from __future__ import annotations
import numpy as np


def central_kinematic_derivative(positive, negative, amplitude, valid):
    p,n=np.asarray(positive,dtype=np.float64),np.asarray(negative,dtype=np.float64)
    h=np.asarray(amplitude,dtype=np.float64);mask=np.asarray(valid)
    if (p.ndim!=3 or p.shape!=n.shape or p.shape[-1]!=3 or min(p.shape[:2])<=0
            or h.shape!=(p.shape[1],) or not np.isfinite(h).all() or (h<=0).any()
            or mask.dtype!=np.bool_ or mask.shape!=p.shape[:2]):
        raise ValueError('RESPONSE_OPERATOR_CENTRAL_PAIR_LAYOUT')
    if not np.isfinite(p[mask]).all() or not np.isfinite(n[mask]).all():
        raise ValueError('RESPONSE_OPERATOR_VALID_RESPONSE_NONFINITE')
    derivative=(p-n)/(2*h[None,:,None])
    return np.where(mask[:,None,:],derivative.transpose(0,2,1),np.nan),mask


def measured_pair_response_tensor(jacobian, action_measure_factor, pairs, valid):
    j=np.asarray(jacobian,dtype=np.float64);factor=np.asarray(action_measure_factor,dtype=np.float64)
    pairs=np.asarray(pairs);valid=np.asarray(valid)
    if (j.ndim!=3 or j.shape[1]!=3 or min(j.shape[::2])<=0
            or factor.ndim!=2 or factor.shape[0]!=j.shape[2] or factor.shape[1]<=0
            or not np.isfinite(factor).all() or not np.any(factor)
            or pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in 'iu'
            or (pairs.size and (pairs.min()<0 or pairs.max()>=j.shape[0]))
            or valid.dtype!=np.bool_ or valid.shape!=(j.shape[0],j.shape[2])):
        raise ValueError('RESPONSE_OPERATOR_LAYOUT_OR_NONZERO_MEASURE')
    if not np.isfinite(j.transpose(0,2,1)[valid]).all():
        raise ValueError('RESPONSE_OPERATOR_VALID_DERIVATIVE_NONFINITE')
    excited=np.any(factor!=0,axis=1)
    locus_valid=(valid | ~excited[None,:]).all(axis=1)
    safe=np.where(valid[:,None,:],j,0)
    response=safe@factor
    cross=response[pairs[:,0]]@response[pairs[:,1]].transpose(0,2,1)
    mask=locus_valid[pairs[:,0]]&locus_valid[pairs[:,1]]
    return np.where(mask[:,None,None],cross,np.nan),mask
