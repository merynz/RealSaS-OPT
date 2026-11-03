"""Versioned, bounded causal measurements in an observable oriented-normal frame.

V1's QR basis and its independent court's edge basis disagree on handedness.
V2 declares its basis and uses the intersection of the old area and conditioning
guards. No change is made to the legacy presentation/TESSA response definition.
"""
from __future__ import annotations

import numpy as np
from models.iris.truth import presentation_response_v1 as source_contract

TANGENT_FRAME = 'SOURCE_WORLD_NORMAL_PROJECTED_LEAST_ALIGNED_AXIS_V1'
VALIDITY_POLICY = 'ALL_REST_ALTITUDES_REST_POSED_AREA_POSITIVE_METRIC_V1'


def _measure_block(rest, faces, posed, weights, tolerance):
    r = rest[faces]
    e1, e2 = r[:,1]-r[:,0], r[:,2]-r[:,0]
    cross = np.cross(e1, e2); twice_area = np.linalg.norm(cross, axis=1)
    edge_max = np.maximum.reduce([np.linalg.norm(e1,axis=1), np.linalg.norm(e2,axis=1), np.linalg.norm(e2-e1,axis=1)])
    height = np.divide(twice_area, edge_max, out=np.zeros_like(twice_area), where=edge_max>0)
    rest_valid = (twice_area*.5 > tolerance) & (height*height > tolerance)
    normal = np.divide(cross, twice_area[:,None], out=np.zeros_like(cross), where=twice_area[:,None]>0)
    # X, then Y, then Z break exact ties. A cyclic face reindex has the same frame.
    axis = np.eye(3)[np.argmin(np.abs(normal),axis=1)]
    u = axis - np.sum(axis*normal,axis=1)[:,None]*normal
    u /= np.maximum(np.linalg.norm(u,axis=1)[:,None], 1e-300)
    v = np.cross(normal,u)
    R = np.stack([np.stack([np.sum(e1*u,axis=1),np.sum(e2*u,axis=1)],axis=-1),
                  np.stack([np.sum(e1*v,axis=1),np.sum(e2*v,axis=1)],axis=-1)],axis=1)
    R[~rest_valid] = np.eye(2)
    p = posed[:,faces,:]
    pe1, pe2 = p[:,:,1]-p[:,:,0], p[:,:,2]-p[:,:,0]
    F = np.stack([pe1,pe2],axis=-1) @ np.linalg.inv(R)
    C = np.swapaxes(F,-1,-2) @ F
    eigval,eigvec = np.linalg.eigh(C)
    posed_area = .5*np.linalg.norm(np.cross(pe1,pe2),axis=-1)
    valid = rest_valid[None,:] & (posed_area>tolerance) & np.isfinite(eigval).all(axis=-1) & (eigval[...,0]>1e-14)
    stretches = np.sqrt(np.maximum(eigval,0.))
    minor,major = stretches[...,0],stretches[...,1]
    area = np.sqrt(np.maximum(np.linalg.det(C),0.))
    anisotropy = major-minor
    direction = eigvec[...,:,1]
    # Double angle directly from the unit eigenvector; no sign choice is needed.
    tangent = np.stack([direction[...,0]**2-direction[...,1]**2, 2*direction[...,0]*direction[...,1]],axis=-1)
    wm = weights[:,None]; mass = np.sum(wm*valid,axis=0); safe = np.where(mass>0,mass,1.)
    mean = lambda x: np.sum(wm*np.where(valid,x,0.),axis=0)/safe
    vector = np.sum((wm*np.where(valid,anisotropy,0.))[...,None]*tangent,axis=0)
    length = np.linalg.norm(vector,axis=1)
    tangent_valid = (mass>0) & (mean(anisotropy)>1e-5) & (length>1e-8)
    tangent_mean = np.zeros((len(faces),2)); tangent_mean[tangent_valid] = vector[tangent_valid]/length[tangent_valid,None]
    nonrigid = np.sqrt(np.sum(wm*np.where(valid,(major-1.)**2+(minor-1.)**2,0.),axis=0)/np.maximum(mass,1e-12))
    response = p.mean(axis=2)-r.mean(axis=1)[None,:,:]
    energy = np.sqrt(np.sum(weights[:,None,None]*response*response,axis=(0,2)))
    return {'surface_face_valid':mass>0, 'surface_stretch_major_mean':mean(major),
            'surface_stretch_minor_mean':mean(minor), 'surface_area_ratio_mean':mean(area),
            'surface_anisotropy_mean':mean(anisotropy), 'surface_tangent_valid':tangent_valid,
            'surface_tangent_direction_2theta':tangent_mean, 'surface_nonrigid_response':nonrigid,
            'surface_response_energy':energy}, response


def build_surface_response_arrays_v2(rest_vertices, faces, isolated_position, actions, *, source_scale=None, face_chunk_size=128):
    if type(face_chunk_size) is not int or face_chunk_size<1:
        raise ValueError('CAUSAL_V2_POSITIVE_CHUNK_REQUIRED')
    rest,tri = source_contract._mesh(rest_vertices,faces)
    posed = np.asarray(isolated_position,dtype=np.float64)
    if posed.ndim!=3 or posed.shape[1:]!=rest.shape or not np.isfinite(posed).all() or len(posed)!=len(actions):
        raise ValueError('CAUSAL_V2_ACTION_POSITION_LAYOUT')
    scale = source_contract._scale(rest) if source_scale is None else float(source_scale)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('CAUSAL_V2_SCALE')
    weights = source_contract.control_balanced_action_weights_v1(actions)
    chunks = {}; response = np.empty((len(posed),len(tri),3))
    for start in range(0,len(tri),face_chunk_size):
        stop = min(start+face_chunk_size,len(tri))
        values,block_response = _measure_block(rest,tri[start:stop],posed,weights,1e-10*scale*scale)
        for name,value in values.items(): chunks.setdefault(name,[]).append(value)
        response[:,start:stop] = block_response
    result = {name:np.concatenate(values) for name,values in chunks.items()}
    pairs = source_contract._face_adjacency(tri)
    coherence = np.zeros(len(pairs)); jump = np.zeros(len(pairs)); valid = np.zeros(len(pairs),bool)
    flat = np.transpose(response*np.sqrt(weights)[:,None,None],(1,0,2)).reshape(len(tri),-1)
    norm = np.linalg.norm(flat,axis=1)
    for i,(a,b) in enumerate(pairs):
        denominator = norm[a]*norm[b]
        if denominator>1e-12:
            c = float(np.clip(np.dot(flat[a],flat[b])/denominator,-1.,1.))
            coherence[i],jump[i],valid[i] = c,.5*(1.-c),True
    result.update(action_weight=weights,response_pair_face=pairs,response_pair_valid=valid,
                  response_pair_coherence=coherence,response_pair_jump=jump)
    return result
