"""New decision analysis of immutable PR351 physical renders (no new renders).

Source-private NPZ input, public aggregate output. Neither FULL target images
nor best-match minimization are used to construct a compositor or edit mapping.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

from tools.corpus.drawable_capability_decision_v2 import (
    AUTHORITY, partitions, reachable_states, compile_edit, decide,
    compatible_pairs, feasible_total_orders)

TOL = .025
DEPTH_EPS = .0002
VIEWS = (1, 3, 5, 7)
BITS = ((0,0,0),(1,0,0),(0,1,0),(0,0,1),
        (1,1,0),(1,0,1),(0,1,1),(1,1,1))
FULL_REQUIRED = ('independent_rgb_edits', 'source_compositing',
    'pose_deformation', 'attachment_release', 'secondary_motion',
    'arbitrary_order_override', 'nonbinary_alpha')
PINS = {
 'cat': {
  'array': ('PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V2.npz',
    'd422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114'),
  'manifest': ('PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V2.json',
    '8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1'),
  'court': ('PRIVATE_INDEPENDENT_CAT_SOURCE_UV_EDIT_CONSEQUENCE_COURT_V2.json',
    '90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08')},
 'tric': {
  'array': ('PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.npz',
    '10bd24f5f0f6bfc482cbcbe1169b46251ea4c7f8c8714022fc577816c2ded7ae'),
  'manifest': ('PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.json',
    'a93b777191a40f3c11faa9735b042058b9984436bf59f947093b799a83d11b37'),
  'court': ('PRIVATE_INDEPENDENT_ORIGINAL_SOURCE_K3_SAME_INDEX_COURT_V1.json',
    '021786e2bc9bc28d30d717c08eec3d92e64fca1ebc714a451ab85e2988b4882d')},
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_inputs(root):
    arrays, provenance = {}, {}
    for name, pins in PINS.items():
        for role, (filename, expected) in pins.items():
            if sha(root/name/filename) != expected:
                raise ValueError('INPUT_SHA_MISMATCH:' + name + ':' + role)
        manifest = json.loads((root/name/pins['manifest'][0]).read_text())
        court = json.loads((root/name/pins['court'][0]).read_text())
        if not court['status'].startswith('PASS_') or any(
                d.get(k) is not False for d in (manifest, court) for k in AUTHORITY):
            raise ValueError('PARENT_COURT_OR_AUTHORITY')
        key = ('source_uv_edit_cycles_sample_rgba' if name == 'cat'
               else 'source_original_palette_k3_sample_rgba')
        with np.load(root/name/pins['array'][0], allow_pickle=False) as z:
            a = z[key].copy()
        expected_shape = (4,8,8 if name == 'cat' else 13,192,192,4)
        if a.shape != expected_shape or not np.isfinite(a).all():
            raise ValueError('PHYSICAL_ARRAY_INVALID')
        arrays[name] = a
        provenance[name] = {role: p[1] for role, p in pins.items()}
    return arrays, provenance


def premul(a):
    a = np.asarray(a, dtype=np.float64)
    return np.concatenate((a[...,:3]*a[...,3:4], a[...,3:4]), axis=-1)


def cat_analysis(source):
    counts = np.zeros((4,8,8), np.int64)
    errors = np.zeros((4,8,8), np.float64)
    for v in range(4):
        a = premul(source[v])
        for i in range(8):
            for j in range(i + 1, 8):
                delta = np.max(np.abs(a[:,i]-a[:,j]), axis=-1)
                counts[v,i,j] = counts[v,j,i] = np.count_nonzero(delta > TOL)
                errors[v,i,j] = errors[v,j,i] = np.max(delta)
    domains = {}
    for domain, views in (('side_views_3_5', (1,2)), ('all_four_views', (0,1,2,3))):
        c = counts[list(views)].sum(axis=0)
        e = errors[list(views)].max(axis=0)
        candidates = []
        for groups in partitions(3):
            reach = sorted(BITS.index(bits) for bits in reachable_states(groups))
            requests = []
            for i, q in enumerate(BITS):
                best = min(reach, key=lambda j: (int(c[i,j]), float(e[i,j]), j))
                requests.append({'requested_state': i,
                    'source_address_compilation': compile_edit(groups,q),
                    'compatible_reachable_states': [j for j in reach if c[i,j] == 0],
                    'optimistic_best_state': best,
                    'optimistic_mismatched_sample_pixels': int(c[i,best]),
                    'optimistic_max_error': float(e[i,best])})
            candidates.append({'tied_address_groups': list(groups),
                'control_count': len(set(groups)), 'reachable_states': reach,
                'source_address_status': 'SATISFIED' if len(reach)==8 else 'VIOLATED',
                'observational_status': 'SATISFIED' if all(
                    r['compatible_reachable_states'] for r in requests) else 'VIOLATED',
                'requests': requests})
        domains[domain] = {'candidate_rows': candidates,
            'compatibility_graph': compatible_pairs(e.tolist(),TOL)}
    return {'per_view_pair_mismatch_counts': counts.tolist(),
        'per_view_pair_max_errors': errors.tolist(), 'domains': domains,
        'observational_matching_is_not_an_operation_compiler': True,
        'pose_scope': 'SOURCE_REST_ONLY', 'new_physical_renders': 0}


def depth_plan(depth, rgba):
    if np.any((depth[...,3]>1e-6)&(depth[...,3]<1-1e-6)) or \
       np.any((rgba[...,3]>1e-6)&(rgba[...,3]<1-1e-6)):
        raise ValueError('NONBINARY_ALPHA_OUTSIDE_CONTRACT')
    hit = depth[...,3] > .5
    if np.any((rgba[...,3] > 1e-6) & ~hit):
        raise ValueError('COLOR_WITHOUT_GEOMETRY')
    z = np.where(hit,depth[...,0],np.inf)
    order = np.argsort(z,axis=1,kind='stable')
    sorted_z = np.take_along_axis(z,order,axis=1)
    sorted_hit = np.take_along_axis(hit,order,axis=1)
    diff = np.zeros_like(sorted_z[:,1:])
    np.subtract(sorted_z[:,1:],sorted_z[:,:-1],out=diff,
                where=sorted_hit[:,1:]&sorted_hit[:,:-1])
    tie = np.any(sorted_hit[:,1:]&sorted_hit[:,:-1]&(diff<=DEPTH_EPS),axis=1)
    bad = np.any(hit&((depth[...,0]<.019)|(depth[...,0]>.981)),axis=1)
    return order, ~(tie|bad), hit


def ordered_over(rgba, order):
    """Front-to-back accumulation. order is a global tuple or per-sample tensor."""
    p = premul(rgba)
    if isinstance(order,tuple):
        ordered = p[:,list(order)]
    else:
        ordered = np.take_along_axis(p,order[...,None],axis=1)
    result = np.zeros_like(p[:,0]); transmittance = np.ones_like(result[...,3:4])
    for i in range(p.shape[1]):
        result += transmittance*ordered[:,i]
        transmittance *= 1.-ordered[:,i,...,3:4]
    return result


def metric(prediction, target, qualified):
    d = np.max(np.abs(prediction-target),axis=-1)
    bad = (d>TOL)&qualified
    ix = np.argwhere(bad)
    return {'mismatched_sample_pixels': int(np.count_nonzero(bad)),
        'max_error': float(np.max(d[qualified])),
        'first_failure_s_y_x': ix[0].tolist() if len(ix) else None}


def tric_analysis(source):
    orders = tuple(itertools.permutations(range(3)))
    rows = []
    for vi,view in enumerate(VIEWS):
        a = source[vi].astype(np.float64)
        isolated, depth = a[:,1:4], a[:,4:7]
        plan,q,hit = depth_plan(depth,isolated)
        # Visible precedence only: winning opaque layer precedes any other
        # present opaque layer. Deeper hidden-vs-hidden order is not imposed.
        front = plan[:,0]
        precedence = [[0]*3 for _ in range(3)]
        opaque = isolated[...,3]>.5
        for i in range(3):
            for j in range(3):
                if i != j:
                    precedence[i][j] = int(np.count_nonzero(q&(front==i)&opaque[:,i]&opaque[:,j]))
        per_arm = []
        for op in range(4):
            colors = isolated.copy()
            if op:
                colors[:,op-1] = a[:,6+op]
            target = premul(a[:,0 if op==0 else 9+op])
            per_arm.append({'operation':op,
                'depth_local':metric(ordered_over(colors,plan),target,q),
                'flat_orders':[{'order':list(o), **metric(ordered_over(colors,o),target,q)}
                               for o in orders]})
        edges = [(i,j) for i in range(3) for j in range(3) if precedence[i][j]]
        rows.append({'view':view,'qualified':int(q.sum()),'unknown':int((~q).sum()),
            'visible_precedence_counts':precedence,
            'feasible_visible_precedence_total_orders': [list(o) for o in feasible_total_orders(edges,3)],
            'arms':per_arm})
    totals = [sum(row['arms'][a]['flat_orders'][o]['mismatched_sample_pixels']
              for row in rows for a in range(4)) for o in range(6)]
    per_view_best = [min(sum(arm['flat_orders'][o]['mismatched_sample_pixels']
                         for arm in row['arms']) for o in range(6)) for row in rows]
    depth_failures = sum(arm['depth_local']['mismatched_sample_pixels']
                         for row in rows for arm in row['arms'])
    return {'per_view':rows,'global_order_totals':[
        {'order':list(o),'mismatched_sample_pixels':n} for o,n in zip(orders,totals)],
        'optimistic_best_global_order_error_count':min(totals),
        'optimistic_camera_specific_order_error_counts':per_view_best,
        'depth_local_error_count':depth_failures,
        'depth_local_status':'SATISFIED' if depth_failures==0 else 'VIOLATED',
        'global_order_status':'SATISFIED' if min(totals)==0 else 'VIOLATED',
        'camera_specific_order_status':'SATISFIED' if sum(per_view_best)==0 else 'VIOLATED',
        'pose_scope':'SOURCE_REST_ONLY','new_physical_renders':0}


def representations(cat, tric):
    # Static resource counts, not measured draw calls or latency. Every encoding
    # keeps three independent edit addresses; per-face packaging also needs
    # subface UV masks. Mechanical/graphics partition is deliberately decoupled.
    rows = []
    source = cat['domains']['all_four_views']['candidate_rows']
    for r in source:
        rows.append({'id':'tied_controls_'+''.join(map(str,r['tied_address_groups'])),
            'capabilities':{'independent_rgb_edits':r['source_address_status']},
            'costs':{'logical_drawables':1,'graphic_controls':r['control_count'],
                     'source_edit_addresses':3}})
    for name,k in [('addressed_aggregate',1),('split_shared_carrier',3),
                   ('per_face_with_uv_masks',684),('hybrid_carrier_graphics',1)]:
        rows.append({'id':name,
            'capabilities':{'independent_rgb_edits':'SATISFIED'},
            'costs':{'logical_drawables':k,'graphic_controls':3,'source_edit_addresses':3}})
    return {'cat_edit_contract':decide(rows,('independent_rgb_edits',)),
        'cat_full_puppet_contract':decide(rows,FULL_REQUIRED),
        'scope':'logical edit compiler equivalence; repackaging runtime NOT benchmarked',
        'structural_policy_example':decide(rows,('independent_rgb_edits',),
            ('logical_drawables','graphic_controls','source_edit_addresses')),
        'tric_compositing_contract':decide([
            {'id':k,'capabilities':{'source_compositing':tric[k+'_status']}}
            for k in ('global_order','camera_specific_order','depth_local')],
            ('source_compositing',))}


def run(root, output):
    arrays, provenance = load_inputs(root)
    cat = cat_analysis(arrays['cat'])
    tric = tric_analysis(arrays['tric'])
    result = {'schema':'RealSaS.DrawablePhysicalDecisionAnalysis.v2',
        'provenance':provenance,'preregistration_commit':'134b050191c7f2cc2007d6fbac551d2771420c74',
        'tolerance':TOL,'depth_tie_epsilon':DEPTH_EPS,
        'cat':cat,'triceratops':tric,'decisions':representations(cat,tric),
        'independent_court_required':True,'artist_partition':'NON_IDENTIFIABLE_IN_ADDRESSED_CLASS',
        'same_drawable_pairs':[],'different_drawable_pairs':[],**AUTHORITY}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'written':str(output),'sha256':sha(output),
        'tric_best_flat_errors':tric['optimistic_best_global_order_error_count'],
        'tric_depth_errors':tric['depth_local_error_count']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.inputs,a.output)
