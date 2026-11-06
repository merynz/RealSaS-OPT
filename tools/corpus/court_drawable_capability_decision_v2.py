"""Independent counter-court: no imports from producer or decision mechanism.

Recomputes physical candidate failures, admissible alternatives and authority
boundaries from original SHA-pinned source measurements. Parent physical render
apparatus is shared evidence; this is not independent Blender re-execution.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def require(condition, reason):
    if not condition:
        raise ValueError('DECISION_INDEPENDENT_COURT:' + reason)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def converted(raw):
    value=raw.astype(np.float64).copy()
    for channel in (0,1,2):
        value[...,channel] *= value[...,3]
    return value


def check_measurement(predicted, actual, valid, reported):
    per_channel=np.abs(actual-predicted)
    wrong=np.any(per_channel>.025,axis=-1)&valid
    count=int(wrong.sum())
    peak=float(per_channel[valid].max())
    require(count==reported['mismatched_sample_pixels'],'PHYSICAL_ERROR_COUNT')
    require(abs(peak-reported['max_error'])<1e-12,'PHYSICAL_PEAK_ERROR')
    locations=np.argwhere(wrong)
    first=locations[0].tolist() if len(locations) else None
    require(first==reported['first_failure_s_y_x'],'COUNTEREXAMPLE_ADDRESS')
    return count


def validate_decision_tables(report):
    """Independent scalar authority/policy audit, also exercised by adversaries."""
    authority_keys=('graphics_owner_authorized','iris_training_authorized',
        'tessa_training_authorized','materialization_authorized','training_authorized')
    require(all(report.get(k) is False for k in authority_keys),'AUTHORITY_PROMOTION')
    require(report.get('same_drawable_pairs')==[] and report.get('different_drawable_pairs')==[],
            'OWNER_LABELS')
    groupings=('000','001','010','011','012')
    ids=['tied_controls_'+g for g in groupings]+['addressed_aggregate',
        'split_shared_carrier','per_face_with_uv_masks','hybrid_carrier_graphics']
    costs={}
    for g in groupings:
        costs['tied_controls_'+g]={'logical_drawables':1,'graphic_controls':len(set(g)),
                                  'source_edit_addresses':3}
    for name,k in [('addressed_aggregate',1),('split_shared_carrier',3),
                   ('per_face_with_uv_masks',684),('hybrid_carrier_graphics',1)]:
        costs[name]={'logical_drawables':k,'graphic_controls':3,'source_edit_addresses':3}
    good=set(ids[4:])
    full=('independent_rgb_edits','source_compositing','pose_deformation',
          'attachment_release','secondary_motion','arbitrary_order_override','nonbinary_alpha')
    for key in ('cat_edit_contract','cat_full_puppet_contract','structural_policy_example',
                'tric_compositing_contract'):
        table=report['decisions'][key]
        require(table['policy_is_artist_ownership'] is False,'POLICY_AS_OWNER')
        require(table['same_drawable_pairs']==[] and table['different_drawable_pairs']==[],
                'OWNER_LABELS')
        require(all(table.get(k) is False for k in authority_keys),'AUTHORITY_PROMOTION')
        rows=table['candidate_decisions']
        if key=='tric_compositing_contract':
            require([r['id'] for r in rows]==['global_order','camera_specific_order','depth_local'],
                    'DECISION_CANDIDATES')
            for r in rows:
                require(r['capabilities']=={'source_compositing':report['triceratops'][r['id']+'_status']},
                        'DECISION_EVIDENCE')
                require(r['costs']=={},'UNMEASURED_COST')
        else:
            require([r['id'] for r in rows]==ids,'DECISION_CANDIDATES')
            for r in rows:
                expected={'independent_rgb_edits':'SATISFIED' if r['id'] in good else 'VIOLATED'}
                if key=='cat_full_puppet_contract':
                    expected.update({k:'UNKNOWN' for k in full[1:]})
                require(r['capabilities']==expected,'UNMEASURED_CAPABILITY_PROMOTION')
                require(r['costs']==costs[r['id']],'STRUCTURAL_COST_DRIFT')
        satisfying=[];unknown=[]
        for r in rows:
            values=list(r['capabilities'].values())
            state='VIOLATED' if 'VIOLATED' in values else 'UNKNOWN' if 'UNKNOWN' in values else 'SATISFIED'
            require(r['status']==state,'DECISION_STATUS')
            if state=='SATISFIED':satisfying.append(r['id'])
            if state=='UNKNOWN':unknown.append(r['id'])
        require(table['satisfying_alternatives']==satisfying and table['unresolved_alternatives']==unknown,
                'ALTERNATIVES_DROPPED')
        axes=['logical_drawables','graphic_controls','source_edit_addresses'] if key=='structural_policy_example' else []
        require(table['policy_cost_axes']==axes,'POLICY_DRIFT')
        frontier=[]
        for r in rows:
            if r['id'] not in satisfying:continue
            dominated=False
            for o in rows:
                if o['id'] in satisfying and axes and all(o['costs'][a]<=r['costs'][a] for a in axes) and any(
                        o['costs'][a]<r['costs'][a] for a in axes):
                    dominated=True
            if not dominated:frontier.append(r['id'])
        require(table['policy_pareto_frontier']==frontier,'PARETO_POLICY')


def verify(root, report_path, output):
    report=json.loads(report_path.read_text())
    require(report['schema']=='RealSaS.DrawablePhysicalDecisionAnalysis.v2','SCHEMA')
    require(report['tolerance']==.025 and report['depth_tie_epsilon']==.0002,'POLICY_DRIFT')
    require(report['preregistration_commit']=='134b050191c7f2cc2007d6fbac551d2771420c74',
            'PREREGISTRATION_IDENTITY')
    validate_decision_tables(report)
    pins={
      'cat':(
        ('PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V2.npz','d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114'),
        ('PRIVATE_CAT_SOURCE_UV_SAME_CYCLES_SAMPLES_V2.json','8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1'),
        ('PRIVATE_INDEPENDENT_CAT_SOURCE_UV_EDIT_CONSEQUENCE_COURT_V2.json','90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08')),
      'tric':(
        ('PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.npz','10bd24f5f0f6bfc482cbcbe1169b46251ea4c7f8c8714022fc577816c2ded7ae'),
        ('PRIVATE_ORIGINAL_SOURCE_K3_SAME_CYCLES_SAMPLES_V1.json','a93b777191a40f3c11faa9735b042058b9984436bf59f947093b799a83d11b37'),
        ('PRIVATE_INDEPENDENT_ORIGINAL_SOURCE_K3_SAME_INDEX_COURT_V1.json','021786e2bc9bc28d30d717c08eec3d92e64fca1ebc714a451ab85e2988b4882d'))}
    for name,entries in pins.items():
        for role,(filename,sha) in zip(('array','manifest','court'),entries):
            require(digest(root/name/filename)==sha==report['provenance'][name][role],
                    'ORIGINAL_SOURCE_SHA')
    with np.load(root/'cat'/pins['cat'][0][0],allow_pickle=False) as z:
        cat=z['source_uv_edit_cycles_sample_rgba']
    require(cat.shape==(4,8,8,192,192,4) and np.isfinite(cat).all(),'CAT_SHAPE')
    numbers=np.zeros((4,8,8),dtype=np.int64)
    maxima=np.zeros((4,8,8))
    for v in range(4):
        pixels=converted(cat[v]).transpose(1,0,2,3,4).reshape(8,-1,4)
        for a in range(8):
            for b in range(8):
                error=np.abs(pixels[a]-pixels[b])
                numbers[v,a,b]=int(np.any(error>.025,axis=1).sum())
                maxima[v,a,b]=error.max()
    require(numbers.tolist()==report['cat']['per_view_pair_mismatch_counts'],'CAT_PAIR_COUNTS')
    require(np.array_equal(maxima,np.asarray(report['cat']['per_view_pair_max_errors'])),
            'CAT_PAIR_ERRORS')
    states=((0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1))
    groupings=((0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,1,2))
    for key,indices in (('side_views_3_5',[1,2]),('all_four_views',[0,1,2,3])):
        rows=report['cat']['domains'][key]['candidate_rows']
        require([tuple(r['tied_address_groups']) for r in rows]==list(groupings),'CANDIDATE_EXHAUSTIVENESS')
        counts=numbers[indices].sum(axis=0); peaks=maxima[indices].max(axis=0)
        for row,g in zip(rows,groupings):
            # Independently derive legality from all constrained address pairs.
            legal=[i for i,s in enumerate(states) if all(
                g[a]!=g[b] or s[a]==s[b] for a in range(3) for b in range(3))]
            require(legal==row['reachable_states'],'CONTROL_REACHABILITY')
            require(row['control_count']==len(set(g)),'CONTROL_COST')
            require(row['source_address_status']==('SATISFIED' if len(legal)==8 else 'VIOLATED'),
                    'ADDRESS_CAPABILITY')
            outcomes=[]
            for i,r in enumerate(row['requests']):
                require(r['requested_state']==i,'REQUEST_ORDER')
                require(r['source_address_compilation']['status']==
                    ('SATISFIED' if i in legal else 'VIOLATED'),'SOURCE_OPERATION_CONFLICT')
                if i not in legal:
                    a,b=r['source_address_compilation']['conflicting_addresses']
                    require(g[a]==g[b] and states[i][a]!=states[i][b],'CONFLICT_WITNESS')
                else:
                    require(r['source_address_compilation']['result_address_state']==list(states[i]),
                            'COMPILED_OPERATION_DRIFT')
                matches=[j for j in legal if counts[i,j]==0]
                best=sorted(legal,key=lambda j:(int(counts[i,j]),float(peaks[i,j]),j))[0]
                require(r['compatible_reachable_states']==matches and r['optimistic_best_state']==best,
                        'OBSERVATION_MATCHES')
                require(r['optimistic_mismatched_sample_pixels']==int(counts[i,best]) and
                    r['optimistic_max_error']==float(peaks[i,best]),'OPTIMISTIC_LOWER_BOUND')
                outcomes.append(bool(matches))
            require(row['observational_status']==('SATISFIED' if all(outcomes) else 'VIOLATED'),
                    'OBSERVATIONAL_VERDICT')
        graph=report['cat']['domains'][key]['compatibility_graph']
        expected=[{'a':i,'b':j,'status':'COMPATIBLE' if peaks[i,j]<=.025 else 'DISTINGUISHED'}
                  for i in range(8) for j in range(i+1,8)]
        require(graph==expected,'COMPATIBILITY_GRAPH')
    del cat
    with np.load(root/'tric'/pins['tric'][0][0],allow_pickle=False) as z:
        tric=z['source_original_palette_k3_sample_rgba']
    require(tric.shape==(4,8,13,192,192,4) and np.isfinite(tric).all(),'TRIC_SHAPE')
    totals=np.zeros(6,dtype=np.int64); best_by_view=[]; depth_total=0
    orders=list(itertools.permutations((0,1,2)))
    for v,row in enumerate(report['triceratops']['per_view']):
        require(row['view']==[1,3,5,7][v],'CAMERA_IDENTITY')
        raw=tric[v];depth=raw[:,4:7]
        hits=depth[...,3]>.5;z=depth[...,0]
        require(np.all(np.isin(raw[:,1:10,...,3],(0.,1.))),'EXACT_BINARY_ALPHA_SCOPE')
        valid=np.ones((8,192,192),dtype=bool)
        for k in range(3):
            valid &= ~(hits[:,k]&((z[:,k]<.019)|(z[:,k]>.981)))
            for j in range(k):
                valid &= ~(hits[:,k]&hits[:,j]&(np.abs(z[:,k]-z[:,j])<=.0002))
        require(int(valid.sum())==row['qualified'] and int((~valid).sum())==row['unknown'],
                'QUALIFICATION_AND_UNKNOWN')
        # No argsort: independent direct winner test against every other hit.
        nearest=[]
        for k in range(3):
            wins=hits[:,k].copy()
            for j in range(3):
                if j!=k: wins &= ~hits[:,j]|(z[:,k]<z[:,j])
            nearest.append(wins)
        precedence=[[0]*3 for _ in range(3)]
        for k in range(3):
            for j in range(3):
                if j!=k:
                    precedence[k][j]=int(np.count_nonzero(valid&nearest[k]&
                        (raw[:,1+k,...,3]>.5)&(raw[:,1+j,...,3]>.5)))
        require(precedence==row['visible_precedence_counts'],'VISIBLE_ORDER_COUNTS')
        legal_orders=[list(o) for o in orders if all(o.index(i)<o.index(j)
            for i in range(3) for j in range(3) if precedence[i][j])]
        require(legal_orders==row['feasible_visible_precedence_total_orders'],'ORDER_CONSTRAINT_SOLVER')
        view_totals=np.zeros(6,dtype=np.int64)
        for op,arm in enumerate(row['arms']):
            require(arm['operation']==op,'ARM_IDENTITY')
            colors=converted(raw[:,1:4])
            if op: colors[:,op-1]=converted(raw[:,6+op])
            target=converted(raw[:,0 if op==0 else 9+op])
            prediction=np.zeros_like(target)
            for k in range(3): prediction[nearest[k]]=colors[:,k][nearest[k]]
            depth_total+=check_measurement(prediction,target,valid,arm['depth_local'])
            for oi,o in enumerate(orders):
                require(arm['flat_orders'][oi]['order']==list(o),'TOTAL_ORDER_IDENTITY')
                # Different algorithm: back-to-front OVER.
                prediction=np.zeros_like(target)
                for k in reversed(o):
                    prediction=colors[:,k]+(1-colors[:,k,...,3:4])*prediction
                view_totals[oi]+=check_measurement(prediction,target,valid,arm['flat_orders'][oi])
        totals+=view_totals;best_by_view.append(int(min(view_totals)))
    t=report['triceratops']
    require(t['global_order_totals']==[{'order':list(o),'mismatched_sample_pixels':int(n)}
                                    for o,n in zip(orders,totals)],'GLOBAL_TOTALS')
    require(t['optimistic_best_global_order_error_count']==int(min(totals)) and
            t['optimistic_camera_specific_order_error_counts']==best_by_view and
            t['depth_local_error_count']==depth_total,'CANDIDATE_TOTALS')
    for name,n in [('global_order',min(totals)),('camera_specific_order',sum(best_by_view)),
                   ('depth_local',depth_total)]:
        require(t[name+'_status']==('SATISFIED' if n==0 else 'VIOLATED'),'COMPOSITOR_VERDICT')
    # Recursively reject authority promotion and any fabricated pair labels.
    def authority(node):
        if isinstance(node,dict):
            for k,v in node.items():
                if k.endswith('_authorized'):require(v is False,'AUTHORITY_PROMOTION')
                if k in ('same_drawable_pairs','different_drawable_pairs'):require(v==[],'OWNER_LABELS')
                authority(v)
        elif isinstance(node,list):
            for item in node:authority(item)
    authority(report)
    for r in report['decisions']['cat_full_puppet_contract']['candidate_decisions']:
        for k in ('pose_deformation','attachment_release','secondary_motion',
                  'arbitrary_order_override','nonbinary_alpha'):
            require(r['capabilities'][k]=='UNKNOWN','UNMEASURED_CAPABILITY_PROMOTION')
        require(r['status']!='SATISFIED','FULL_PUPPET_PROMOTION')
    # Evidence-side guards on logical equivalence and explicit costs.
    edit=report['decisions']['cat_edit_contract']
    require(set(edit['satisfying_alternatives'])=={
        'tied_controls_012','addressed_aggregate','split_shared_carrier',
        'per_face_with_uv_masks','hybrid_carrier_graphics'},'EQUIVALENT_ALTERNATIVES')
    require(report['cat']['new_physical_renders']==0 and t['new_physical_renders']==0,
            'RETROSPECTIVE_SCOPE')
    result={'schema':'RealSaS.IndependentDrawableDecisionCourt.v2','status':'PASS',
        'analysis_sha256':digest(report_path),'source_provenance':report['provenance'],
        'independent_cat_physical_pair_comparisons':4*64,
        'independent_tric_compositor_arm_comparisons':4*4*7,
        'independent_new_blender_execution':False,'same_drawable_pairs':[],
        'different_drawable_pairs':[],'training_authorized':False,
        'materialization_authorized':False,'graphics_owner_authorized':False}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('inputs','report','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();verify(a.inputs,a.report,a.output)
