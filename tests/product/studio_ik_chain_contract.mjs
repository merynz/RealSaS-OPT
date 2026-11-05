import assert from 'node:assert/strict';
import {solveTwoBoneReach,reachableTwoBoneConstraints,worldJointAnchor} from '../../product/living_compile/static/js/studio_ik_chain_core.js';
const near=(a,b,eps=1e-5)=>assert.ok(Math.abs(a-b)<eps,`expected ${a} ~= ${b}`);
const point=(a,b)=>{near(a[0],b[0]);near(a[1],b[1]);};

// Straight chain reaches a diagonal target by rotating BOTH bones.
let result=solveTwoBoneReach({start:[0,0],middle:[1,0],end:[2,0],target:[1,1]});
point(result.reached_end,[1,1]);
near(result.start_delta_degrees,90);
near(result.middle_delta_degrees,-90);
assert.equal(result.unreachable,false);
assert.equal(result.authority,'STUDIO_AUTHORING_PREVIEW_ONLY');

result=solveTwoBoneReach({start:[0,0],middle:[1,0],end:[2,0],target:[4,0]});
point(result.reached_end,[2,0]);
assert.equal(result.unreachable,true);
near(result.residual,2);

result=solveTwoBoneReach({start:[0,0],middle:[1,0],end:[2,0],target:[1,1],bendDirection:-1});
point(result.reached_end,[1,1]);
near(result.start_delta_degrees,0);
near(result.middle_delta_degrees,90);
assert.ok(result.start_delta_degrees!==90);

// Mix is a partial change, not an invented completed reach.
result=solveTwoBoneReach({start:[0,0],middle:[1,0],end:[2,0],target:[1,1],mix:.5});
assert.ok(result.residual>.1&&result.residual<2);
assert.equal(result.unreachable,false);
assert.throws(()=>solveTwoBoneReach({start:[0,0],middle:[0,0],end:[1,0],target:[1,1]}),/zero-length/);
assert.throws(()=>solveTwoBoneReach({start:[0,0],middle:[1,0],end:[2,0],target:[NaN,0]}),/finite/);

const joints=[
 {control_id:'shoulder',view_id:'V0',anchor:[0,0]},
 {control_id:'elbow',view_id:'V0',anchor:[1,0]},
 {control_id:'wrist',view_id:'V0',anchor:[2,0]},
];
const edges=[
 {parent_control_id:'shoulder',child_control_id:'elbow',relation_type:'skeleton_hierarchy',entity_kind:'DEFORM_JOINT'},
 {parent_control_id:'elbow',child_control_id:'wrist',relation_type:'skeleton_hierarchy',entity_kind:'DEFORM_JOINT'}
];
const animatorControls=[
 {control_id:'anim_shoulder',view_id:'V0',target_joint_ids:['shoulder']},
 {control_id:'anim_elbow',view_id:'V0',target_joint_ids:['elbow']},
];
const constraints=[{
 constraint_id:'IK:USER:1',start_bone_id:'shoulder',end_bone_id:'wrist',target_bone_id:'wrist',
 mix:1,bend_direction:1,enabled:true
}];
const good=reachableTwoBoneConstraints({joints,edges,animatorControls,constraints,viewID:'V0'});
assert.equal(good.length,1);
assert.equal(good[0].middle_id,'elbow');
assert.equal(good[0].start_control_id,'anim_shoulder');

// Fail closed for ambiguous animator coverage and invalid hierarchy.
assert.equal(reachableTwoBoneConstraints({joints,edges,animatorControls:[
 ...animatorControls,{control_id:'duplicate_shoulder',view_id:'V0',target_joint_ids:['shoulder']}
],constraints,viewID:'V0'}).length,0);
assert.equal(reachableTwoBoneConstraints({joints,edges,animatorControls:[
 {control_id:'multi',view_id:'V0',target_joint_ids:['shoulder','elbow']}
],constraints,viewID:'V0'}).length,0);
assert.equal(reachableTwoBoneConstraints({joints,edges,animatorControls,constraints,viewID:'V1'}).length,0);
assert.equal(reachableTwoBoneConstraints({joints,edges,animatorControls,constraints:[{
 ...constraints[0],target_bone_id:'elbow'
}],viewID:'V0'}).length,0);
const invalidEdges=[...edges,{parent_control_id:'wrist',child_control_id:'elbow',relation_type:'skeleton_hierarchy'}];
assert.equal(reachableTwoBoneConstraints({joints,edges:invalidEdges,animatorControls,constraints,viewID:'V0'}).length,0);

// FK world anchors agree with the authored transform composition.
point(worldJointAnchor('wrist',{joints,edges,jointPoses:new Map()}),[2,0]);
point(worldJointAnchor('wrist',{joints,edges,jointPoses:new Map([
 ['shoulder',{rotation_degrees:90,translation_xy:[0,0],scale_xy:[1,1]}],
])}),[0,2]);

console.log('STUDIO_IK_CHAIN_CONTRACT_PASS');
