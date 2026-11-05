/* BUSTED Studio IK Core / v1
 * Deterministic, topology-aware 2-segment IK math for *user-authoring*.
 * It does not mint Mechanics authority or mutate canonical rig data.
 */
const EPS=1e-7;
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
const asPoint=(v,label)=>{
  if(!Array.isArray(v)||v.length!==2||v.some(n=>!Number.isFinite(n)))throw new TypeError(label+' must be finite [x,y]');
  return [+v[0],+v[1]];
};
const angle=v=>Math.atan2(v[1],v[0]);
const delta=(b,a)=>{let d=b-a;while(d>Math.PI)d-=2*Math.PI;while(d< -Math.PI)d+=2*Math.PI;return d};
const distance=(a,b)=>Math.hypot(b[0]-a[0],b[1]-a[1]);
const move=(from,angle,d)=>[from[0]+Math.cos(angle)*d,from[1]+Math.sin(angle)*d];

/**
 * Solve reach for positioned canonical chain start→middle→end.
 * Returns relative rotation DELTAS in degrees for start/middle animator controls,
 * so existing keys and root translation remain intact.
 * bendDirection (+1|-1) selects a stable elbow branch when the chain is straight.
 * Constraints/range acceptance remains owned by Mechanics requalification.
 */
export function solveTwoBoneReach({start,middle,end,target,bendDirection=1,mix=1}){
  const a=asPoint(start,'start'),b=asPoint(middle,'middle'),c=asPoint(end,'end'),t=asPoint(target,'target');
  const l1=distance(a,b),l2=distance(b,c);
  if(l1<EPS||l2<EPS)throw new RangeError('IK chain contains a zero-length segment');
  const wanted=[t[0]-a[0],t[1]-a[1]];
  const requested=Math.hypot(...wanted);
  const unit=requested>EPS?[wanted[0]/requested,wanted[1]/requested]:
    [(b[0]-a[0])/l1,(b[1]-a[1])/l1];
  const minReach=Math.abs(l1-l2),maxReach=l1+l2;
  const d=clamp(requested,Math.max(EPS,minReach),maxReach);
  const clampedTarget=[a[0]+unit[0]*d,a[1]+unit[1]*d];
  const startAngle=angle([b[0]-a[0],b[1]-a[1]]);
  const endAngle=angle([c[0]-b[0],c[1]-b[1]]);
  const targetAngle=angle(unit);
  const rawCos=(l1*l1+d*d-l2*l2)/(2*l1*d);
  const theta=Math.acos(clamp(rawCos,-1,1));
  const branch=bendDirection<0?-1:1;
  const desiredStart=targetAngle+branch*theta;
  const desiredMid=move(a,desiredStart,l1);
  const desiredEndAngle=angle([clampedTarget[0]-desiredMid[0],clampedTarget[1]-desiredMid[1]]);
  const rotateStart=delta(desiredStart,startAngle);
  const rotateMid=delta(desiredEndAngle,endAngle)-rotateStart;
  const strength=clamp(Number(mix),0,1);
  if(!Number.isFinite(strength))throw new RangeError('IK mix must be finite');
  const resultStart=startAngle+rotateStart*strength;
  const resultMid=endAngle+(rotateStart+rotateMid)*strength;
  const reachedMid=move(a,resultStart,l1);
  const reachedEnd=move(reachedMid,resultMid,l2);
  return {
    start_delta_degrees:rotateStart*strength*180/Math.PI,
    middle_delta_degrees:rotateMid*strength*180/Math.PI,
    target:clampedTarget,
    reached_mid:reachedMid,
    reached_end:reachedEnd,
    requested_target:t,
    unreachable:requested>maxReach+EPS||requested<minReach-EPS,
    residual:distance(t,reachedEnd),
    segment_lengths:[l1,l2],
    authority:'STUDIO_AUTHORING_PREVIEW_ONLY',
  };
}

export function hierarchyParentMap(joints,edges){
  const valid=new Set((joints||[]).map(x=>String(x.control_id)));
  const parent=new Map();
  const invalid=new Set();
  for(const edge of edges||[]){
    if(edge.relation_type&&edge.relation_type!=='skeleton_hierarchy')continue;
    if(edge.entity_kind&&edge.entity_kind!=='DEFORM_JOINT')continue;
    const child=String(edge.child_control_id||''),root=String(edge.parent_control_id||'');
    if(!valid.has(child)||!valid.has(root)||child===root)continue;
    if(parent.has(child)&&parent.get(child)!==root)invalid.add(child);
    parent.set(child,root);
  }
  for(const child of invalid)parent.delete(child);
  return {parent,invalid};
}

/** Strict, authored 2-segment constraints with unambiguous one-to-one controllers. */
export function reachableTwoBoneConstraints({joints=[],edges=[],animatorControls=[],constraints=[],viewID}={}){
  const byJoint=new Map(joints.filter(r=>r.view_id===viewID).map(r=>[String(r.control_id),r]));
  const {parent,invalid}=hierarchyParentMap([...byJoint.values()],edges);
  const controllers=new Map(),collisions=new Set();
  for(const row of animatorControls){
    if(row.view_id!==viewID)continue;
    const targets=(row.target_joint_ids||[]).map(String);
    // Multi-target animator controls would rotate more than the intended chain.
    if(targets.length!==1||!byJoint.has(targets[0]))continue;
    if(controllers.has(targets[0]))collisions.add(targets[0]);
    else controllers.set(targets[0],row);
  }
  for(const key of collisions)controllers.delete(key);
  const usable=[];
  for(const constraint of constraints){
    if(constraint.enabled===false)continue;
    const startID=String(constraint.start_bone_id||''),endID=String(constraint.end_bone_id||'');
    const middleID=parent.get(endID);
    if(!middleID||parent.get(middleID)!==startID)continue;
    if([startID,middleID,endID].some(id=>!byJoint.has(id)||invalid.has(id)))continue;
    if(!controllers.has(startID)||!controllers.has(middleID))continue;
    if(controllers.get(startID).control_id===controllers.get(middleID).control_id)continue;
    const mix=Number(constraint.mix??1);
    if(!Number.isFinite(mix)||mix<=0||mix>1)continue;
    usable.push({
      constraint_id:String(constraint.constraint_id||''),
      start_id:startID,middle_id:middleID,end_id:endID,
      start_control_id:String(controllers.get(startID).control_id),
      middle_control_id:String(controllers.get(middleID).control_id),
      mix,bend_direction:Number(constraint.bend_direction||1)<0?-1:1,
    });
  }
  return usable;
}

/** FK anchor evaluation matches Studio's deformation preview semantics. */
export function worldJointAnchor(jointID,{joints,edges,jointPoses}){
  const byID=new Map((joints||[]).map(x=>[String(x.control_id),x]));
  const {parent,invalid}=hierarchyParentMap(joints,edges);
  let cursor=jointID;const chain=[],seen=new Set();
  while(cursor&&byID.has(cursor)){
    if(seen.has(cursor)||invalid.has(cursor))return null;
    seen.add(cursor);chain.unshift(cursor);cursor=parent.get(cursor);
  }
  if(!chain.length)return null;
  let point=asPoint(byID.get(jointID).anchor,'joint anchor');
  const prior=[];
  for(const id of chain){
    const joint=byID.get(id);
    let a=asPoint(joint.anchor,'joint anchor');
    for(const p of prior)a=localTransform(a,p.anchor,p.pose);
    const pose=jointPoses?.get(id)||{};
    point=localTransform(point,a,pose);
    prior.push({anchor:a,pose});
  }
  return point;
}
function localTransform(p,a,pose){
  const tr=pose?.translation_xy||[0,0],scale=pose?.scale_xy||[1,1];
  const r=(Number(pose?.rotation_degrees)||0)*Math.PI/180;
  const c=Math.cos(r),s=Math.sin(r),x=(p[0]-a[0])*(Number(scale[0])||1),y=(p[1]-a[1])*(Number(scale[1])||1);
  return [a[0]+c*x-s*y+(Number(tr[0])||0),a[1]+s*x+c*y+(Number(tr[1])||0)];
}
