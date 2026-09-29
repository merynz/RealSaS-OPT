from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict


def pair(a,b):
    a,b=str(a),str(b)
    return (a,b) if a<b else (b,a)


class MutexDSU:
    def __init__(self, ids):
        self.parent={x:x for x in ids}
        self.size={x:1 for x in ids}
        self.mutex={x:set() for x in ids}

    def find(self,x):
        p=self.parent[x]
        if p!=x:
            self.parent[x]=self.find(p)
        return self.parent[x]

    def _canon_mutex(self,r):
        r=self.find(r)
        out=set()
        for x in tuple(self.mutex[r]):
            rx=self.find(x)
            if rx!=r:
                out.add(rx)
        self.mutex[r]=out
        return out

    def add_mutex(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:
            return False
        self.mutex[ra].add(rb);self.mutex[rb].add(ra)
        return True

    def are_mutex(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return False
        return rb in self._canon_mutex(ra) or ra in self._canon_mutex(rb)

    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return True
        if self.are_mutex(ra,rb):return False
        if self.size[ra]<self.size[rb]:ra,rb=rb,ra
        ma=self._canon_mutex(ra);mb=self._canon_mutex(rb)
        self.parent[rb]=ra
        self.size[ra]+=self.size[rb]
        merged={self.find(x) for x in (ma|mb)}
        merged.discard(ra);merged.discard(rb)
        self.mutex[ra]=set(merged);self.mutex[rb]=set()
        for x in tuple(merged):
            rx=self.find(x)
            mx=self._canon_mutex(rx)
            mx.discard(rb);mx.add(ra)
            self.mutex[rx]=mx
        return True


def component_labels_after_deleting_seeds(ids, edges, seed_set):
    nbr={x:[] for x in ids}
    for a,b in edges:
        if (a,b) in seed_set:continue
        nbr[a].append(b);nbr[b].append(a)
    lab={}
    comps=[]
    for s in ids:
        if s in lab:continue
        q=[s];lab[s]=len(comps);members=[]
        while q:
            u=q.pop();members.append(u)
            for v in nbr[u]:
                if v not in lab:
                    lab[v]=lab[s];q.append(v)
        comps.append(tuple(sorted(members)))
    return lab,comps


def shortest_alt_path(a,b,adj,seed_set,max_depth=12):
    q=deque([(a,0)])
    prev={a:None}
    while q:
        u,d=q.popleft()
        if d>=max_depth:continue
        for v in adj[u]:
            if pair(u,v) in seed_set:continue
            if v in prev:continue
            prev[v]=u
            if v==b:
                path=[b];x=b
                while prev[x] is not None:
                    x=prev[x];path.append(x)
                return tuple(reversed(path))
            q.append((v,d+1))
    return ()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--directive-json",type=Path,required=True)
    ap.add_argument("--arm",required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    surface=rigging_surface_from_dict(json.loads(a.surface_json.read_text()))
    directive=json.loads(a.directive_json.read_text())
    ids=tuple(sorted(str(n.surface_id) for n in surface.surface_nodes))
    known=set(ids)
    edges=sorted({
        pair(r.a_surface_id,r.b_surface_id)
        for r in surface.local_relations
    })
    edge_set=set(edges)
    seeds=tuple(sorted({
        pair(x["a_surface_id"],x["b_surface_id"])
        for x in directive["proposed_boundary_overrides"]
    }))
    seed_set=set(seeds)
    if len(seeds)!=int(directive["candidate_separate_pair_count"]):
        raise RuntimeError("SEED_COUNT_DRIFT")
    if any(x not in edge_set for x in seeds):
        raise RuntimeError("SEED_NOT_LOCAL_RELATION")
    if any(u not in known or v not in known for u,v in seeds):
        raise RuntimeError("SEED_UNKNOWN_NODE")

    naive_lab,naive_comps=component_labels_after_deleting_seeds(ids,edges,seed_set)
    noncut=[x for x in seeds if naive_lab[x[0]]==naive_lab[x[1]]]

    adj={x:[] for x in ids}
    for u,v in edges:
        adj[u].append(v);adj[v].append(u)
    alt=[]
    for u,v in noncut[:100]:
        p=shortest_alt_path(u,v,adj,seed_set,max_depth=12)
        alt.append({"pair":[u,v],"alternative_path_hops":None if not p else len(p)-1,"path":list(p)})

    # Preload all must-separate constraints before any continuity merge.
    dsu=MutexDSU(ids)
    added=sum(bool(dsu.add_mutex(u,v)) for u,v in seeds)

    # Deterministic maximum-continuity policy from current structural evidence:
    # safe/high-score relations first, then unknown/low score; pair is final tie break.
    relation_by_pair={}
    for r in surface.local_relations:
        p=pair(r.a_surface_id,r.b_surface_id)
        relation_by_pair.setdefault(p,[]).append(r)
    attractive=[]
    for p in edges:
        if p in seed_set:continue
        rels=relation_by_pair[p]
        unknown=any(
            ("UNKNOWN" in str(r.relation_kind).upper())
            or ("AMBIGUOUS" in str(r.relation_kind).upper())
            or ("OCCLUDED" in str(r.relation_kind).upper())
            or ("UNOBSERVED" in str(r.relation_kind).upper())
            or ("UNSUPPORTED" in str(r.relation_kind).upper())
            or bool(dict(getattr(r,"metadata",{}) or {}).get("crosses_unknown",False))
            or bool(dict(getattr(r,"metadata",{}) or {}).get("unknown_bridge",False))
            or float(r.score)<=0.0
            for r in rels
        )
        score=min(max(float(r.score),0.0) for r in rels)
        attractive.append((1 if unknown else 0,-score,p[0],p[1]))
    attractive.sort()

    blocked=[]
    union_added=0
    for unknown_flag,neg_score,u,v in attractive:
        before=dsu.find(u)==dsu.find(v)
        ok=dsu.union(u,v)
        if not before and ok:
            union_added+=1
        elif not ok:
            blocked.append(pair(u,v))

    roots={x:dsu.find(x) for x in ids}
    remap={}
    labels={}
    for x in ids:
        r=roots[x]
        if r not in remap:remap[r]=len(remap)
        labels[x]=remap[r]

    violated=[x for x in seeds if labels[x[0]]==labels[x[1]]]
    crossing=[p for p in edges if labels[p[0]]!=labels[p[1]]]
    closure_added=sorted(set(crossing)-seed_set)
    components={}
    for x,l in labels.items():components.setdefault(l,[]).append(x)
    sizes=sorted((len(v) for v in components.values()),reverse=True)

    report={
        "schema":"RealSaS.KnightStage35SeedCutClosureDiagnostic.v1",
        "status":"PASS_DIAGNOSTIC_EXECUTED",
        "arm":a.arm,
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "directive_hash":directive["directive_hash"],
        "graph":{
            "node_count":len(ids),
            "local_relation_edge_count":len(edges),
        },
        "seed_separation":{
            "seed_count":len(seeds),
            "naive_delete_seed_component_count":len(naive_comps),
            "naive_noncut_seed_count":len(noncut),
            "naive_noncut_seed_fraction":len(noncut)/max(1,len(seeds)),
            "noncut_seed_examples":[list(x) for x in noncut[:100]],
            "alternative_path_examples":alt,
        },
        "preloaded_mutex_closure":{
            "preloaded_mutex_count":added,
            "attractive_union_count":union_added,
            "attractive_union_blocked_by_mutex":len(blocked),
            "component_count":len(components),
            "largest_component_sizes":sizes[:50],
            "seed_constraint_violation_count":len(violated),
            "cross_component_relation_count":len(crossing),
            "closure_added_separate_pair_count":len(closure_added),
            "final_separate_pair_count":len(crossing),
            "closure_added_pair_examples":[list(x) for x in closure_added[:100]],
            "all_seed_constraints_satisfied":len(violated)==0,
        },
        "interpretation":{
            "current_directive_is_partition_closed":len(noncut)==0,
            "preloaded_mutex_can_produce_partition_valid_boundary_closure":len(violated)==0,
            "stage17_requires_closed_cross_component_boundary_set":True,
        },
        "claim_boundary":[
            "Diagnostic only; no MechanicalPartitionIR is created or promoted.",
            "All Stage35 seed cannot-link constraints are frozen before continuity unions.",
            "No teacher topology, teacher weights, manual labels, or product mutation are used.",
            "Attractive ordering uses only existing RiggingSurface local-relation safety/score authority and deterministic pair tie-breaking.",
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_SEED_CUT_CLOSURE_DIAGNOSTIC="+json.dumps({
        "arm":a.arm,
        "seed_count":len(seeds),
        "naive_noncut_seed_count":len(noncut),
        "preloaded_component_count":len(components),
        "blocked_attractive":len(blocked),
        "closure_added_separate_pair_count":len(closure_added),
        "final_separate_pair_count":len(crossing),
        "violated":len(violated),
    },sort_keys=True))


if __name__=="__main__":
    main()
