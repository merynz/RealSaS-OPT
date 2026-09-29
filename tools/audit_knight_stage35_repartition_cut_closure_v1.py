from __future__ import annotations

import argparse, json
from collections import defaultdict, deque
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import ComponentBoundaryConstraintIR


def pair(a,b):
    a,b=str(a),str(b)
    return (a,b) if a<b else (b,a)


def components(surface, removed):
    ids=tuple(str(n.surface_id) for n in surface.surface_nodes)
    adj={x:set() for x in ids}
    for r in surface.local_relations:
        p=pair(r.a_surface_id,r.b_surface_id)
        if p in removed:
            continue
        adj[p[0]].add(p[1]); adj[p[1]].add(p[0])
    comp={}
    groups=[]
    for sid in sorted(ids):
        if sid in comp: continue
        ci=len(groups); q=[sid]; comp[sid]=ci; members=[]
        while q:
            x=q.pop(); members.append(x)
            for y in sorted(adj[x]):
                if y not in comp:
                    comp[y]=ci; q.append(y)
        groups.append(tuple(sorted(members)))
    return comp,tuple(groups)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--directive-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    surface=rigging_surface_from_dict(json.loads(a.surface_json.read_text()))
    directive=json.loads(a.directive_json.read_text())
    proposals=tuple(directive["proposed_boundary_overrides"])
    proposed_pairs={pair(x["a_surface_id"],x["b_surface_id"]) for x in proposals}

    comp,groups=components(surface,proposed_pairs)
    true_cut={p for p in proposed_pairs if comp[p[0]]!=comp[p[1]]}
    redundant=proposed_pairs-true_cut

    all_pairs={pair(r.a_surface_id,r.b_surface_id) for r in surface.local_relations}
    crossing_all={p for p in all_pairs if comp[p[0]]!=comp[p[1]]}
    missing_crossing=crossing_all-proposed_pairs
    extra_true=true_cut-crossing_all

    by_pair={pair(x["a_surface_id"],x["b_surface_id"]):x for x in proposals}
    canonical=[]
    for p in sorted(true_cut):
        row=by_pair[p]
        canonical.append(ComponentBoundaryConstraintIR(
            constraint_id=str(row["constraint_id"]),
            a_surface_id=p[0], b_surface_id=p[1],
            decision="SEPARATE",
            evidence_refs=tuple(row["evidence_refs"]),
            confidence=float(row["confidence"]),
            metadata=dict(row.get("metadata") or {}),
        ))

    child=None
    build_error=None
    try:
        child=build_structural_partition(surface,boundary_overrides=tuple(canonical))
    except Exception as exc:
        build_error=f"{type(exc).__name__}:{exc}"

    report={
      "schema":"RealSaS.KnightStage35RepartitionCutClosureAudit.v1",
      "status":"MEASURED",
      "directive_hash":directive["directive_hash"],
      "proposed_pair_count":len(proposed_pairs),
      "removed_graph_component_count":len(groups),
      "component_sizes_desc":sorted((len(g) for g in groups),reverse=True)[:50],
      "true_cut_pair_count":len(true_cut),
      "redundant_noncut_pair_count":len(redundant),
      "crossing_graph_edge_count":len(crossing_all),
      "missing_crossing_edge_count":len(missing_crossing),
      "extra_true_cut_not_graph_crossing_count":len(extra_true),
      "canonical_cut_partition_build_passed":child is not None,
      "canonical_child_component_count":None if child is None else len(child.components),
      "canonical_child_separate_count":None if child is None else sum(x.decision=="SEPARATE" for x in child.boundary_constraints),
      "build_error":build_error,
      "redundant_pairs_head":[list(x) for x in sorted(redundant)[:100]],
      "finding":{
        "proposal_contains_redundant_noncut_edges":bool(redundant),
        "proposed_set_contains_complete_induced_cut":len(missing_crossing)==0,
        "pruning_redundant_edges_is_sufficient_for_valid_partition":child is not None and len(missing_crossing)==0,
      }
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("REPARTITION_CUT_CLOSURE="+json.dumps({k:report[k] for k in (
      "proposed_pair_count","removed_graph_component_count","true_cut_pair_count",
      "redundant_noncut_pair_count","crossing_graph_edge_count","missing_crossing_edge_count",
      "canonical_cut_partition_build_passed","canonical_child_component_count"
    )},sort_keys=True))

if __name__=="__main__":main()
