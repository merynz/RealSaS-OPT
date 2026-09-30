from __future__ import annotations

"""Fail-closed application seam for Stage35 -> Stage17 mechanical repartition.

The Stage35 diagnostic directive is not execution authority.  A child attempt may
apply it only when a separately sealed trustworthy-skin authorization binds the
exact parent surface/partition/skin evidence that produced the directive.
"""

from dataclasses import replace

from .hashing import content_sha256
from .mechanical_partition_v1 import build_structural_partition
from .product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    mechanical_partition_lineage_hash,
)
from .types import QualificationError

AUTH_SCHEMA="RealSaS.TrustworthySkinRepartitionAuthorization.v1"
AUTH_STATUS="PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION"
DIRECTIVE_SCHEMA="RealSaS.MechanicalRepartitionDirective.v2"


def repartition_authorization_hash_v1(payload:dict)->str:
    value=dict(payload)
    value.pop("authorization_hash",None)
    return content_sha256(value)


def validate_trustworthy_skin_repartition_authorization_v1(
    authorization:dict,
    *,
    directive:dict,
    surface,
    parent_partition,
)->None:
    if str(directive.get("schema") or "")!=DIRECTIVE_SCHEMA:
        raise QualificationError("REPARTITION_DIRECTIVE_SCHEMA_DRIFT")
    if str(directive.get("status") or "")!="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY":
        raise QualificationError("REPARTITION_DIRECTIVE_NOT_APPLICABLE")
    if directive.get("face_deletion_count")!=0 or directive.get("weight_mutation") is not False:
        raise QualificationError("REPARTITION_DIRECTIVE_MUTATION_SCOPE_INVALID")
    if directive.get("requires_trustworthy_skin_reliability_authority") is not True:
        raise QualificationError("REPARTITION_DIRECTIVE_RELIABILITY_GATE_MISSING")
    if str(directive.get("source_surface_lineage_hash") or "")!=str(surface.geometry_lineage_hash):
        raise QualificationError("REPARTITION_DIRECTIVE_SURFACE_LINEAGE_DRIFT")
    if str(directive.get("source_partition_lineage_hash") or "")!=str(parent_partition.partition_lineage_hash):
        raise QualificationError("REPARTITION_DIRECTIVE_PARENT_PARTITION_DRIFT")

    if str(authorization.get("schema") or "")!=AUTH_SCHEMA:
        raise QualificationError("REPARTITION_AUTHORIZATION_SCHEMA_DRIFT")
    if str(authorization.get("status") or "")!=AUTH_STATUS:
        raise QualificationError("REPARTITION_AUTHORIZATION_NOT_PASS")
    if str(authorization.get("directive_hash") or "")!=str(directive.get("directive_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_DIRECTIVE_DRIFT")
    if str(authorization.get("source_surface_lineage_hash") or "")!=str(surface.geometry_lineage_hash):
        raise QualificationError("REPARTITION_AUTHORIZATION_SURFACE_DRIFT")
    if str(authorization.get("source_partition_lineage_hash") or "")!=str(parent_partition.partition_lineage_hash):
        raise QualificationError("REPARTITION_AUTHORIZATION_PARTITION_DRIFT")
    if str(authorization.get("source_skin_lineage_hash") or "")!=str(directive.get("source_skin_lineage_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_SKIN_LINEAGE_DRIFT")
    if str(authorization.get("source_skeleton_lineage_hash") or "")!=str(directive.get("source_skeleton_lineage_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_SKELETON_LINEAGE_DRIFT")
    if authorization.get("teacher_inputs_used_by_predictor") is not False:
        raise QualificationError("REPARTITION_AUTHORIZATION_TEACHER_PREDICTOR_INPUT_FORBIDDEN")
    if authorization.get("weight_reliability_closure_passed") is not True:
        raise QualificationError("REPARTITION_AUTHORIZATION_WEIGHT_CLOSURE_NOT_PASS")
    evidence_hash=str(authorization.get("weight_reliability_evidence_hash") or "")
    if len(evidence_hash)!=64:
        raise QualificationError("REPARTITION_AUTHORIZATION_WEIGHT_EVIDENCE_HASH_INVALID")
    if str(authorization.get("authorization_hash") or "")!=repartition_authorization_hash_v1(authorization):
        raise QualificationError("REPARTITION_AUTHORIZATION_HASH_DRIFT")



def _pair(a:str,b:str)->tuple[str,str]:
    a,b=str(a),str(b)
    if not a or not b or a==b:
        raise QualificationError("REPARTITION_CUT_PAIR_INVALID")
    return (a,b) if a<b else (b,a)


class _MutexDSU:
    """Deterministic union-find with component-level cannot-link constraints."""

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
        rr=self.find(r)
        out={self.find(x) for x in self.mutex.get(rr,set())}
        out.discard(rr)
        self.mutex[rr]=set(out)
        return self.mutex[rr]

    def add_mutex(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:
            return False
        self.mutex[ra].add(rb)
        self.mutex[rb].add(ra)
        return True

    def are_mutex(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:
            return False
        return rb in self._canon_mutex(ra) or ra in self._canon_mutex(rb)

    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:
            return True
        if self.are_mutex(ra,rb):
            return False
        if self.size[ra] < self.size[rb] or (self.size[ra]==self.size[rb] and rb<ra):
            ra,rb=rb,ra
        ma=set(self._canon_mutex(ra))
        mb=set(self._canon_mutex(rb))
        self.parent[rb]=ra
        self.size[ra]+=self.size[rb]
        merged={self.find(x) for x in (ma|mb)}
        merged.discard(ra)
        merged.discard(rb)
        self.mutex[ra]=set(merged)
        self.mutex[rb]=set()
        for x in tuple(merged):
            rx=self.find(x)
            mx=set(self._canon_mutex(rx))
            mx.discard(rb)
            mx.add(ra)
            self.mutex[rx]=mx
        return True


def _partition_closed_dynamic_overrides(
    *,
    surface,
    parent_partition,
    directive:dict,
    authorization:dict,
):
    """Close Stage35 direct cannot-link seeds into a valid Stage17 graph cut.

    Direct Stage35 pairs are *seeds*, not necessarily graph bridges.  Stage17 owns
    the global mechanical partition, so all direct seeds and inherited SEPARATE
    constraints are preloaded as cannot-links before any continuity union.  The
    remaining local-relation graph is then maximally merged in deterministic
    structural-authority order.  Every local relation crossing the resulting
    components becomes explicit SEPARATE, yielding a partition-closed cut set.
    """
    ids=tuple(sorted(str(n.surface_id) for n in surface.surface_nodes))
    known=set(ids)
    parent_by_pair={
        _pair(row.a_surface_id,row.b_surface_id):row
        for row in parent_partition.boundary_constraints
    }

    raw_proposed=tuple(directive.get("proposed_boundary_overrides") or ())
    if not raw_proposed:
        raise QualificationError("REPARTITION_DIRECTIVE_HAS_NO_BOUNDARY_OVERRIDES")
    raw_by_pair={}
    for raw in raw_proposed:
        if str(raw.get("decision") or "")!="SEPARATE":
            raise QualificationError("REPARTITION_DIRECTIVE_NONSEPARATE_OVERRIDE")
        p=_pair(raw.get("a_surface_id"),raw.get("b_surface_id"))
        if p not in parent_by_pair:
            raise QualificationError("REPARTITION_DIRECTIVE_REQUIRES_LOCAL_RELATION")
        refs=tuple(map(str,raw.get("evidence_refs") or ()))
        if not refs:
            raise QualificationError("REPARTITION_DIRECTIVE_EVIDENCE_MISSING")
        if p in raw_by_pair:
            raise QualificationError("REPARTITION_DIRECTIVE_DUPLICATE_PAIR")
        raw_by_pair[p]=raw

    seed_pairs=set(raw_by_pair)
    inherited_separate={
        p for p,row in parent_by_pair.items()
        if str(row.decision)=="SEPARATE"
    }
    must_separate=set(seed_pairs)|set(inherited_separate)

    dsu=_MutexDSU(ids)
    for a,b in sorted(must_separate):
        if a not in known or b not in known:
            raise QualificationError("REPARTITION_CUT_SEED_UNKNOWN_NODE")
        if not dsu.add_mutex(a,b):
            raise QualificationError("REPARTITION_CUT_SEED_CONTRADICTION")

    # Structural authority ordering only: preserve before unknown, higher
    # confidence first, canonical pair tie-break.  No skin similarity is used
    # to decide attractive unions.
    attractive=[]
    for p,row in parent_by_pair.items():
        if p in must_separate:
            continue
        decision=str(row.decision)
        unknown_flag=1 if decision=="UNKNOWN" else 0
        attractive.append((
            unknown_flag,
            -float(row.confidence),
            p[0],
            p[1],
        ))
    attractive.sort()

    blocked=[]
    union_count=0
    for _,_,a,b in attractive:
        already=dsu.find(a)==dsu.find(b)
        ok=dsu.union(a,b)
        if not already and ok:
            union_count+=1
        elif not ok:
            blocked.append((a,b))

    labels={sid:dsu.find(sid) for sid in ids}
    violated=[p for p in sorted(must_separate) if labels[p[0]]==labels[p[1]]]
    if violated:
        raise QualificationError("REPARTITION_CUT_SEED_VIOLATION")

    crossing={
        p for p in parent_by_pair
        if labels[p[0]]!=labels[p[1]]
    }
    if not must_separate.issubset(crossing):
        raise QualificationError("REPARTITION_CUT_CLOSURE_LOST_SEED")

    directive_hash=str(directive["directive_hash"])
    authorization_hash=str(authorization["authorization_hash"])
    out={}
    for p in sorted(crossing):
        parent_row=parent_by_pair[p]
        raw=raw_by_pair.get(p)
        if raw is not None:
            refs=tuple(sorted(set(
                map(str,raw.get("evidence_refs") or ())
            ) | set(map(str,parent_row.evidence_refs))))
            confidence=max(float(raw.get("confidence",0.0)),float(parent_row.confidence))
            cid=str(raw.get("constraint_id") or "")
            md={
                **dict(parent_row.metadata or {}),
                **dict(raw.get("metadata") or {}),
                "evidence_class":"STAGE35_DYNAMIC_SKIN_TOPOLOGY_MECHANICAL__PARTITION_CLOSED",
                "direct_stage35_seed":True,
            }
        elif p in inherited_separate:
            refs=tuple(map(str,parent_row.evidence_refs))
            confidence=float(parent_row.confidence)
            cid=str(parent_row.constraint_id)
            md={
                **dict(parent_row.metadata or {}),
                "inherited_parent_separate":True,
            }
        else:
            refs=tuple(sorted(set(map(str,parent_row.evidence_refs)) | {
                f"{directive_hash}:CUT_CLOSURE",
            }))
            confidence=float(parent_row.confidence)
            cid="DYNCLOSE:"+content_sha256({
                "directive_hash":directive_hash,
                "pair":p,
                "parent_constraint_id":str(parent_row.constraint_id),
            })[:20]
            md={
                **dict(parent_row.metadata or {}),
                "evidence_class":"STAGE35_DYNAMIC_SKIN_TOPOLOGY_MECHANICAL__CUT_CLOSURE",
                "direct_stage35_seed":False,
                "closure_added":True,
                "closure_policy":"STRUCTURAL_AUTHORITY_MAX_CONTINUITY",
            }
        md={
            **md,
            "repair_directive_hash":directive_hash,
            "repartition_authorization_hash":authorization_hash,
            "dynamic_parent_skin_lineage_hash":str(directive["source_skin_lineage_hash"]),
        }
        out[p]=ComponentBoundaryConstraintIR(
            constraint_id=cid,
            a_surface_id=p[0],
            b_surface_id=p[1],
            decision="SEPARATE",
            evidence_refs=refs,
            confidence=confidence,
            metadata=md,
        )

    components={}
    for sid,r in labels.items():
        components.setdefault(r,[]).append(sid)
    audit={
        "closure_policy":"STRUCTURAL_AUTHORITY_MAX_CONTINUITY",
        "direct_seed_count":len(seed_pairs),
        "inherited_separate_count":len(inherited_separate),
        "must_separate_count":len(must_separate),
        "attractive_union_count":int(union_count),
        "attractive_union_blocked_by_mutex":len(blocked),
        "component_count":len(components),
        "closure_added_separate_count":len(crossing-must_separate),
        "final_separate_count":len(crossing),
        "seed_constraint_violation_count":0,
    }
    return tuple(out[p] for p in sorted(out)),audit

def build_repartitioned_partition_v2(
    *,
    surface,
    parent_partition,
    directive:dict,
    authorization:dict,
):
    """Build a distinct Stage17 child partition from an authorized Stage35 directive.

    Stage35 direct SEPARATE pairs are treated as cannot-link seeds.  Stage17,
    which owns global partition semantics, closes those seeds into a complete
    induced cut set before materializing MechanicalPartitionIR.
    """
    validate_trustworthy_skin_repartition_authorization_v1(
        authorization,
        directive=directive,
        surface=surface,
        parent_partition=parent_partition,
    )
    overrides,closure_audit=_partition_closed_dynamic_overrides(
        surface=surface,
        parent_partition=parent_partition,
        directive=directive,
        authorization=authorization,
    )

    child=build_structural_partition(
        surface,
        boundary_overrides=overrides,
    )
    if child.partition_lineage_hash==parent_partition.partition_lineage_hash:
        raise QualificationError("REPARTITION_CHILD_NOT_DISTINCT")
    parent_separate=sum(x.decision=="SEPARATE" for x in parent_partition.boundary_constraints)
    child_separate=sum(x.decision=="SEPARATE" for x in child.boundary_constraints)
    if child_separate<=parent_separate:
        raise QualificationError("REPARTITION_CHILD_DID_NOT_ADD_SEPARATION")

    child=replace(
        child,
        metadata={
            **dict(child.metadata or {}),
            "repair_child_attempt":True,
            "parent_partition_lineage_hash":str(parent_partition.partition_lineage_hash),
            "repair_directive_hash":str(directive["directive_hash"]),
            "repartition_authorization_hash":str(authorization["authorization_hash"]),
            "source_skin_lineage_hash":str(directive["source_skin_lineage_hash"]),
            "automatic_manual_authoring_used":False,
            "partition_cut_closure":closure_audit,
        },
        partition_lineage_hash="",
    )
    child=replace(child,partition_lineage_hash=mechanical_partition_lineage_hash(child))
    return child
