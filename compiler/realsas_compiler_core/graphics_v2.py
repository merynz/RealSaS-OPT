"""Scope-local geometry order/compositing; no cycle deletion or invented owner.

This additive consumer is not an activated Graphics provider or native renderer.
It preserves measured per-ray constraints and reports where local interleaving is
required. A per-view total order is not fabricated from crossing surfaces.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from .hashing import content_sha256
from .graphics_evidence_v2 import checked_hash, require


@dataclass(frozen=True)
class CompiledRayGraphicsV2:
    canonical_mesh_binding_hash: str
    canonical_mesh_skin_binding_hash: str
    source_graphics_binding_hash: str
    rays: tuple[dict,...]
    scopes: tuple[dict,...]
    evidence_artifact_hashes: tuple[str,...]
    graph_hash: str
    schema_version: str='RealSaS.CompiledRayGraphics.v2'

    def to_dict(self):
        return asdict(self)


def _has_cycle(edges):
    graph={};degree={}
    for a,b in set(edges):
        graph.setdefault(a,set()).add(b)
        degree.setdefault(a,0);degree[b]=degree.get(b,0)+1
    ready=[n for n,d in degree.items() if d==0];count=0
    while ready:
        n=ready.pop();count+=1
        for other in graph.get(n,()):
            degree[other]-=1
            if degree[other]==0:ready.append(other)
    return count!=len(degree)


def _composite(ray):
    hits=ray.hits
    known_alpha=all(h.effective_alpha is not None for h in hits)
    known_rgb=all(h.linear_rgb is not None for h in hits)
    tie=any(b.depth-a.depth<=ray.tie_epsilon for a,b in zip(hits,hits[1:]))
    reason=('INVALID_MEASUREMENT' if not ray.measurement_valid else
            'INCOMPLETE_STACK' if not ray.complete else
            'UNSUPPORTED_BLEND' if ray.blend_law!='SOURCE_OVER' else
            'UNKNOWN_ALPHA' if not known_alpha else
            'SOURCE_ORDER_UNQUALIFIED' if ray.source_artistic_order_status!='NO_AUTHORED_ORDER__DEPTH_QUALIFIED' else
            'AMBIGUOUS_DEPTH_TIE' if tie else
            'UNKNOWN_COLOUR' if not known_rgb else 'MEASURED')
    alpha_valid=ray.measurement_valid and ray.complete and ray.blend_law=='SOURCE_OVER' and known_alpha
    alpha=1-math.prod(1-h.effective_alpha for h in hits) if alpha_valid else None
    hit_valid=reason in {'MEASURED','UNKNOWN_COLOUR'}
    rgb=[0.,0.,0.] if reason=='MEASURED' else None
    transmission=1.;weights=[];incoming=[]
    for h in hits:
        if hit_valid:
            incoming.append(transmission);weight=transmission*h.effective_alpha;weights.append(weight)
            transmission*=1-h.effective_alpha
            if rgb is not None:
                for i in range(3):rgb[i]+=weight*h.linear_rgb[i]
        else:
            incoming.append(None);weights.append(None)
    return {'reason':reason,'alpha':alpha,'alpha_valid':alpha_valid,'premultiplied_linear_rgb':rgb,
            'composite_valid':reason=='MEASURED','hit_weights_valid':hit_valid,
            'incoming_transmittance':incoming,'contribution':weights}


def compile_ray_graphics_v2(*, canonical_mesh_binding_hash, canonical_mesh_skin_binding_hash,
                           source_graphics_binding_hash, face_count, evidence,
                           address_witnesses, admitted_artifacts):
    for h in (canonical_mesh_binding_hash,canonical_mesh_skin_binding_hash,source_graphics_binding_hash):checked_hash(h)
    require(type(face_count) is int and face_count>0,'GRAPHICS_V2_FACE_COUNT_INVALID')
    rays=[];keys=set();scopes={};artifacts=set()
    for row in sorted(evidence,key=lambda r:(r.scope_hash,r.view_index,r.ray_key)):
        row.validate(mesh_hash=canonical_mesh_binding_hash,face_count=face_count,
                     address_witnesses=address_witnesses,admitted_artifacts=admitted_artifacts)
        key=(row.scope_hash,row.view_index,row.ray_key)
        require(key not in keys,'GRAPHICS_V2_DUPLICATE_RAY');keys.add(key)
        artifacts.add(row.evidence_artifact_hash)
        constraints=[]
        if row.measurement_valid and row.source_artistic_order_status=='NO_AUTHORED_ORDER__DEPTH_QUALIFIED':
            for i,a in enumerate(row.hits):
                for b in row.hits[i+1:]:
                    if b.depth-a.depth>row.tie_epsilon:
                        constraints.append((a.address.face_index,b.address.face_index))
        scope=scopes.setdefault((row.scope_hash,row.view_index),[])
        scope.extend(constraints)
        rays.append({'scope_hash':row.scope_hash,'view_index':row.view_index,'ray_key':row.ray_key,
                     'record_hash':row.record_hash(),'canonical_addresses':[asdict(h.address) for h in row.hits],
                     'order_constraints':constraints,'compositing':_composite(row)})
    scope_rows=tuple({'scope_hash':scope,'view_index':view,'order_constraints':sorted(set(edges)),
                      'order_status':'LOCAL_INTERLEAVING_REQUIRED' if _has_cycle(edges) else 'SCOPED_PARTIAL_ORDER',
                      'edges_removed':0,'unobserved_visibility':'UNKNOWN'} for (scope,view),edges in sorted(scopes.items()))
    body={'canonical_mesh_binding_hash':canonical_mesh_binding_hash,
          'canonical_mesh_skin_binding_hash':canonical_mesh_skin_binding_hash,
          'source_graphics_binding_hash':source_graphics_binding_hash,'rays':tuple(rays),'scopes':scope_rows,
          'evidence_artifact_hashes':tuple(sorted(artifacts)),'schema_version':'RealSaS.CompiledRayGraphics.v2'}
    return CompiledRayGraphicsV2(**body,graph_hash=content_sha256(body))
