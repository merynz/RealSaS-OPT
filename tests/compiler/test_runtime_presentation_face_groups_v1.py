from compiler.realsas_compiler_core.runtime_projection_v1 import _project_assets

class V:
    def __init__(self,vid,cid,p):
        self.canonical_mesh_vertex_id=vid; self.component_id=cid; self.P=p

class Mesh:
    mesh_lineage_hash="mesh"
    vertices=(
        V("a","c",(0,0,0)),V("b","c",(1,0,0)),V("c","c",(0,1,0)),
        V("d","c",(2,0,0)),V("e","c",(3,0,0)),V("f","c",(2,1,0)),
    )
    faces=(("a","b","c"),("d","e","f"))

class A:
    def __init__(self,aid,sid,faces):
        self.attachment_id=aid; self.slot_id=sid
        self.mechanical_component_ids=("c",)
        self.mechanical_class="DEFORMABLE"
        self.metadata={"mesh_face_indices":faces,"presentation_group_hash":aid}

class P:
    attachments=(A("a0","s0",(0,)),A("a1","s1",(1,)))

def test_runtime_assets_respect_presentation_face_groups_within_one_mechanical_component():
    assets,by=_project_assets(Mesh(),P())
    assert len(assets)==2
    assert sorted(a.source_face_indices for a in assets)==[(0,),(1,)]
    assert set().union(*(set(x) for x in by.values()))=={0,1}
