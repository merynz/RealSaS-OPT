from compiler.realsas_compiler_core.presentation_structure_v1 import _presentation_face_groups

class V:
    def __init__(self,vid,cid):
        self.canonical_mesh_vertex_id=vid
        self.component_id=cid

class M:
    vertices=(V("a","c"),V("b","c"),V("c","c"),V("d","c"),V("e","c"),V("f","c"))
    faces=(("a","b","c"),("d","e","f"))
    mesh_lineage_hash="m"

def test_presentation_groups_split_disconnected_face_islands_without_semantics():
    groups=_presentation_face_groups(M(),"c")
    assert groups==((0,),(1,))
