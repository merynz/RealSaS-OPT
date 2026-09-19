from pathlib import Path
import zipfile
from compiler.realsas_compiler_core.authoring_bundle_v1 import materialize_editable_puppet_bundle_v1, editable_puppet_bundle_hash

class T:
    def __init__(self,i,p,sha):
        self.view_index=i; self.transport_png_path=str(p); self.transport_png_sha256=sha
class P:
    product_state_binding_hash="state"
    projection_hash="projection"
    textures=()

def test_editable_bundle_is_self_contained(tmp_path):
    import hashlib
    tex=[]
    for i in range(8):
        p=tmp_path/f"v{i}.png"; p.write_bytes(bytes([i,1,2,3]))
        tex.append(T(i,p,hashlib.sha256(p.read_bytes()).hexdigest()))
    projection=P(); projection.textures=tuple(tex)
    seal=materialize_editable_puppet_bundle_v1(
        out_path=tmp_path/"p.rsedit",
        state_payload={"product_state_hash":"state","skeleton_lineage_hash":"sk","mesh_lineage_hash":"m","mesh_skin_lineage_hash":"ms"},
        skeleton_payload={"skeleton_lineage_hash":"sk","joints":[]},
        mesh_payload={"mesh_lineage_hash":"m"},
        mesh_skin_payload={"mesh_skin_lineage_hash":"ms"},
        presentation_payload={"presentation_lineage_hash":"pr","product_state_binding_hash":"state"},
        appearance_payload={"appearance_set_hash":"ap","mesh_binding_hash":"m"},
        motion_payload={"motion_lineage_hash":"mo","product_state_binding_hash":"state"},
        projection=projection,
    )
    assert seal.authoring_bundle_hash==editable_puppet_bundle_hash(seal)
    assert seal.qualification_report["weights_editable"] is True
    with zipfile.ZipFile(seal.archive_path) as zf:
        names=set(zf.namelist())
        assert "puppet.json" in names
        assert {f"textures/V{i}.png" for i in range(8)}.issubset(names)
