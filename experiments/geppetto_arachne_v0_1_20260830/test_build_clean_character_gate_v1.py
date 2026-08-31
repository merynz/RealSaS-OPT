import importlib.util
import pathlib

P = pathlib.Path(__file__).with_name("build_clean_character_gate_v1.py")
spec = importlib.util.spec_from_file_location("gate", P); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def row(occ=.1, border=0, face=.01, tri=.02, sig=1, largest=1.0, centroid=.1):
    views=[]
    for _ in range(8):
        views.append({"occupancy_fraction":occ,"max_visible_triangle_pixel_fraction":tri,"significant_component_count":sig,"largest_component_fraction":largest,"centroid_offset_fraction":centroid})
    return {"aggregate":{"border_touch_view_count":border},"mesh":{"max_face_area_fraction":face},"views":views}


def test_objective_pass_clean():
    s,h,f=gate.objective_decision(row())
    assert s == "PASS_OBJECTIVE_RENDER_C0" and h == [] and f == []


def test_tiny_is_hard_fail():
    s,h,f=gate.objective_decision(row(occ=.0049))
    assert s == "FAIL_OBJECTIVE_RENDER_C0" and h == ["TOO_SMALL_NATIVE_SUBSTRATE"]


def test_tail_flags_do_not_hard_fail():
    s,h,f=gate.objective_decision(row(face=.2,tri=.3,sig=2,largest=.89,centroid=.251))
    assert s == "PASS_OBJECTIVE_RENDER_C0" and h == []
    assert set(f) == {"MESH_FACE_DOMINANCE","VISIBLE_TRIANGLE_DOMINANCE","MULTI_COMPONENT","FRAGMENTED_SUPPORT","OFF_CENTER"}
