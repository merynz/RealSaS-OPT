"""Pure topology and matrix safety courts; no fake native Blender admission."""
import pytest

from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    topology_of_polygons, finite_matrix,
)


def test_open_modular_triangle_mesh_is_not_solid_sdf():
    r=topology_of_polygons(4,[[0,1,2],[0,2,3]])
    assert r["boundary_edge_count"]==4
    assert r["source_index_component_count"]==1
    assert r["all_used_edges_two_sided"] is False
    assert r["watertight_solid_sdf_authorized"] is False
    assert r["drawable_count_minted"] is False


def test_closed_tetra_index_graph_does_not_mint_semantic_drawables():
    r=topology_of_polygons(4,[[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    assert r["boundary_edge_count"]==0
    assert r["all_used_edges_two_sided"] is True
    assert r["watertight_solid_sdf_authorized"] is False
    assert r["drawable_count_minted"] is False


def test_two_islands_and_loose_verts_are_observed_separately():
    r=topology_of_polygons(8,[[0,1,2],[3,4,5]])
    assert r["source_index_component_count"]==2
    assert r["loose_vertex_count"]==2
    assert r["boundary_edge_count"]==6


def test_invalid_topology_and_transform_refused():
    with pytest.raises(ValueError,match="POLYGON_VERTEX_OOB"):
        topology_of_polygons(3,[[0,1,3]])
    m=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,float("nan")]]
    with pytest.raises(ValueError,match="NONFINITE_IMPORT_MATRIX"):
        finite_matrix(m)
