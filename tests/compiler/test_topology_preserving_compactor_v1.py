import numpy as np

from compiler.realsas_compiler_core.substrate.topology_preserving_compactor_v1 import topology_preserving_component_voxel_compact_v1


def test_disconnected_same_voxel_patches_are_not_fused():
    # Two disconnected quads are spatially close and intentionally share coarse voxels.
    p=np.array([
        [0.00,0.00,0.00],[0.10,0.00,0.00],[0.10,0.10,0.00],[0.00,0.10,0.00],
        [0.00,0.00,0.01],[0.10,0.00,0.01],[0.10,0.10,0.01],[0.00,0.10,0.01],
    ],np.float64)
    f=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],np.int64)
    n=np.tile(np.array([[0,0,1.]],np.float64),(8,1))
    # target_nodes >=64 is a production guard, so pad with disconnected far triangles
    # while keeping the two close patches as the case under test.
    extra=[]; faces=f.tolist(); normals=n.tolist(); base=len(p)
    for k in range(20):
        x=10.0+k
        tri=[[x,0,0],[x+0.1,0,0],[x,0.1,0]]
        start=base+len(extra)
        extra.extend(tri); normals.extend([[0,0,1]]*3); faces.append([start,start+1,start+2])
    p=np.concatenate([p,np.asarray(extra)],axis=0)
    n=np.asarray(normals); f=np.asarray(faces,np.int64)
    r=topology_preserving_component_voxel_compact_v1(p,f,n,target_nodes=64,max_divisions=8)
    assert r.compact_node_count<=64
    # First two patches must never map to the same compact id even when their XYZ
    # falls in the same coarse voxel.
    assert set(r.inverse[:4]).isdisjoint(set(r.inverse[4:8]))
    assert r.split_voxel_count>=1
