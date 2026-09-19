import hashlib

from compiler.realsas_compiler_core.camera_authority_v1 import camera_projection_binding_hash
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import derive_component_observation_rasters_v1
from compiler.realsas_compiler_core.observation_authority_v1 import QualifiedObservationViewIR, build_qualified_observation_set
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, ComponentRegionIR, build_component_carrier_policy, build_mechanical_partition,
)
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode

def test_g5_component_masks_derive_from_source_foreground_and_S():
    cameras=tuple(CameraProjectionV3(f"V{i}",i,(0,0,-2),(1,0,0),(0,1,0),(0,0,1),1.0,8) for i in range(8))
    nodes=(
        SurfaceNode("a",(-.5,0,0),tuple(range(8)),("p",),(),tuple((i,(1.0,4.0)) for i in range(8)),derived_normal=(0,0,1)),
        SurfaceNode("b",(.5,0,0),tuple(range(8)),("p",),(),tuple((i,(6.0,4.0)) for i in range(8)),derived_normal=(0,0,1)),
    )
    surface=RiggingSurfaceIR(nodes,(),"S","test")
    partition=build_mechanical_partition(
        surface=surface,
        components=(ComponentRegionIR("left",("a",)),ComponentRegionIR("right",("b",))),
        boundary_constraints=(),
    )
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=(ComponentCarrierDecisionIR("left","MESH",("auto",)),ComponentCarrierDecisionIR("right","MESH",("auto",))),
    )
    fg=bytes([1])*64
    views=tuple(QualifiedObservationViewIR(
        i,8,8,f"obs{i}","a"*64,hashlib.sha256(fg).hexdigest(),
        camera_projection_binding_hash(cameras[i]),"PASS",("e",)
    ) for i in range(8))
    obs=build_qualified_observation_set(views)
    rows=derive_component_observation_rasters_v1(
        surface=surface,partition=partition,carrier_policy=carrier,
        observation_set=obs,source_foreground_masks={i:fg for i in range(8)},cameras=cameras,
    )
    assert len(rows)==16
    for view in range(8):
        rs=[r for r in rows if r.view_index==view]
        assert all(r.metadata["external_component_mask_used"] is False for r in rs)
        assert all(sum(int(r.mask_bytes[p]) for r in rs)==1 for p in range(64))
