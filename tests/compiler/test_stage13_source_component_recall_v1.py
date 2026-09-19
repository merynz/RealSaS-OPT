from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    source_connected_component_recall_metrics,
)


def _mask(width,height,points):
    out=bytearray(width*height)
    for x,y in points:
        out[y*width+x]=1
    return bytes(out)


def test_component_recall_prevents_small_visible_component_from_hiding_in_global_recall():
    w=h=20
    body={(x,y) for y in range(2,18) for x in range(2,12)}
    accessory={(17,4),(17,5),(17,6),(17,7)}
    authority=_mask(w,h,body|accessory)
    predicted=_mask(w,h,body)
    result=source_connected_component_recall_metrics(
        authority,predicted,width=w,height=h,minimum_foreground_fraction=0.02
    )
    assert result["source_component_count"]==2
    assert result["eligible_component_count"]==2
    assert result["minimum_eligible_component_recall"]==0.0
    recalls=sorted(row["recall"] for row in result["components"] if row["eligible"])
    assert recalls==[0.0,1.0]


def test_component_fraction_floor_ignores_subthreshold_speckle_only():
    w=h=20
    body={(x,y) for y in range(2,18) for x in range(2,12)}
    speck={(19,19)}
    authority=_mask(w,h,body|speck)
    predicted=_mask(w,h,body)
    result=source_connected_component_recall_metrics(
        authority,predicted,width=w,height=h,minimum_foreground_fraction=0.01
    )
    assert result["source_component_count"]==2
    assert result["eligible_component_count"]==1
    assert result["minimum_eligible_component_recall"]==1.0
