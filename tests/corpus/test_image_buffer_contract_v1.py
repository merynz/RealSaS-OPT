import json
import pytest
from tools.truth_foundry.image_buffer_contract_v1 import validate_image_buffer_layout


@pytest.mark.parametrize('width,height,count',[(0,4,16),(4,0,16),(4,4,0),(4,4,63),(4,4,80)])
def test_missing_or_malformed_pixels_fail_with_source_diagnostics(width,height,count):
    with pytest.raises(ValueError,match='TEACHER_SCENE_IMAGE_BUFFER_LAYOUT:') as failure:
        validate_image_buffer_layout(width=width,height=height,pixel_count=count,
            context={'image_name':'missing_basecolor','material':'body','image_filepath':'textures/body.png'})
    detail=json.loads(str(failure.value).split(':',1)[1])
    assert detail['image_name']=='missing_basecolor' and detail['material']=='body'
    assert detail['pixel_count']==count and detail['width']==width and detail['height']==height


@pytest.mark.parametrize('channels',[1,2,3,4])
def test_supported_pixel_layout_retains_exact_channel_count(channels):
    assert validate_image_buffer_layout(width=4,height=8,pixel_count=4*8*channels,context={})==channels
