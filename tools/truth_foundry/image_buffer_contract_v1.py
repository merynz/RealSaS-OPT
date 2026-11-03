"""Validate a Blender pixel buffer before allocation, retaining failure context."""
import json


def validate_image_buffer_layout(*, width, height, pixel_count, context):
    diagnostic = {**context, 'width':int(width), 'height':int(height), 'pixel_count':int(pixel_count)}
    if width<=0 or height<=0 or pixel_count<=0 or pixel_count%(width*height):
        raise ValueError('TEACHER_SCENE_IMAGE_BUFFER_LAYOUT:'+json.dumps(diagnostic,sort_keys=True))
    channels = pixel_count//(width*height)
    if channels not in (1,2,3,4):
        raise ValueError('TEACHER_SCENE_IMAGE_BUFFER_LAYOUT:'+json.dumps(diagnostic,sort_keys=True))
    return channels
