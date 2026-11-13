"""Content-bound cache of frozen features, never trainable scene/endpoint state.

L2 means zero-based block index 2 normalized by the frozen model's final norm,
matching the feature-access/fusion apparatus. Alpha resampling is deliberately
not guessed here: the observation producer must seal its exact 518px raster and
resampling policy. DINO's RGB transform is the existing whole1024->518 contract.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch

from models.iris.v2.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2, preprocess_dinov2_rgb_flat_v2,
)
from models.training_runtime.durable_state_v1 import atomic_json, file_sha256
from .model_v31 import alpha_patches_v31

SCHEMA = 'RealSaS.IRIS.FrozenFeatureCache.v3_1'


def cache_identity(*, observation_bundle_sha256, camera_bundle_sha256, alpha_resampling_policy_sha256,
                   numerical_environment_sha256):
    hashes = locals().copy()
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in hashes.values()):
        raise ValueError('FROZEN_CACHE_EXACT_HASH_BINDINGS_REQUIRED')
    authority = DINOv2FoundationAuthorityV2()
    return {'schema':SCHEMA, **hashes, 'foundation_authority_sha256':authority.authority_hash,
            'rgb_preprocessing':authority.preprocessing_id,
            'layer2':'BLOCK_INDEX_2__FINAL_NORM__PATCH_ONLY', 'final':'X_NORM_PATCHTOKENS',
            'alpha':'SEALED_518_RASTER__ROW_MAJOR_14X14', 'storage_dtype':'float32',
            'cached_trainable_values':False}


def _digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def extract_frozen_features(model, native_rgb, alpha_518, *, view_batch_size, preprocess=preprocess_dinov2_rgb_flat_v2):
    if model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FROZEN_CACHE_BACKBONE_MUST_BE_FROZEN_FP32_EVAL')
    if (native_rgb.ndim != 5 or native_rgb.shape[2:] != (3, 1024, 1024)
            or alpha_518.shape != (*native_rgb.shape[:2], 1, 518, 518)
            or view_batch_size <= 0):
        raise ValueError('FROZEN_CACHE_OBSERVATION_SHAPE')
    if not torch.isfinite(alpha_518).all() or (alpha_518 < 0).any() or (alpha_518 > 1).any():
        raise ValueError('FROZEN_CACHE_ALPHA_INVALID')
    b, views = native_rgb.shape[:2]
    parameters = [(p, p._version) for p in model.parameters()]
    final, layer2 = [], []
    for images in native_rgb.reshape(-1, 3, 1024, 1024).split(view_batch_size):
        captured = []
        hook = model.blocks[2].register_forward_hook(lambda module, args, output:captured.append(output))
        try:
            # Cache construction has one declared FP32 numerical policy even if
            # the surrounding trainer uses BF16. Storage precision is separate.
            with torch.autocast(device_type=images.device.type, enabled=False):
                result = model.forward_features(preprocess(images))
                if len(captured) != 1:
                    raise ValueError('FROZEN_CACHE_LAYER_CAPTURE_DRIFT')
                detail = model.norm(captured[0])[:, 1:]
                last = result['x_norm_patchtokens']
        finally:
            hook.remove()
        if last.shape != (len(images), 1369, 384) or detail.shape != last.shape:
            raise ValueError('FROZEN_CACHE_TOKEN_LAYOUT_DRIFT')
        if not torch.isfinite(last).all() or not torch.isfinite(detail).all():
            raise ValueError('FROZEN_CACHE_NONFINITE_FEATURES')
        final.append(last.float().cpu())
        layer2.append(detail.float().cpu())
    if any(p._version != version for p, version in parameters):
        raise ValueError('FROZEN_CACHE_BACKBONE_MUTATED')
    return {'final':torch.cat(final).reshape(b, views, 37, 37, 384).contiguous(),
            'layer2':torch.cat(layer2).reshape(b, views, 37, 37, 384).contiguous(),
            'alpha':alpha_patches_v31(alpha_518.float()).cpu().contiguous()}


def _validate_payload(payload, identity):
    if not isinstance(payload, dict) or set(payload) != {'identity','features'} or payload['identity'] != identity:
        raise ValueError('FROZEN_CACHE_BINDING_DRIFT')
    features = payload['features']
    if not isinstance(features, dict) or set(features) != {'final','layer2','alpha'}:
        raise ValueError('FROZEN_CACHE_FIELDS_DRIFT')
    final = features['final']
    if (not isinstance(final, torch.Tensor) or final.ndim != 5 or final.shape[-3:] != (37,37,384)
            or min(final.shape[:2]) <= 0):
        raise ValueError('FROZEN_CACHE_SHAPE_DRIFT')
    for key, channels in (('final',384),('layer2',384),('alpha',196)):
        tensor = features[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (*final.shape[:4],channels)
                or tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                or tensor.requires_grad or not torch.isfinite(tensor).all()):
            raise ValueError('FROZEN_CACHE_TENSOR_INVALID')
    if (features['alpha'] < 0).any() or (features['alpha'] > 1).any():
        raise ValueError('FROZEN_CACHE_ALPHA_INVALID')
    return features


def get_or_build_cache(root, identity, *, build):
    # Callers must first verify the input packet, loaded foundation authority and
    # environment whose hashes are in identity. Cache read is not that admission.
    if identity.get('schema') != SCHEMA or identity.get('cached_trainable_values') is not False:
        raise ValueError('FROZEN_CACHE_IDENTITY_INVALID')
    folder = Path(root)/_digest(identity)
    receipt_path = folder/'CACHE.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        path = folder/'features.pt'
        if (receipt.get('identity') != identity or path.is_symlink() or not path.is_file()
                or path.stat().st_size != receipt['size_bytes'] or file_sha256(path) != receipt['sha256']):
            raise ValueError('FROZEN_CACHE_BYTES_OR_IDENTITY_DRIFT')
        features = _validate_payload(torch.load(path, map_location='cpu', weights_only=True), identity)
        return features, True
    folder.mkdir(parents=True, exist_ok=True)
    payload = {'identity':identity, 'features':build()}
    features = _validate_payload(payload, identity)
    temp = folder/(uuid.uuid4().hex+'.partial')
    try:
        with temp.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        digest, size = file_sha256(temp), temp.stat().st_size
        os.replace(temp, folder/'features.pt')
        atomic_json(receipt_path, {'schema':SCHEMA,'identity':identity,'sha256':digest,'size_bytes':size})
    finally:
        temp.unlink(missing_ok=True)
    return features, False
