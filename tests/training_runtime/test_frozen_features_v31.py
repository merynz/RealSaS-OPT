import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
import torch
from models.iris.v31.frozen_features_v31 import cache_identity, get_or_build_cache, extract_frozen_features


class FrozenCacheTests(unittest.TestCase):
    def identity(self, value='a'):
        return cache_identity(observation_bundle_sha256=value*64,camera_bundle_sha256='b'*64,
                 alpha_resampling_policy_sha256='c'*64,numerical_environment_sha256='d'*64)

    def features(self):
        return {'final':torch.ones(1,1,37,37,384),'layer2':torch.zeros(1,1,37,37,384),'alpha':torch.ones(1,1,37,37,196)}

    def test_hot_cache_avoids_extractor_and_input_binding_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            build=Mock(side_effect=self.features)
            cold, hit=get_or_build_cache(tmp,self.identity(),build=build)
            self.assertFalse(hit)
            hot, hit=get_or_build_cache(tmp,self.identity(),build=build)
            self.assertTrue(hit); self.assertEqual(build.call_count,1)
            self.assertTrue(torch.equal(cold['final'],hot['final']))
            get_or_build_cache(tmp,self.identity('e'),build=build)
            self.assertEqual(build.call_count,2)

    def test_corrupt_cache_fails_and_trainable_values_cannot_be_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            get_or_build_cache(tmp,self.identity(),build=self.features)
            path=next(Path(tmp).glob('*/features.pt'))
            with path.open('ab') as stream:stream.write(b'drift')
            with self.assertRaisesRegex(ValueError,'BYTES_OR_IDENTITY'):
                get_or_build_cache(tmp,self.identity(),build=self.features)
        with tempfile.TemporaryDirectory() as tmp:
            def trainable():
                out=self.features();out['layer2'].requires_grad_();return out
            with self.assertRaisesRegex(ValueError,'TENSOR_INVALID'):
                get_or_build_cache(tmp,self.identity(),build=trainable)

    def test_extractor_rejects_trainable_backbone_before_execution(self):
        model=torch.nn.Linear(1,1)
        with self.assertRaisesRegex(ValueError,'FROZEN_FP32_EVAL'):
            extract_frozen_features(model,torch.empty(0),torch.empty(0),view_batch_size=1)

    def test_layer_index_norm_patch_layout_and_view_chunk_equivalence(self):
        class Scale(torch.nn.Module):
            def __init__(self,factor):super().__init__();self.factor=factor
            def forward(self,x):return x*self.factor
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__();self.blocks=torch.nn.ModuleList([Scale(x) for x in (2,3,5,7)])
                self.norm=Scale(.1)
            def forward_features(self,x):
                token=x[:,0,0,0].float()[:,None,None].expand(-1,1370,384)
                for block in self.blocks:token=block(token)
                return {'x_norm_patchtokens':self.norm(token)[:,1:]}
        model=Backbone().eval()
        rgb=torch.ones(1,2,3,1024,1024,dtype=torch.uint8)
        rgb[:,1]*=2
        alpha=torch.ones(1,2,1,518,518)
        a=extract_frozen_features(model,rgb,alpha,view_batch_size=1,preprocess=lambda x:x)
        b=extract_frozen_features(model,rgb,alpha,view_batch_size=2,preprocess=lambda x:x)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))
        self.assertEqual(float(a['layer2'][0,0,0,0,0]),3.)
        self.assertEqual(float(a['final'][0,0,0,0,0]),21.)
        self.assertEqual(float(a['final'][0,1,0,0,0]),42.)
        self.assertEqual(len(model.blocks[2]._forward_hooks),0)
