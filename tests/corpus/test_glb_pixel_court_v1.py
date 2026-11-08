import io, json, struct, tempfile, unittest
from pathlib import Path
from PIL import Image
from tools.corpus.truth_foundry.glb_pixel_court_v1 import audit

class GLBPixelCourtV1Test(unittest.TestCase):
    def make_glb(self, *, image=True, uv=True, alpha='BLEND', external=False):
        im=Image.new('RGBA',(4,4),(25,115,200,120))
        buf=io.BytesIO();im.save(buf,format='PNG');b=buf.getvalue()
        pad=lambda v:v+b'\0'*((-len(v))%4)
        root={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(pad(b))}],
              'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(b)}],
              'images':[{'mimeType':'image/png','bufferView':0}] if image else [],
              'textures':[{'source':0}] if image else [],
              'materials':[{'alphaMode':alpha,'pbrMetallicRoughness':{'baseColorTexture':{'index':0}} if image else {}}],
              'accessors':[{'type':'VEC2','componentType':5126,'count':4}],
              'meshes':[{'primitives':[{'material':0,'attributes':{'POSITION':0,**({'TEXCOORD_0':0} if uv else {})}}]}],
              'skins':[{'joints':[0]}], 'nodes':[{}]}
        if external:
            root['images']=[{'uri':'texture.png'}]
        j=json.dumps(root).encode();j=j+b' '*((-len(j))%4)
        bin_=pad(b)
        header=struct.pack('<4sII',b'glTF',2,12+8+len(j)+8+len(bin_))
        return header+struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(bin_),b'BIN\0')+bin_
    def write(self,b):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup)
        p=Path(d.name)/'test.glb';p.write_bytes(b);return p
    def test_decoded_canonical_rgba_source_uv_is_candidate_only(self):
        r=audit(self.write(self.make_glb()))
        self.assertEqual(r['decoded_image_count'],1)
        self.assertEqual(r['decoded_image_alpha_channel_count'],1)
        self.assertEqual(r['primitive_with_texture_and_uv_count'],1)
        self.assertTrue(r['candidate_pixel_rasterization_possible'])
        self.assertFalse(r['source_original_byte_sha_verified'])
        self.assertFalse(r['approved_stylized_2d_art_pass'])
        self.assertFalse(r['actual_graphics_training_admitted'])
    def test_no_uv_is_not_rasterization_ready(self):
        r=audit(self.write(self.make_glb(uv=False)))
        self.assertEqual(r['decoded_image_count'],1)
        self.assertFalse(r['candidate_pixel_rasterization_possible'])
    def test_external_image_is_not_source_pixel(self):
        r=audit(self.write(self.make_glb(external=True)))
        self.assertEqual(r['decoded_image_count'],0)
        self.assertEqual(r['embedded_image_details'][0]['status'],'EXTERNAL_URI_NOT_RECOVERED')
        self.assertFalse(r['candidate_pixel_rasterization_possible'])
    def test_tampered_header_rejected(self):
        b=bytearray(self.make_glb());b[0]=0
        with self.assertRaisesRegex(ValueError,'INVALID_GLB_HEADER'):
            audit(self.write(bytes(b)))

if __name__=='__main__':unittest.main()