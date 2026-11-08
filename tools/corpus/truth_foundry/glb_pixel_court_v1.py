#!/usr/bin/env python3
"""Source-addressed canonical GLB pixel/UV candidate court.

Verifies *canonical GLB's own bytes*, not an original Objaverse source, 2D-art
approval, true backside visibility, rig controls, or author texture authority.
No render targets are minted by this importer.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,struct
from pathlib import Path
from PIL import Image,UnidentifiedImageError

SCHEMA='RealSaS.TruthFoundryCanonicalGLBPixelEvidence.v1'
HEADS=['IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA']
MAX_IMAGE_PIXELS=4096*4096


def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def read_glb(path):
    blob=Path(path).read_bytes()
    if len(blob)<20 or len(blob)>30*1024*1024:raise ValueError('GLB_SIZE_OUT_OF_BOUNDS')
    magic,version,total=struct.unpack_from('<4sII',blob,0)
    if magic!=b'glTF' or version!=2 or total!=len(blob):raise ValueError('INVALID_GLB_HEADER')
    pos=12;chunks=[]
    while pos+8<=len(blob):
        size,kind=struct.unpack_from('<I4s',blob,pos);pos+=8
        if size>len(blob)-pos or size%4:raise ValueError('INVALID_GLB_CHUNK_SIZE')
        chunks.append((kind,blob[pos:pos+size]));pos+=size
    if pos!=len(blob) or not chunks or chunks[0][0]!=b'JSON':raise ValueError('INVALID_GLB_CHUNK_SEQUENCE')
    root=json.loads(chunks[0][1].decode('utf-8').strip('\0 \n\r\t'))
    binary=next((chunk for kind,chunk in chunks if kind==b'BIN\0'),b'')
    return root,binary,sha_bytes(blob)


def audit(path):
    root,binary,glb_sha=read_glb(path)
    images=root.get('images',[])
    views=root.get('bufferViews',[])
    tex=root.get('textures',[])
    mats=root.get('materials',[])
    meshes=root.get('meshes',[])
    accessors=root.get('accessors',[])
    stats={'glb_sha256':glb_sha,'glb_images_declared':len(images),'glb_textures_declared':len(tex),
           'glb_materials_declared':len(mats),'glb_meshes_declared':len(meshes),
           'glb_skin_count':len(root.get('skins',[])),
           'glb_animation_count':len(root.get('animations',[])),
           'embedded_image_bytes_decoded':0,'decoded_image_count':0,
           'decoded_image_total_pixels':0,'decoded_image_alpha_channel_count':0,
           'mesh_primitive_count':0,'primitive_has_texcoord0_count':0,
           'textured_material_primitive_count':0,'primitive_with_texture_and_uv_count':0,
           'material_blend_count':0,'material_mask_count':0,
           'source_native_pixel_authority':'CANONICAL_GLB_ONLY__ORIGINAL_SOURCE_PARITY_UNKNOWN',
           'source_original_byte_sha_verified':False,'source_original_rights_approved':False,
           'original_3d_source_texture_uv_parity_pass':False,
           'source_front_back_eight_view_target_pass':False,
           'approved_stylized_2d_art_pass':False,'actual_graphics_training_admitted':False}
    img_info=[]
    for i,x in enumerate(images):
        row={'index':i,'mime_type':x.get('mimeType') or 'UNKNOWN','status':'UNKNOWN',
             'bufferView':x.get('bufferView'),'sha256':None,'width':None,'height':None,'alpha':False}
        idx=x.get('bufferView')
        if isinstance(idx,int) and 0<=idx<len(views):
            v=views[idx]
            offset=v.get('byteOffset',0);size=v.get('byteLength',-1)
            if not all(isinstance(n,int) for n in (offset,size)) or offset<0 or size<=0 or size>16*1024*1024 or offset+size>len(binary):
                row['status']='BAD_BUFFER_VIEW'
            else:
                imagebytes=binary[offset:offset+size]
                row['sha256']=sha_bytes(imagebytes)
                row['bytes']=len(imagebytes)
                try:
                    with Image.open(io.BytesIO(imagebytes)) as im:
                        if im.width*im.height>MAX_IMAGE_PIXELS or im.width<=0 or im.height<=0:raise ValueError('EXCESSIVE_PIXELS')
                        im.load()
                        row.update(status='DECODED_CANONICAL_GLB_IMAGE',width=im.width,height=im.height,
                                   alpha=('A' in im.getbands() or 'transparency' in im.info))
                        stats['decoded_image_count']+=1
                        stats['decoded_image_total_pixels']+=im.width*im.height
                        stats['embedded_image_bytes_decoded']+=len(imagebytes)
                        stats['decoded_image_alpha_channel_count']+=int(row['alpha'])
                except (UnidentifiedImageError,ValueError,OSError) as e:
                    row['status']='IMAGE_DECODE_FAIL__'+type(e).__name__
        elif 'uri' in x:
            row['status']='EXTERNAL_URI_NOT_RECOVERED'
        else:row['status']='MISSING_IMAGE_BUFFER_REFERENCE'
        img_info.append(row)
    material_texture=set()
    for i,m in enumerate(mats):
        stats['material_blend_count']+=int(m.get('alphaMode')=='BLEND')
        stats['material_mask_count']+=int(m.get('alphaMode')=='MASK')
        pbr=m.get('pbrMetallicRoughness') or {}
        refs=[pbr.get('baseColorTexture'),pbr.get('metallicRoughnessTexture'),m.get('normalTexture'),m.get('emissiveTexture'),m.get('occlusionTexture')]
        if any(isinstance(r,dict) and isinstance(r.get('index'),int) and 0<=r['index']<len(tex) for r in refs):material_texture.add(i)
    for mesh in meshes:
        for p in mesh.get('primitives',[]):
            stats['mesh_primitive_count']+=1
            attrs=p.get('attributes') or {}
            uv_idx=attrs.get('TEXCOORD_0')
            has_uv=isinstance(uv_idx,int) and 0<=uv_idx<len(accessors) and accessors[uv_idx].get('type')=='VEC2'
            has_tex=p.get('material') in material_texture
            stats['primitive_has_texcoord0_count']+=int(has_uv)
            stats['textured_material_primitive_count']+=int(has_tex)
            stats['primitive_with_texture_and_uv_count']+=int(has_uv and has_tex)
    stats['embedded_image_details']=img_info
    stats['candidate_pixel_rasterization_possible']=(stats['decoded_image_count']>0 and stats['primitive_with_texture_and_uv_count']>0)
    return stats


def run(input_glb_dir:Path, original_csv:Path, out:Path):
    input_glb_dir,original_csv,out=Path(input_glb_dir),Path(original_csv),Path(out)
    with original_csv.open(newline='',encoding='utf-8') as f:original=list(csv.DictReader(f))
    if len(original)!=96:raise ValueError('EXPECTED_96_GLBS_SOURCE_MANIFEST')
    out.mkdir(parents=True,exist_ok=True)
    rows=[];details=[]
    for row in original:
        if row['download_status']!='DOWNLOADED':continue
        fn=Path(row['repo_path']).name
        if fn!=row['object_type']+'.glb' or '/' in row['object_type'] or '\\' in row['object_type']:
            raise ValueError('SOURCE_KEY_FILENAME_MISMATCH')
        p=input_glb_dir/fn
        if not p.is_file():raise ValueError('MISSING_SHA_PINNED_GLB:'+fn)
        result=audit(p)
        if result['glb_sha256']!=row['download_sha256']:raise ValueError('INPUT_GLB_SHA_DIFF:'+fn)
        core={k:v for k,v in result.items() if k!='embedded_image_details'}
        core.update(provider='UNIML3D',object_type=row['object_type'],morphology_provider_claim=row['category_provider_claim'],file=fn)
        rows.append(core);details.append({'object_type':row['object_type'],'glb_sha256':result['glb_sha256'],
                                          'images':result['embedded_image_details']})
    if len(rows)!=84:raise ValueError('EXPECTED_84_VALID_DOWNLOADED_GLBS')
    keys=list(rows[0]);
    with (out/'CANONICAL_GLB_GRAPHICS_PIXEL_EVIDENCE_V1.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    with (out/'CANONICAL_GLB_IMAGE_PIXEL_HASHES_V1.jsonl').open('w',encoding='utf-8') as f:
        for r in details:f.write(json.dumps(r,sort_keys=True)+'\n')
    summary={'schema':SCHEMA,'analyzed_source_canonical_glbs':len(rows),
             'verified_same_canonical_byte_sha':len(rows),
             'embedded_image_decoded_source_count':sum(r['decoded_image_count']>0 for r in rows),
             'texture_and_uv_rasterization_candidate_count':sum(r['candidate_pixel_rasterization_possible'] for r in rows),
             'pixel_decoded_images_total':sum(r['decoded_image_count'] for r in rows),
             'canonical_source_pixels_not_original_authority':True,
             'original_source_appearance_parity_pass':0,'source_rights_qualified':0,
             'stylized_2d_art_approved':0,'eight_view_teacher_gt_pass':0,
             'front_to_back_student_prior_train_pairs_admitted':0,
             'all_head_training_admitted':{h:0 for h in HEADS},
             'input_manifest_sha256':sha_bytes(original_csv.read_bytes()),
             'file_sha256':{p.name:sha_bytes(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
    (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--glb-dir',type=Path,required=True)
    p.add_argument('--source-manifest-csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.glb_dir,a.source_manifest_csv,a.out),indent=2))
if __name__=='__main__':main()