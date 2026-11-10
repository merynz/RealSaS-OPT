#!/usr/bin/env node
/**
 * Real original authored Spine export -> blind flattened RGBA source input and
 * hidden source slot hide/isolate evidence. Official runtime is imported as
 * an ephemeral evaluation dependency under its own license, not vendored.
 * No copied artwork/model training authority. Native editor file is absent.
 */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import CanvasKitInit from "canvaskit-wasm";
import {Physics} from "@esotericsoftware/spine-core";
import {loadTextureAtlas,loadSkeletonData,SkeletonDrawable,SkeletonRenderer}
  from "@esotericsoftware/spine-canvaskit";

function seal(x) {return Buffer.from(JSON.stringify(x,null,2)+"\n")}
function sha(x) {return crypto.createHash("sha256").update(x).digest("hex")}
function requireTrue(ok,code) {if(!ok)throw Error("ORIGINAL_ARTIST_SOURCE_"+code)}
const SIZE=768;
const argv={};
for(let i=2;i<process.argv.length;i+=2)argv[process.argv[i].replace(/^--/,"")]=process.argv[i+1];

async function main(){
  requireTrue(argv.source&&argv.atlas&&argv.output,"CLI_SOURCE_ATLAS_OUTPUT");
  const source=await fs.readFile(argv.source);
  const atlasText=await fs.readFile(argv.atlas);
  const skeletonJSON=JSON.parse(source.toString());
  const ck=await CanvasKitInit();
  const read=async p=>fs.readFile(p);
  const atlas=await loadTextureAtlas(ck,argv.atlas,read);
  const data=await loadSkeletonData(argv.source,atlas,read);
  const draw=new SkeletonDrawable(data);
  const painter=new SkeletonRenderer(ck);
  draw.skeleton.setupPose();
  draw.update(0);
  const bounds=draw.skeleton.getBoundsRect();
  requireTrue(bounds&&bounds.width>0&&bounds.height>0,"SOURCE_SETUP_BOUNDS_EMPTY");
  const scale=Math.min(720/bounds.width,720/bounds.height);
  requireTrue(Number.isFinite(scale)&&scale>0,"ORIGINAL_BOUNDS_SCALE");
  const skeleton=draw.skeleton;
  skeleton.scaleX=scale;skeleton.scaleY=-scale;
  skeleton.x=SIZE/2-(bounds.x+bounds.width/2)*scale;
  skeleton.y=SIZE/2+(bounds.y+bounds.height/2)*scale;
  skeleton.updateWorldTransform(Physics.update);
  const surface=ck.MakeSurface(SIZE,SIZE);
  requireTrue(surface,"CANVASKIT_SURFACE_NULL");
  const canvas=surface.getCanvas();
  const info={width:SIZE,height:SIZE,colorType:ck.ColorType.RGBA_8888,
              alphaType:ck.AlphaType.Unpremul,colorSpace:ck.ColorSpace.SRGB};
  const dest=ck.Malloc(Uint8Array,SIZE*SIZE*4);
  function render() {
    canvas.clear(ck.TRANSPARENT);
    skeleton.updateWorldTransform(Physics.update);
    painter.render(canvas,skeleton);
    const status=canvas.readPixels(0,0,info,dest);
    requireTrue(status!==false,"CANVASKIT_READ_RGBA_FAILED");
    return Buffer.from(dest.toTypedArray().slice());
  }
  function occupied(bytes){
    let total=0;for(let i=3;i<bytes.length;i+=4)total+=(bytes[i]>0)?1:0;
    return total;
  }
  function differs(a,b){
    let changed=0;for(let i=0;i<a.length;i+=4){
      if(a[i]!==b[i]||a[i+1]!==b[i+1]||a[i+2]!==b[i+2]||a[i+3]!==b[i+3])changed++;
    }
    return changed;
  }
  const baseline=render();
  requireTrue(baseline.equals(render()),"ORIGINAL_RUNTIME_NOOP_NONDETERMINISTIC");
  requireTrue(occupied(baseline)>100,"ORIGINAL_ARTWORK_ALPHA_EMPTY");
  const slots=skeleton.slots;
  const attachments=slots.map(s=>s.pose.attachment);
  const active=slots.map((slot,i)=>({slot,i,attachment:attachments[i]}))
                    .filter(o=>o.attachment!=null);
  requireTrue(active.length>0,"ORIGINAL_NO_ACTIVE_SLOTS");
  const privateRows=[];
  const out=path.resolve(argv.output);
  const modelDir=path.join(out,"model_input");
  const courtDir=path.join(out,"private_artist_court");
  await fs.mkdir(modelDir,{recursive:true});await fs.mkdir(courtDir,{recursive:true});
  for(const selected of active){
    for(const slot of slots)slot.pose.setAttachment(null);
    selected.slot.pose.setAttachment(selected.attachment);
    const isolated=render();
    slots.forEach((slot,j)=>slot.pose.setAttachment(j===selected.i?null:attachments[j]));
    const hidden=render();
    const stem="slot"+String(selected.i);
    await fs.writeFile(path.join(courtDir,stem+"_isolated.rgba"),isolated);
    await fs.writeFile(path.join(courtDir,stem+"_hidden.rgba"),hidden);
    privateRows.push({
      original_slot_index_teacher_only:selected.i,
      original_slot_name_teacher_only:selected.slot.data.name,
      original_attachment_name_teacher_only:selected.attachment.name,
      original_isolated_source_RGBA_sha256:sha(isolated),
      original_hidden_source_RGBA_sha256:sha(hidden),
      original_isolated_alpha_pixel_count:occupied(isolated),
      original_hide_changes_composite_RGBA_pixel_count:differs(hidden,baseline)
    });
  }
  slots.forEach((slot,j)=>slot.pose.setAttachment(attachments[j]));
  requireTrue(baseline.equals(render()),"ORIGINAL_ARTIST_EDIT_RESTORE_FAILED");
  // The only image the future PRODUCT algorithm may read.
  await fs.writeFile(path.join(modelDir,"anonymous_flattened_rgba.bin"),baseline);
  const publicModel={
    schema:"RealSaS.IndependentHumanArtNativeRGBAImageOnlyInput.v1",
    width:SIZE,height:SIZE,
    format:"RGBA8888_UNPREMULTIPLIED",
    image_RGBA_SHA256:sha(baseline),
    original_artist_slot_names_present:false,
    original_artist_layer_count_present:false,
    source_mesh_face_incidence_present:false,
    source_bone_rig_or_attachment_names_present:false,
    independent_source_artist_hidden_reference_available_to_algorithm:false,
    consumer_image_only_source_input:true,
    source_native_editor_artwork_original_CAA_qualified:false,
    training_authorized:false
  };
  await fs.writeFile(path.join(modelDir,"manifest.json"),seal(publicModel));
  const court={
    schema:"RealSaS.IndependentOriginalArtistSlotRGBAHideIsolateCourtOnly.v1",
    original_source_skeleton_export_SHA256:sha(source),
    original_source_atlas_SHA256:sha(atlasText),
    official_spine_runtime_reference:true,
    original_model_input_RGBA_SHA256:sha(baseline),
    original_artist_slot_count_teacher_only:skeletonJSON.slots.length,
    active_original_author_slot_count_teacher_only:active.length,
    source_author_slot_hide_isolate_references_teacher_only:privateRows,
    native_editable_source_artist_project_file_verified:false,
    original_hide_isolate_restoration_exact:true,
    source_art_layers_as_IRIS_TESSA_training_authority:false,
    training_authorized:false
  };
  const courtBytes=seal(court);
  await fs.writeFile(path.join(courtDir,"reference.json"),courtBytes);
  const receipt={
    schema:"RealSaS.IndependentOriginalArtistHeadlessRasterSourceReceipt.v1",
    status:"OFFICIAL_RUNTIME_ORIGINAL_SOURCE_RGBA_AND_PRIVATE_HIDE_ISOLATE_REFS_PRODUCED__NO_MODEL_OWNER",
    original_source_export_SHA256:sha(source),
    source_atlas_SHA256:sha(atlasText),
    original_rendered_RGBA_SHA256:sha(baseline),
    anonymous_flattened_image_manifest_SHA256:sha(seal(publicModel)),
    private_hidden_art_behavior_reference_SHA256:sha(courtBytes),
    original_artist_slots_in_source_teacher_only:skeletonJSON.slots.length,
    original_active_attachments_teacher_only:active.length,
    author_hide_isolate_measured_pair_count:privateRows.length,
    source_isolated_artwork_nonzero_alpha_cases:privateRows.filter(x=>x.original_isolated_alpha_pixel_count>0).length,
    source_hide_causes_observable_RGBA_change_cases:privateRows.filter(x=>x.original_hide_changes_composite_RGBA_pixel_count>0).length,
    source_hidden_or_compositor_masked_hide_cases:privateRows.filter(x=>x.original_hide_changes_composite_RGBA_pixel_count===0).length,
    original_raster_occupied_alpha_pixels:occupied(baseline),
    source_renderer:"OFFICIAL_SPINE_CANVASKIT",
    original_source_runtime_image_and_alpha_rendered:true,
    comparison_against_independent_RealSaS_compositor:false,
    original_editable_spine_project_verified:false,
    source_layer_identifiers_in_model_input:false,
    model_recovered_artist_drawables_measured:false,
    product_source_image_only_benchmark_input_created:true,
    external_artwork_redistribution_authorized:false,
    training_authorized:false
  };
  await fs.writeFile(path.join(out,"receipt.json"),seal(receipt));
  console.log(JSON.stringify(receipt));
  if(typeof ck.Free==="function")ck.Free(dest);
  surface.delete();
}
main().catch(e=>{console.error(String(e.stack||e));process.exit(9)});
