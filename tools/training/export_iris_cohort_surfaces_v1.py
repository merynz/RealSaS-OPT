"""Learned shared checkpoint -> exact existing decoder + per-view raw court data."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from models.iris.training.cohort_v1 import SubjectCache,file_hash,value_hash,atomic_json
from models.iris.v4.scene_first_signed_v4 import SceneFirstSignedGeometryV4,SceneFirstSignedGeometryConfigV4
from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5,write_stage12_npz_v5
from models.iris.v5.contained_sparse_tetra_decoder_v5 import decode_indexed_sparse_tetra_mt_contained_v5
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,rasterize_triangles_half_integer_top_left,source_connected_component_recall_metrics)
from compiler.realsas_compiler_core.silhouette_metrics_v2 import silhouette_distance_metrics

ROOT=Path(__file__).resolve().parents[2]
COURT={"min_recall":.999,"min_precision":.999,"max_largest_coherent_hole_fraction":.00025,
       "max_interior_uncovered_fraction":.0005,"min_component_recall":.999,
       "component_min_foreground_fraction":0.,"max_silhouette_edge_p95_px":.5}

def per_view_court(vertices,faces,camera,source_mask):
    delta=vertices-camera["origin"]
    xy=np.stack([delta@camera["right"],-(delta@camera["screen_up"])],axis=1)
    xy=(xy/camera["half_extent"]+1.)*(camera["resolution"]*.5)
    r=camera["resolution"]
    predicted=rasterize_triangles_half_integer_top_left(xy[faces],width=r,height=r)
    metrics=coverage_metrics(source_mask,predicted,width=r,height=r)
    components=source_connected_component_recall_metrics(source_mask,predicted,width=r,height=r,
                                                          minimum_foreground_fraction=0.)
    pred=np.frombuffer(predicted,np.uint8).reshape(r,r)>0
    actual=np.frombuffer(source_mask,np.uint8).reshape(r,r)>0
    mean,p95,maximum=silhouette_distance_metrics(actual,pred)
    metrics.update(components=components,silhouette_edge_mean_px=mean,
                   silhouette_edge_p95_px=p95,silhouette_edge_max_px=maximum)
    component_recall=float(components["minimum_eligible_component_recall"])
    metrics["pass"]=(metrics["recall"]>=COURT["min_recall"] and metrics["precision"]>=COURT["min_precision"]
        and metrics["largest_coherent_hole_fraction"]<=COURT["max_largest_coherent_hole_fraction"]
        and metrics["interior_uncovered_fraction"]<=COURT["max_interior_uncovered_fraction"]
        and component_recall>=COURT["min_component_recall"] and p95<=COURT["max_silhouette_edge_p95_px"])
    return metrics,pred,actual

def export(checkpoint,data_root,input_root,output_root,persist_root,device="cuda",decoder_policy=None):
    checkpoint=Path(checkpoint);data_root=Path(data_root);input_root=Path(input_root)
    output_root=Path(output_root);persist_root=Path(persist_root)
    manifest=json.loads((data_root/"COHORT_MANIFEST.json").read_text())
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if state["binding"]["manifest_sha256"]!=value_hash(manifest):
        raise ValueError("SURFACE_COHORT_BINDING_DRIFT")
    model=SceneFirstSignedGeometryV4(SceneFirstSignedGeometryConfigV4(**state["model_config"])).to(device)
    model.load_state_dict(state["state_dict"],strict=True);model.eval()
    cache=SubjectCache(data_root,manifest,device,maximum=1)
    policy=decoder_policy or SparseRegularTetraPolicyV5()
    source={p:file_hash(ROOT/p) for p in (
        "models/iris/v5/indexed_sparse_tetra_decoder_v5.py",
        "models/iris/v5/contained_sparse_tetra_decoder_v5.py",
        "tools/training/export_iris_cohort_surfaces_v1.py")}
    checkpoint_hash=file_hash(checkpoint)
    reports=[]
    for index,row in enumerate(manifest["subjects"]):
        key=row["subject_id"];folder=output_root/key;folder.mkdir(parents=True,exist_ok=True)
        persisted=persist_root/key;persisted.mkdir(parents=True,exist_ok=True)
        binding={"checkpoint_sha256":checkpoint_hash,"observation_binding":row["observation_binding"],
                 "decoder_sources":source,"decoder_policy":asdict(policy),"court_policy":COURT}
        binding_hash=value_hash(binding)
        old=persisted/"SURFACE_COURT.json"
        if old.exists():
            prior=json.loads(old.read_text())
            if prior.get("binding_sha256")==binding_hash and (
                    prior["status"]=="FAIL" or (persisted/"zero_surface.npz").is_file()
                    and file_hash(persisted/"zero_surface.npz")==prior["npz_sha256"]):
                reports.append(prior);print(f"[DECODE_REUSE] {key} {prior['status']}",flush=True);continue
        started=time.monotonic()
        try:
            subject=cache.get(index)
            with torch.no_grad(),torch.autocast(str(device).split(":")[0],dtype=torch.bfloat16,enabled=str(device).startswith("cuda")):
                planes=model.encode_scene(subject["image_tokens"],subject["token_xy"],subject["camera_features"])
            queried=0;last=time.monotonic()
            def query(points):
                nonlocal queried,last
                with torch.no_grad():
                    values=model.query(planes.float(),torch.from_numpy(np.asarray(points,np.float32)).to(device)[None])["sdf"]
                queried+=len(points)
                if time.monotonic()-last>15:
                    print(f"[DECODE] {key} actual_field_queries={queried} elapsed_s={time.monotonic()-started:.1f}",flush=True);last=time.monotonic()
                return values.detach().float().cpu().numpy().reshape(-1)
            mesh=decode_indexed_sparse_tetra_mt_contained_v5(query,policy=policy,work_dir=folder/"scratch")
            npz=write_stage12_npz_v5(mesh,folder/"zero_surface.npz")
            cameras=json.loads((input_root/key/"camera_bundle.json").read_text())["cameras"]
            world=np.asarray(mesh.vertices_normalized)*subject["normalization_half_extent"]+subject["center_xyz"].cpu().numpy()
            views=[]
            for view,camera in enumerate(cameras):
                mask=(input_root/key/f"V{view}.mask.bin").read_bytes()
                metrics,pred,actual=per_view_court(world,np.asarray(mesh.faces),camera,mask)
                Image.fromarray(pred.astype(np.uint8)*255).save(folder/f"V{view}.predicted.png")
                overlay=np.stack([actual*255,pred*255,np.zeros_like(pred)],axis=-1).astype(np.uint8)
                Image.fromarray(overlay).save(folder/f"V{view}.comparison.png")
                views.append({"view_index":view,**metrics})
                print(f"[COURT] {key} V{view} pass={metrics['pass']} recall={metrics['recall']:.6f} "
                      f"precision={metrics['precision']:.6f} edge_p95={metrics['silhouette_edge_p95_px']:.2f}",flush=True)
            report={"status":"PASS_RAW_COURT" if all(v["pass"] for v in views) else "FAIL",
                    "npz_sha256":file_hash(npz),"vertex_count":len(mesh.vertices_normalized),"face_count":len(mesh.faces),
                    "views":views,"decoder_diagnostics":mesh.diagnostics}
        except Exception as exc:
            report={"status":"FAIL","error_type":type(exc).__name__,"error":str(exc),"views":[]}
            print(f"[COURT_FAIL] {key} {type(exc).__name__}: {exc}",flush=True)
        report.update(schema="RealSaS.SharedIrisSurfaceCourtPayload.v1",subject_id=key,binding=binding,
                      binding_sha256=binding_hash,elapsed_seconds=time.monotonic()-started,
                      go_qualification_claimed=False,fit8_pass_claimed=False)
        atomic_json(folder/"SURFACE_COURT.json",report)
        for path in folder.iterdir():
            if path.is_file():
                staged=persisted/(path.name+".staged")
                shutil.copyfile(path,staged)
                if file_hash(staged)!=file_hash(path):raise ValueError("SURFACE_PERSISTENT_COPY_HASH")
                staged.replace(persisted/path.name)
        reports.append(report)
        shutil.rmtree(folder/"scratch",ignore_errors=True)
    summary={"schema":"RealSaS.SharedIrisCohortSurfaceCourt.v1","checkpoint_sha256":checkpoint_hash,
             "subjects":reports,"all_subjects_raw_court_pass":all(r["status"]=="PASS_RAW_COURT" for r in reports),
             "go_qualification_required":True,"fit8_pass_claimed":False}
    atomic_json(persist_root/"COHORT_SURFACE_COURT.json",summary)
    return summary
