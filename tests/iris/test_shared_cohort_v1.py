import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from models.iris.training.cohort_v1 import SharedIrisTrainer, AtomicCheckpointStore, file_hash, value_hash
from models.iris.training.apparatus_v1 import certified_ray_pools,training_rays,training_queries,straddle_pool,targets
from models.iris.v5.source_signed_field_v5 import source_signed_distance_fields_v5, SourceSignedFieldPolicyV5
from tools.corpus.iris_observation_admission_v1 import project_mask
from tools.observation.source_scene_contract_v2 import prepare_bind_rest_gltf

torch.set_num_threads(2)

def fixture(root):
    policy = {"subject_count":8,"seed":31,"steps":16,"queries_per_step":32,
              "learning_rate":.0001,"minimum_learning_rate":.00001,"weight_decay":0.,
              "gradient_clip":1.,"huber_beta":.01,"near_zero_band":.03,"sign_loss_weight":0.,
              "hard_replay_bank_size":16,"silhouette_pool_per_stratum":2,
              "source_boundary_band_px":8.,"source_background_support_px":32.,
              "silhouette_loss_weight":1.,"silhouette_rays_per_step":32,
              "silhouette_depth_samples":4,"silhouette_softmin_tau":.001953125,
              "log_interval":8,"checkpoint_interval":4,"evaluation_interval":8,
              "model_config":{"image_token_dim":8,"scene_dim":16,"camera_embed_dim":8,
                              "plane_size":2,"plane_channels":4,"field_hidden_dim":8,
                              "field_residual_blocks":1,"attention_heads":2,"scene_layers":1}}
    generator = torch.Generator().manual_seed(11)
    resolution = 16
    yy,xx = np.mgrid[:resolution,:resolution]
    mask = (xx-7.5)**2+(yy-7.5)**2<25
    signed = source_signed_distance_fields_v5(np.stack([mask]*8),
                 policy=SourceSignedFieldPolicyV5(required_resolution=resolution))
    subjects = []
    for i in range(8):
        key = f"fixture_{i}"
        data = {"subject_id":key,"observation_binding":key,"resolution":resolution,
                "image_tokens":torch.rand(1,8,4,8,generator=generator),
                "token_xy":torch.rand(1,8,4,2,generator=generator),
                "camera_features":torch.rand(1,8,13,generator=generator),
                "signed_2d":torch.from_numpy(signed),"center_xyz":torch.zeros(3),
                "normalization_half_extent":1.,"camera_origins":torch.zeros(8,3),
                "camera_right":torch.tensor([[1.,0.,0.]]*8),
                "camera_screen_up":torch.tensor([[0.,1.,0.]]*8),
                "camera_forward":torch.tensor([[0.,0.,-1.]]*8),
                "boundary_pixels":[torch.tensor([[3,3],[4,4],[7,7],[10,10],[11,11]]) for _ in range(8)],
                "camera_half_extent":torch.ones(8),
                "near_points":torch.rand(64,3,generator=generator)*2-1,
                "straddle_points":torch.rand(64,3,generator=generator)*2-1,
                "replay_points":torch.rand(64,3,generator=generator)*2-1,
                "heldout_points":torch.rand(32,3,generator=generator)*2-1}
        from models.iris.training.cohort_v1 import field_targets
        data["heldout_targets"] = field_targets(data,data["heldout_points"])
        data["replay_targets"] = field_targets(data,data["replay_points"])
        data.update(certified_ray_pools(data,policy,seed=19))
        path = root/(key+".pt");torch.save(data,path)
        subjects.append({"subject_id":key,"observation_binding":key,
                         "shard_file":path.name,"shard_sha256":file_hash(path)})
    return {"subjects":subjects,"teacher_inputs_in_forward":False}, policy

class SharedCohortTests(unittest.TestCase):
    def test_exterior_sign_audit_fails_wrong_sign_and_source_count_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,policy=fixture(root)
            s=torch.load(root/manifest['subjects'][0]['shard_file'],weights_only=True)
            from tools.training.iris_exterior_audit_v1 import exterior_scan
            census=exterior_scan(s,resolution=8,device='cpu')
            positive=exterior_scan(s,lambda q:np.ones(len(q)),resolution=8,device='cpu',expected_count=census['certified_exterior_count'])
            negative=exterior_scan(s,lambda q:-np.ones(len(q)),resolution=8,device='cpu',expected_count=census['certified_exterior_count'])
            self.assertTrue(positive['sign_pass']);self.assertFalse(positive['production_r512'])
            self.assertFalse(negative['sign_pass'])
            self.assertEqual(negative['nonpositive_certified_exterior_count'],census['certified_exterior_count'])
            with self.assertRaisesRegex(ValueError,'COUNT_DRIFT'):
                exterior_scan(s,lambda q:np.ones(len(q)),resolution=8,device='cpu',expected_count=census['certified_exterior_count']+1)
    def test_knight_reference_rejects_optimizer_or_core_drift(self):
        from models.iris.training.cohort_v1 import validate_knight_recipe,source_identity
        root=Path(__file__).resolve().parents[2]
        policy=json.loads((root/"canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json").read_text())
        source=source_identity(root)
        validate_knight_recipe(policy,source)
        for name,bad in (("weight_decay",.01),("huber_beta",.02),("silhouette_loss_weight",0.),("initial_sdf_bias",0.)):
            changed=copy.deepcopy(policy);changed[name]=bad
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"POLICY_DRIFT"):
                validate_knight_recipe(changed,source)
        changed=copy.deepcopy(source);changed["models/iris/v4/scene_first_signed_v4.py"]="wrong"
        with self.assertRaisesRegex(ValueError,"CORE_SOURCE_DRIFT"):
            validate_knight_recipe(policy,changed)

    def test_source_rays_are_certified_and_cover_every_view_and_stratum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,policy=fixture(root)
            s=torch.load(root/manifest['subjects'][0]['shard_file'],weights_only=True)
            q,foreground=training_rays(s,policy,19,0,'cpu')
            values=torch.from_numpy(targets(s,q.reshape(-1,3).numpy())).reshape(32,4)
            from models.iris.v5.objective_tail_repair_v5 import soft_ray_minimum_v5
            m=soft_ray_minimum_v5(values,tau=policy['silhouette_softmin_tau'])
            self.assertTrue(bool((m[foreground]<=0).all()))
            self.assertTrue(bool((m[~foreground]>=0).all()))
            self.assertEqual(int(foreground.sum()),16)
            self.assertEqual(len(s['ray_court']['strata']),32)
            self.assertEqual({r['view'] for r in s['ray_court']['strata']},set(range(8)))
            pairs=straddle_pool(s,64,np.random.default_rng(7),1/512,.006)
            f=targets(s,pairs).reshape(-1,2)
            self.assertTrue(bool(np.all((f[:,0]>=0)!=(f[:,1]>=0))))
            self.assertTrue(bool(np.all(np.abs(pairs)<=1.)))
    def test_prediction_outside_teacher_range_still_receives_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,policy=fixture(root)
            trainer=SharedIrisTrainer(root,manifest,policy,{},root/"cp",root/"out","cpu")
            with torch.no_grad():
                trainer.model.surface_field.sdf_head.weight.zero_()
                trainer.model.surface_field.sdf_head.bias.fill_(2.)
            before=trainer.model.surface_field.sdf_head.bias.detach().clone()
            row=trainer.one_step()
            self.assertGreater(row["loss"],1.)
            self.assertGreater(row["gradient_norm"],0.)
            self.assertFalse(torch.equal(before,trainer.model.surface_field.sdf_head.bias))

    def test_real_optimizer_resume_matches_uninterrupted_all_eight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,policy=fixture(root)
            source={"tested_source":"same"}
            continuous=SharedIrisTrainer(root,manifest,policy,source,root/"full",root/"out_full","cpu")
            continuous.run()
            expected={k:v.clone() for k,v in continuous.model.state_dict().items()}
            interrupted=SharedIrisTrainer(root,manifest,policy,source,root/"resume",root/"out_resume","cpu")
            interrupted.run(stop_after=8)
            resumed=SharedIrisTrainer(root,manifest,policy,source,root/"resume",root/"out_resume","cpu")
            result=resumed.run()
            self.assertEqual(result["subject_exposures"],{f"fixture_{i}":2 for i in range(8)})
            self.assertEqual(result["shared_model_count"],1)
            self.assertFalse(result["fit8_pass_claimed"])
            self.assertEqual(continuous.trace,resumed.trace)
            for key,value in expected.items():
                self.assertTrue(torch.equal(value,resumed.model.state_dict()[key]),key)
            for drift in ("code","inputs","policy"):
                m,p,s=copy.deepcopy(manifest),copy.deepcopy(policy),copy.deepcopy(source)
                if drift=="code":s["tested_source"]="changed"
                if drift=="inputs":m["subjects"][0]["observation_binding"]="changed"
                if drift=="policy":p["learning_rate"]*=2
                with self.subTest(drift=drift),self.assertRaisesRegex(ValueError,"DRIFT"):
                    SharedIrisTrainer(root,m,p,s,root/"resume",root/"out_resume","cpu")

    def test_orphan_and_corruption_never_resume_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=AtomicCheckpointStore(root,"binding")
            payload={"step":0,"binding_sha256":"binding","model":torch.ones(2)}
            pointer=store.save(payload)
            torch.save({"step":999},root/"step_99999999_orphan.pt")
            self.assertEqual(store.load()["step"],0)
            (root/pointer["file"]).write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError,"HASH_DRIFT"):
                store.load()

    def test_thin_triangle_partial_pixel_coverage(self):
        # Tip intersects the unit pixel but does not contain its center.
        vertices=np.array([[-.9,-.9,0.],[-.89,-.9,0.],[-.895,-.7,0.]])
        camera={"origin":[0,0,0],"right":[1,0,0],"screen_up":[0,-1,0],
                "half_extent":1.,"resolution":16}
        self.assertTrue(project_mask(vertices,np.array([[0,1,2]]),camera).any())

    def test_bind_anchor_preserves_authored_nodes_and_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/"raw";raw.mkdir()
            doc={"asset":{"version":"2.0"},"nodes":[{"children":[1,2],"translation":[3,4,5]},
                 {},{}],"skins":[{"joints":[1,2]}],"animations":[{"name":"native"}]}
            path=raw/"source.gltf";path.write_text(json.dumps(doc));before=path.read_bytes()
            out=root/"derivative"/path.name;receipt=prepare_bind_rest_gltf(path,out)
            actual=json.loads(out.read_text())
            self.assertEqual(actual["nodes"],doc["nodes"])
            self.assertEqual(actual["skins"][0]["skeleton"],0)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(receipt["source_bind_parity_claimed"])
            doc["skins"][0]["skeleton"]=1;path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"NOT_ANCESTOR"):
                prepare_bind_rest_gltf(path,out)
if __name__=="__main__":
    unittest.main()
