import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from models.iris.v31.model_v31 import IRISConfigV31, IRISV31, DirectDinoFusionV31, alpha_patches_v31, admit_observations_v31
from models.tessa.v31.attention_v31 import rolling_sdpa
from models.tessa.v31.model_v31 import TESSAStructuralizerV31
from models.tessa.v31.incremental_v31 import IncrementalDecoderV31
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.training_runtime.semantic_batch_v1 import SemanticExample, physical_batches, execute_semantic_step
from models.training_runtime.durable_state_v1 import save_resume, load_resume, step_generator, REQUIRED_BINDINGS
from models.training_runtime.execution_policy_v1 import MeasuredExecution, select_execution
from models.training_runtime.telemetry_v1 import WorkerTelemetry

torch.set_num_threads(1)


def tiny_tessa(**kwargs):
    config = TESSAStructuralizerConfigV3(d_model=32, n_heads=4, decoder_layers=2, coordinate_bins=128,
                scene_memory_tokens=2, surface_memory_tokens=2, presentation_memory_tokens=2,
                conditioning_token_budget=6, memory_source_width=16, memory_refinement_layers=0,
                local_attention_window=8, query_chunk_size=3, gradient_checkpointing=True)
    return TESSAStructuralizerV31(scene_input_dim=5, surface_input_dim=7, presentation_input_dim=9, config=config, **kwargs)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def observations(self):
        final, layer2 = (torch.randn(1, 2, 3, 3, 384) for _ in range(2))
        alpha = torch.rand(1, 2, 3, 3, 196)
        cameras = torch.tensor([0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 0, -1, 2.]).view(1, 1, 13).repeat(1, 2, 1)
        return dict(final=final, layer2=layer2, alpha=alpha, cameras=cameras)

    def model(self, chunk=13):
        return IRISV31(IRISConfigV31(4, 8, ("response_energy",), ("relative_transport",), lift_chunk=chunk))

    def test_direct_final_and_exact_alpha_patch_layout(self):
        o = self.observations()
        fusion = DirectDinoFusionV31()
        t = fusion(o['final'], o['layer2'], o['alpha'])
        self.assertTrue(torch.equal(t[..., :384], o['final']))
        raster = torch.arange(518*518.).reshape(1, 1, 1, 518, 518)
        patches = alpha_patches_v31(raster)
        self.assertTrue(torch.equal(patches[0, 0, 1, 2], raster[0, 0, 0, 14:28, 28:42].flatten()))

    def test_observation_numeric_admission(self):
        o = self.observations()
        admit_observations_v31(**o)
        for index, value in ((12, 0), (3, 2), (0, float('nan'))):
            bad = dict(o, cameras=o['cameras'].clone())
            bad['cameras'][0, 0, index] = value
            with self.assertRaises(ValueError):
                admit_observations_v31(**bad)

    def test_lift_chunk_output_and_gradient_equivalence(self):
        a, b = self.model(7), self.model(64)
        b.load_state_dict(a.state_dict())
        o = self.observations()
        ca, cb = a.encode(**o), b.encode(**o)
        torch.testing.assert_close(ca, cb)
        ca.square().sum().backward(); cb.square().sum().backward()
        for pa, pb in zip(a.encoder.parameters(), b.encoder.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=2e-6)
        changed = a.encode(**dict(o, final=o['final']+1))
        self.assertFalse(torch.allclose(ca, changed))

    def test_relation_gradient_firewall_including_addresses(self):
        model = self.model()
        xyz = torch.rand(1, 6, 3, requires_grad=True)
        gm, gp = model.query_states(model.encode(**self.observations()), xyz)
        model.pair(gp[:, :3], gp[:, 3:], xyz[:, :3], xyz[:, 3:], torch.randn(1, 3, 13)).sum().backward()
        self.assertIsNone(xyz.grad)
        for group in (model.encoder, model.query, model.gm, model.gp, model.fusion):
            self.assertTrue(all(p.grad is None for p in group.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.relation.parameters()))

    def test_private_gm_gp_ownership(self):
        model = self.model()
        xyz = torch.rand(1, 3, 3)
        out = model.continuous(model.encode(**self.observations()), xyz, torch.randn(1, 3, 13))
        out['rgba'].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.gm.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.gp.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.query.parameters()))

    def test_rolling_attention_dense_formula_and_gradients(self):
        original = [torch.randn(2, 2, 19, 5, dtype=torch.float64) for _ in range(3)]
        pos = torch.arange(19)
        mask = (pos[None] <= pos[:, None]) & (pos[None] > pos[:, None]-8)
        inputs = [v.clone().requires_grad_() for v in original]
        expected = F.scaled_dot_product_attention(*inputs, attn_mask=mask)
        expected.square().sum().backward()
        for chunk in (1, 3, 6, 11, 19):
            other = [v.clone().requires_grad_() for v in original]
            actual = rolling_sdpa(*other, window=8, query_chunk_size=chunk)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            actual.square().sum().backward()
            for a, b in zip(inputs, other):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-11)

    def test_incremental_prefix_parity_beyond_window(self):
        model = tiny_tessa().eval()
        memory = torch.randn(1, 6, 32)
        state = IncrementalDecoderV31(model, memory, sequence_position_offset=5)
        ids = torch.randint(0, 128, (1, 21))
        with torch.no_grad():
            for i in range(ids.shape[1]):
                actual = state.step(int(ids[0, i]))
                expected = model.decode_tokens(memory=memory, input_ids=ids[:, :i+1], sequence_position_offset=5).logits[:, -1]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
        self.assertTrue(all(x.stable_k.shape[2] <= 8 for x in state.layers))
        with torch.no_grad():
            model.lm_head.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'MODEL_CHANGED'):
            state.step(1)

    def test_real_tessa_backward_with_checkpointing(self):
        model = tiny_tessa().train()
        out = model(scene_source=torch.randn(1, 3, 5), surface_source=torch.randn(1, 4, 7),
                    presentation_source=torch.randn(1, 5, 9), input_ids=torch.randint(0, 128, (1, 19)))
        out.logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.decoder[0].self_attn.qkv.weight.grad).all())
        self.assertTrue(model.memory.presentation.source_projection[0].weight.grad is not None)

    def test_ragged_eight_update_equivalence_with_clip_and_scheduler(self):
        a, b = torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)
        b.load_state_dict(a.state_dict())
        examples = [SemanticExample(str(i), i+2, (i+1)/8, i+2) for i in range(8)]
        data = {x.key: (torch.randn(x.cost, 3), torch.randn(x.cost, 1)) for x in examples}
        results = []
        for model, max_examples in ((a, 1), (b, 8)):
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.8)
            def sums(batch):
                return torch.stack([(model(data[x.key][0])-data[x.key][1]).square().sum() for x in batch])
            results.append(execute_semantic_step(examples, loss_sums=sums, optimizer=opt, scheduler=sched,
                            parameters=model.parameters(), max_padded_cost=1000, max_examples=max_examples, clip_norm=.1))
            self.assertEqual(sched.last_epoch, 1)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=6)

    def test_no_oversized_subject_or_duplicate_is_silently_dropped(self):
        xs = [SemanticExample('a', 2, 1, 2), SemanticExample('b', 20, 1, 20)]
        with self.assertRaisesRegex(ValueError, 'EXCEEDS_MEMORY_PLAN:b'):
            physical_batches(xs, max_padded_cost=10, max_examples=2)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            physical_batches(xs+xs, max_padded_cost=100, max_examples=8)

    def test_resume_same_next_adam_update_rng_sampler_and_terminal_bytes(self):
        bindings = {k:'binding-'+k for k in REQUIRED_BINDINGS}
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=.02)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
        def update():
            values = (random.random(), float(np.random.rand()), float(torch.rand(())))
            opt.zero_grad(set_to_none=True)
            loss = model(torch.randn(5, 3)).square().sum()*sum(values)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            return values
        with tempfile.TemporaryDirectory() as temp:
            local, durable = Path(temp)/'local', Path(temp)/'durable'
            save_resume(local_dir=local, durable_dir=durable, model=model, optimizer=opt, scheduler=sched,
                        global_step=0, sampler_state={'cursor':0}, bindings=bindings)
            update()
            save_resume(local_dir=local, durable_dir=durable, model=model, optimizer=opt, scheduler=sched,
                        global_step=1, sampler_state={'cursor':8}, bindings=bindings)
            values = update()
            expected = copy.deepcopy(model.state_dict())
            expected_opt = copy.deepcopy(opt.state_dict())
            result = load_resume(durable_dir=durable, model=model, optimizer=opt, scheduler=sched, bindings=bindings)
            self.assertEqual(result['sampler_state'], {'cursor':8})
            self.assertEqual(result['next_global_step'], 2)
            self.assertEqual(values, update())
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected[name]))
            for key, state in opt.state_dict()['state'].items():
                for name, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, expected_opt['state'][key][name]))
            bad = dict(bindings, corpus_sha256='different')
            with self.assertRaisesRegex(ValueError, 'BINDING'):
                load_resume(durable_dir=durable, model=model, optimizer=opt, scheduler=sched, bindings=bad)
            receipt = json.loads((durable/'LATEST.json').read_text())
            (durable/receipt['file']).unlink()
            with self.assertRaisesRegex(ValueError, 'BYTES_MISSING'):
                load_resume(durable_dir=durable, model=model, optimizer=opt, scheduler=sched, bindings=bindings)

    def test_absolute_sampler_stream_is_logging_independent(self):
        a = torch.rand(10, generator=step_generator(3, 17, 'training'))
        torch.rand(100)
        b = torch.rand(10, generator=step_generator(3, 17, 'training'))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, torch.rand(10, generator=step_generator(3, 17, 'eval'))))

    def test_total_eta_rejects_synthetic_and_changed_recipe(self):
        base = MeasuredExecution('base', 'r', 'c', 'm', 'A100', True, True, ('small', 'large'),
                                 2, 3, 4, 0, 60, 100, 100, 2, 2)
        compiled = replace(base, name='compiled', update_p95_seconds=1, cold_setup_seconds=200)
        selected, _ = select_execution([base, compiled], required_cost_strata=('small', 'large'), max_total_seconds=1000)
        self.assertEqual(selected.name, 'base')
        for bad in (replace(base, real_corpus=False), replace(base, update_equivalence_pass=False),
                    replace(base, covered_cost_strata=('small',)), replace(base, peak_reserved_bytes=95)):
            with self.assertRaisesRegex(ValueError, 'NO_QUALIFIED'):
                select_execution([bad], required_cost_strata=('small', 'large'), max_total_seconds=1000)
        with self.assertRaisesRegex(ValueError, 'CHANGE_SCIENCE'):
            select_execution([base, replace(compiled, updates=1)], required_cost_strata=(), max_total_seconds=1000)

    def test_heartbeat_during_blocked_phase_and_crash_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'events.jsonl'
            with self.assertRaisesRegex(RuntimeError, 'example'):
                with WorkerTelemetry(path, heartbeat_seconds=.01, stream=io.StringIO()) as log:
                    log.update(phase='EVALUATION', global_step=7)
                    time.sleep(.05)
                    raise RuntimeError('example')
            events = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertTrue(any(x['event']=='HEARTBEAT' and x['phase']=='EVALUATION' for x in events))
            self.assertIn('RuntimeError: example', events[-1]['traceback'])
            self.assertEqual(events[-1]['event'], 'CRASH')


if __name__ == '__main__':
    unittest.main()
