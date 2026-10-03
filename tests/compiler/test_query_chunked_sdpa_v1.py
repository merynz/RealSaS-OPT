import pytest

torch = pytest.importorskip('torch')
from torch.nn.attention import sdpa_kernel, SDPBackend
from tools.inference.query_chunked_sdpa_v1 import query_chunked_cuda_bf16_sdpa


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA BF16 math backend')
@pytest.mark.parametrize('mask_kind', ['none', 'padding', 'full'])
def test_chunked_attention_matches_complete_math_attention(mask_kind):
    torch.manual_seed(19)
    q = torch.randn(1, 2, 131, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 2, 137, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = None
    if mask_kind != 'none':
        mask = torch.ones(1, 1, 1 if mask_kind == 'padding' else 131, 137,
                          device='cuda', dtype=torch.bool)
        mask[..., 97:] = False
        if mask_kind == 'full':
            mask[..., 65:, 70:] = False
    original = torch.nn.functional.scaled_dot_product_attention
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        reference = original(q, k, v, attn_mask=mask)
    with torch.inference_mode(), query_chunked_cuda_bf16_sdpa(37) as telemetry:
        actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, reference)
    assert telemetry == {'intercepted_calls': 1, 'query_blocks': 4, 'maximum_query_block': 37}
    assert torch.nn.functional.scaled_dot_product_attention is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_training_and_causal_attention_fail_closed_and_restore_function():
    q = torch.ones(1, 1, 4, 8, device='cuda', dtype=torch.bfloat16)
    original = torch.nn.functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match='REQUIRES_NONCAUSAL'):
        with query_chunked_cuda_bf16_sdpa(2):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
    with torch.inference_mode(), query_chunked_cuda_bf16_sdpa(2):
        with pytest.raises(RuntimeError, match='REQUIRES_NONCAUSAL'):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert torch.nn.functional.scaled_dot_product_attention is original
