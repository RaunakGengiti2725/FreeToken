"""Numerical checks for the built-in activation and norm fallback paths."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


@pytest.fixture(autouse=True)
def _use_triton_fallbacks(monkeypatch):
    from freetoken.kernel import backend

    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)
    monkeypatch.setattr(backend, "is_sgl_kernel_installed", lambda: False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_activation_fallback_matches_torch(dtype, activation):
    from freetoken.layers.activation import gated_act_and_mul

    torch.manual_seed(0)
    x = torch.randn((7, 256), device="cuda", dtype=dtype)
    out = torch.full((7, 128), float("nan"), device="cuda", dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        expected = torch.nn.functional.silu(gate) * up
    else:
        approx = "tanh" if activation == "gelu_tanh" else "none"
        expected = torch.nn.functional.gelu(gate, approximate=approx) * up

    gated_act_and_mul(activation, x, out)

    # CUDA intentionally uses approximate math (notably tanh.approx.f32),
    # while ROCm uses libdevice; allow for both against the Torch reference.
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-3)
    torch.testing.assert_close(out, expected.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["rmsnorm", "gemma"])
def test_norm_fallback_matches_torch(dtype, kind):
    from freetoken.layers.norm import GemmaRMSNorm, RMSNorm

    torch.manual_seed(0)
    norm = (RMSNorm if kind == "rmsnorm" else GemmaRMSNorm)(128, 1e-6)
    norm.weight = torch.randn(128, device="cuda", dtype=dtype)
    x = torch.randn((7, 128), device="cuda", dtype=dtype)
    xf = x.float()
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    expected *= norm.weight.float()

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-3)
    torch.testing.assert_close(norm.forward(x), expected.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("mask", ["prefix", "allowed_ids"])
def test_sampling_softmax_fallback_with_masked_logits(mask):
    from freetoken.kernel.triton.sampling import softmax

    torch.manual_seed(0)
    logits = torch.randn((3, 151936), device="cuda") * 4
    if mask == "prefix":
        # Long enough that some column chunk starts with an all -inf block.
        logits[:2, :16384] = float("-inf")
    else:
        allowed = torch.tensor([5, 70000, 151000], device="cuda")
        logits[:2] = torch.full_like(logits[:2], float("-inf")).index_copy(
            1, allowed, logits[:2, allowed])
    temperature = torch.tensor([1.0, 0.7, 1.3], device="cuda")
    expected = torch.softmax(logits / temperature[:, None], dim=-1)

    torch.testing.assert_close(softmax(logits, temperature), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("rows, calls", [(4, 4000), (256, 200)])
def test_sampling_min_p_draw_matches_torch(rows, calls):
    """min_p_sampling_from_probs draws only tokens with at least min_p times the row's top probability, in proportion to their probability, over many seeds; 4 rows split each row over several CTAs, 256 leave one per row."""
    from freetoken.kernel.triton.sampling import min_p_sampling_from_probs

    torch.manual_seed(0)
    distinct, vocab = 4, 151936
    probs = torch.softmax(torch.randn((distinct, vocab), device="cuda") * 4, dim=-1)
    min_p = torch.tensor([0.02, 0.1, 0.5, 1.0], device="cuda")
    expected = torch.where(probs >= min_p[:, None] * probs.amax(dim=1, keepdim=True), probs, 0.0)
    expected /= expected.sum(dim=1, keepdim=True)

    copies = rows // distinct
    batch_probs, batch_min_p = probs.repeat(copies, 1), min_p.repeat(copies)
    counts = torch.zeros_like(probs)
    ones = torch.ones(distinct, copies, device="cuda")
    for seed in range(calls):
        tokens = min_p_sampling_from_probs(batch_probs, batch_min_p, seed=seed).long().view(copies, distinct)
        counts.scatter_add_(1, tokens.T, ones)

    n = copies * calls
    assert not counts[expected == 0].any()
    # min_p = 1 leaves the top token alone
    assert counts[3, probs[3].argmax()] == n
    # 5 sigma of a frequency over n draws
    assert (counts / n - expected).abs().max() < 5 * (0.25 / n) ** 0.5
