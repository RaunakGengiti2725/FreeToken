"""The GDN verify kernels: the fused recurrence's verify mode stashes each row's inputs, and gdn_state_replay over the first n rows reproduces decode mode bitwise for every n.
Also the recurrence kernel with the gated output norm in it, against the norm kernel.

The layer-level verify (outputs, live state, commit) is checked against the reference in tests/layers/test_gdn.py.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NK, NV, HD = 2, 4, 32
STEPS = 5  # verify width: k drafts + 1


@cuda
# the second geometry launches more layer x row x head programs than one grid axis holds (48 * 29 * 48 > 65535)
@pytest.mark.parametrize("layers,bs,h,hv,dim", [(2, 3, NK, NV, HD), (48, 29, 16, 48, 16)], ids=["small", "past-grid-axis-limit"])
def test_kernel_verify_stash_replay_matches_decode_bitwise(layers, bs, h, hv, dim):
    """The fused kernel's verify mode (disable_state_update + input stash) followed by
    gdn_state_replay over the first n tokens must reproduce decode mode BITWISE on
    identical per-token inputs, for every n, across layers and with verify rows that
    differ from the live slots; the verify itself must leave the live slab untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import gdn_state_replay

    torch.manual_seed(0)
    dev = torch.device("cuda")
    steps, kd, vd = STEPS, dim, dim
    slots = [(i + 2) % bs for i in range(bs)]  # verify row i commits into live slot slots[i]

    def r(*s, dt=torch.bfloat16, sc=0.5):
        return (torch.randn(*s, device=dev) * sc).to(dt)

    q, k, v = r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, hv, vd)
    a, b = r(layers, bs * steps, hv), r(layers, bs * steps, hv)
    a_log, dt_bias = r(layers, hv, dt=torch.float32, sc=0.1), r(layers, hv, dt=torch.float32, sc=0.1)
    state0 = torch.randn(layers, bs + 1, hv, kd, vd, device=dev) * 0.1
    slot_t = torch.tensor(slots, dtype=torch.int32, device=dev)
    common = dict(softplus_beta=1.0, softplus_threshold=20.0, scale=kd ** -0.5,
                  use_qk_l2norm_in_kernel=True, initial_state_indices=slot_t)

    per_step = []  # per_step[t][l] = live slab after tokens 0..t
    st_decode = state0.clone()
    for t in range(steps):
        rows = torch.arange(bs, device=dev) * steps + t
        for l in range(layers):
            fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l][rows], q=q[l][:, rows], k=k[l][:, rows],
                  v=v[l][:, rows], b=b[l][rows], initial_state_source=st_decode[l],
                  cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev), **common)
        per_step.append(st_decode.clone())

    stash = tuple(torch.zeros(layers, bs, steps, *shape, device=dev)
                  for shape in ((h, kd), (hv, vd), (hv,), (hv,)))
    st_verify = state0.clone()
    for l in range(layers):
        fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l], q=q[l], k=k[l], v=v[l], b=b[l],
              initial_state_source=st_verify[l],
              cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=dev) * steps,
              disable_state_update=True, input_stash=tuple(t[l] for t in stash), **common)
    assert torch.equal(st_verify, state0)  # live slab untouched

    for n in range(1, steps + 1):
        st = state0.clone()
        meta = torch.tensor([list(range(bs)), slots, [n] * bs], dtype=torch.int32, device=dev)
        gdn_state_replay(st, stash, meta)
        assert torch.equal(st[:, slots], per_step[n - 1][:, slots]), f"n={n}"
        assert torch.equal(st[:, bs], state0[:, bs])  # the slot no row commits to


def _assert_same_bits(out, ref, what):
    """Equal bits of a bf16 or fp32 tensor; a NaN matches a NaN of any payload."""
    assert out.shape == ref.shape and out.dtype == ref.dtype, f"{what}: {out.shape} {out.dtype} against {ref.shape} {ref.dtype}"
    as_int = torch.int16 if out.dtype == torch.bfloat16 else torch.int32
    differ = (out.contiguous().view(as_int) != ref.contiguous().view(as_int)) & ~(out.isnan() & ref.isnan())
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(differ.sum())} of {out.numel()} elements differ, first at {first}: "
            f"fused {float(out[first])!r}, unfused {float(ref[first])!r}"
        )


@cuda
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("bs,steps", [(1, 1), (4, 1), (2, 4)])
@pytest.mark.parametrize(
    "h,hv,dim", [(16, 32, 128), (16, 48, 128), (1, 5, 96), (1, 5, 32)],
    ids=["qwen3.6-35b", "qwen3.8-flash-next", "partial-block", "one-program-a-head"],
)
def test_kernel_fused_output_norm_is_the_norm_kernel_bitwise(h, hv, dim, bs, steps, activation):
    """The gated RMS norm inside the recurrence kernel against the norm kernel over that kernel's output, with one
    token a sequence (decode) and several (verify). 1/rms is compared in fp32, where another order of summing a row's
    squares shows at once. Value heads cycle through the regimes below; the gate is read in place, strided."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import layernorm_gated

    dev = torch.device("cuda")
    tokens = bs * steps
    if layernorm_gated.gated_norm_row_tile(tokens * hv, dim, dev) is None:
        pytest.skip("the output norm is not fused here: the norm kernel takes another tile for these rows, or the GPU is not NVIDIA")
    gen = torch.Generator(device="cuda").manual_seed(dim + 131 * tokens)

    def r(*shape, sc=0.5, dt=torch.bfloat16):
        return (torch.randn(*shape, device=dev, generator=gen) * sc).to(dt)

    q, k = r(1, tokens, h, dim), r(1, tokens, h, dim)
    v = r(1, tokens, hv, dim).float()
    # by head: decode magnitudes on a live state, a wide spread, squares that overflow, squares that underflow, a NaN
    regime = torch.arange(hv, device=dev) % 5
    spread = torch.ldexp(torch.ones(tokens, hv, dim, device=dev), torch.randint(-10, 11, (tokens, hv, dim), device=dev, generator=gen))
    v[0] = torch.where((regime == 1)[None, :, None], v[0] * spread, v[0])
    v[0] = torch.where((regime == 2)[None, :, None], v[0] * 1e28, v[0])
    v[0] = torch.where((regime == 3)[None, :, None], v[0] * 1e-25, v[0])
    v[0, :, regime == 4, dim // 3] = float("nan")
    # head 1 shows the order of its sum of squares: column 0 is 2^10, every other square is under half a step of its
    # square and their sum is over a step. q = k on its key head makes a first token's output this v times a scalar.
    built = 1
    small = torch.randint(160, 320, (tokens, dim), device=dev, generator=gen).float() / 128 * 2.0**-4
    v[0, :, built] = small * (torch.randint(0, 2, (tokens, dim), device=dev, generator=gen) * 2 - 1)
    v[0, :, built, 0] = 2.0**10
    q[0, :, built // (hv // h)] = k[0, :, built // (hv // h)]
    v = v.to(torch.bfloat16)
    a, b = r(tokens, hv), r(tokens, hv)
    a_log, dt_bias = r(hv, sc=0.1, dt=torch.float32), r(hv, sc=0.1, dt=torch.float32)
    # a live state under the decode-magnitude heads only: the other heads' outputs are their v times a scalar
    state0 = torch.randn(bs + 1, hv, dim, dim, device=dev, generator=gen) * 0.1 * (regime == 0)[None, :, None, None]
    slots = torch.tensor([(i + 1) % bs for i in range(bs)], dtype=torch.int32, device=dev)
    z_buf = r(tokens, 24 + hv * dim, sc=3.0)
    z_buf[:, 24::7] *= 40
    z_buf[:, 25::97] = float("inf")
    z_buf[:, 26::97] = float("-inf")
    z = z_buf[:, 24:].reshape(tokens, hv, dim)
    weight = (torch.randn(dim, device=dev, generator=gen) * 0.5 + 1).to(torch.bfloat16)
    eps = 1e-6
    common = dict(A_log=a_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0, q=q, k=k, v=v, b=b,
                  initial_state_indices=slots, scale=dim ** -0.5, use_qk_l2norm_in_kernel=True,
                  cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev) * steps)

    state_ref = state0.clone()
    o_ref = fused(initial_state_source=state_ref, **common)
    x = o_ref[0].reshape(-1, dim)
    y_ref, _, rstd_ref = layernorm_gated._layer_norm_fwd(
        x, weight, None, eps, z=z.reshape(-1, dim).contiguous(), norm_before_gate=True, is_rms_norm=True, activation=activation,
    )

    # the built head tells one order of summing from another on the first token of every sequence, and the other
    # heads reach the edges they are named for
    first = torch.arange(bs, device=dev) * steps
    squares = x.float().reshape(tokens, hv, dim)[first, built] ** 2
    fwd, rev = torch.zeros_like(squares[..., 0]), torch.zeros_like(squares[..., 0])
    for c in range(dim):
        fwd, rev = fwd + squares[..., c], rev + squares[..., dim - 1 - c]
    assert bool((fwd != rev).all()), "a built row's sum of squares does not depend on its order"
    by_head = rstd_ref.view(tokens, hv)
    assert bool((by_head[:, regime == 2] == 0).all()), "the large heads' squares do not overflow"
    assert bool((by_head[:, regime == 3] > 900).all()), "the small heads' squares do not underflow"
    assert bool(by_head[:, regime == 4].isnan().all())

    for launch in range(2):  # a launch must leave the arrival counters zeroed for the next
        state = state0.clone()
        o, normed, rstd = fused(initial_state_source=state, gated_norm=(z, weight, eps, activation), **common)
        what = f"{bs} sequences of {steps}, {hv} heads of {dim}, {activation}, launch {launch}"
        _assert_same_bits(o[0].reshape(-1, dim), x, f"the recurrence's output, {what}")
        _assert_same_bits(state.reshape(-1, dim), state_ref.reshape(-1, dim), f"the recurrent state, {what}")
        _assert_same_bits(rstd.view(-1, hv), rstd_ref.view(-1, hv), f"1/rms, {what}")
        _assert_same_bits(normed.reshape(-1, dim), y_ref, f"the normed output, {what}")
