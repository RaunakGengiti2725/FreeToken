from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5Attention(BaseOP):
    """Gated full attention: per-head output gate, q/k RMSNorm, partial NeoX rope.

        query, gate = chunk(q_proj(x).view(.., num_q, head_dim*2), 2, -1)
        q = qnorm(query); k = knorm(k_proj(x)); v = v_proj(x)
        q, k = rope(q, k)                       # first rotary_dim dims
        attn = paged_attention(q, k, v)
        out = o_proj(attn * sigmoid(gate))

    TP note: uses replicated linears (tp=1 correctness milestone); swap to
    column/row-parallel for tensor parallelism later.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        # Fused q/k/v projection (one GEMM instead of three); q half is 2x for the
        # output gate. Split sizes: [num_q*head_dim*2, num_kv*head_dim, num_kv*head_dim].
        self._qkv_split = [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        self.q_norm = GemmaPlusOneRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
            mrope_section=(
                tuple(config.rotary_config.mrope_section)
                if config.rotary_config.mrope_section is not None
                else None
            ),
            mrope_layout=config.rotary_config.mrope_layout,
        )
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        from freetoken.kernel.triton.attn_gate import attn_gate_fusable

        self._fuse_gate = attn_gate_fusable()

    def _project(self, x: torch.Tensor, gate_in_place: bool = False, gate_view: bool = False):
        """Returns (q, k, v, gate): q [N, num_q, head_dim] post qk-norm+rope,
        k [N, num_kv*head_dim] post norm+rope, v [N, num_kv*head_dim], gate [N, num_q*head_dim].
        ``gate_in_place`` returns the q|gate projection [N, num_q, head_dim*2] as the gate, uncopied;
        ``gate_view`` its gate half [N, num_q, head_dim], a strided view of the projection."""
        positions = get_global_ctx().batch.get_attn_positions()
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()  # [N, num_q, head_dim]
        if gate_in_place:
            gate = qg
        elif gate_view:
            gate = qg[..., self.head_dim :]
        else:
            gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        v = v.contiguous()  # split view has the qkv row stride; the KV store needs contiguous
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self.num_q, self.head_dim), k, v, gate

    def _combine(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # the sigmoid of a strided gate view comes out dense: the reshape is free either way
        gated = attn_out.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate).reshape(-1, self.qo_attn_dim)
        return self.o_proj.forward(gated)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        # a bf16 decode step gates the output in one kernel, bit for bit what _combine gives
        fuse = self._fuse_gate and x.dtype == torch.bfloat16 and not ctx.batch.is_prefill
        # a prefill's gate is read by the sigmoid where it lies: no [N, num_q*head_dim] copy of it
        q, k, v, gate = self._project(x, gate_in_place=fuse, gate_view=ctx.batch.is_prefill)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        if fuse and o.dtype == torch.bfloat16:
            from freetoken.kernel.triton.attn_gate import attn_gate_mul

            return self.o_proj.forward(attn_gate_mul(o.reshape(-1, self.qo_attn_dim), gate))
        if fuse:
            gate = gate[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        return self._combine(o, gate)


__all__ = ["Qwen3_5Attention"]
