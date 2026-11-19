from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)
_FOLD_SAID = False  # the router fold's one line a process


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self._fuse_shared = self._shared_expert_fusable(config)
        self._router_weight: torch.Tensor | None | bool = False  # False until the first decode step decides
        self._router_off: str | None = None

    def _shared_expert_fusable(self, config: ModelConfig) -> bool:
        """Whether decode can run the shared expert in the fused NVFP4 kernels: both projections
        resident under the triton kernel, on one rank, where those kernels' own gate holds."""
        from freetoken.layers.quantization.linear.nvfp4 import TritonNvfp4LinearKernel

        gate_up, down = self.shared_expert.gate_up_proj, self.shared_expert.down_proj
        if down._tp_size > 1 or not all(isinstance(proj.quant_method.kernel, TritonNvfp4LinearKernel) for proj in (gate_up, down)):
            return False
        from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_fusable

        return shared_expert_fusable(config.hidden_size, config.shared_expert_intermediate_size)

    def _fuses_step(self, hidden_states: torch.Tensor) -> bool:
        if not self._fuse_shared or hidden_states.dtype != torch.bfloat16 or get_global_ctx().batch.is_prefill:
            return False
        from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_fuses_rows

        down = self.shared_expert.down_proj
        return shared_expert_fuses_rows(hidden_states.shape[0], down.out_features, down.in_features, hidden_states.device)

    def _router_weight_off(self) -> str | None:
        """Why the router's weight cannot go into its top-k kernel, or None when it can: a bf16 [E, H]
        torch linear without a bias on the GPU."""
        from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

        layer = self.gate
        w = layer.weight
        if layer.bias is not None:
            return "the router has a bias"
        if not isinstance(layer.quant_method.kernel, TorchLinearKernel):
            return f"the router runs {type(layer.quant_method.kernel).__name__}, not the torch linear"
        if not w.is_cuda:
            return f"the router's weight is on {w.device}"
        if w.dtype != torch.bfloat16:
            return f"the router's weight is {w.dtype}"
        if w.dim() != 2 or w.stride(1) != 1:
            return f"the router's weight is not a row-major matrix (shape {tuple(w.shape)}, strides {w.stride()})"
        return None

    def _router_weight_in_kernel(self) -> torch.Tensor | None:
        """The router's weight when its top-k kernel can take the logits too. Decided once, at the
        first decode step, when the weights are final."""
        if self._router_weight is False:
            self._router_off = self._router_weight_off()
            self._router_weight = self.gate.weight if self._router_off is None else None
        return self._router_weight

    def _fold_off(self, hidden_states: torch.Tensor, w: torch.Tensor | None) -> str | None:
        """Why no decode step folds the router, or None when decode-sized steps do; the step's own row
        count is left out, as CUDA-graph capture also runs steps past the fold's cap."""
        if w is None:
            return self._router_off
        if hidden_states.dtype != torch.bfloat16:
            return f"the hidden states are {hidden_states.dtype}"
        if hidden_states.stride(1) != 1:
            return "the hidden states' rows are not contiguous"
        from freetoken.kernel.triton.moe_router import router_gemv_topk_folds

        if not router_gemv_topk_folds(1, w.shape[0], w.shape[1], self.experts.top_k):
            return f"{w.shape[0]} experts of hidden size {w.shape[1]} and top-{self.experts.top_k} are not the fold's shapes"
        return None

    def _routes_in_kernel(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(topk weights, topk ids) of a decode step from one launch that also computes the router's
        logits, or None where the linear and the router stay two launches. The logits are summed in
        that kernel's order, not cuBLAS's, so a near-tie may route to the other expert."""
        global _FOLD_SAID
        if get_global_ctx().batch.is_prefill:
            return None
        w = self._router_weight_in_kernel()
        if not _FOLD_SAID:
            _FOLD_SAID = True
            off = self._fold_off(hidden_states, w)
            if off is None:
                logger.info_rank0("router fold on decode: on (the router's linear and top-k in one launch for decode-sized steps)")
            else:
                logger.info_rank0(f"router fold on decode: off ({off})")
        if w is None or hidden_states.dtype != torch.bfloat16 or hidden_states.stride(1) != 1:
            return None
        from freetoken.kernel.triton.moe_router import router_gemv_topk_folds, router_gemv_topk_softmax

        experts = self.experts
        if not router_gemv_topk_folds(hidden_states.shape[0], w.shape[0], w.shape[1], experts.top_k):
            return None
        return router_gemv_topk_softmax(hidden_states, w, experts.top_k, experts.renormalize)

    def _routed(self, hidden_states, router_logits, routes, sum_routes: bool = True) -> torch.Tensor:
        if routes is None:
            return self.experts.forward(hidden_states=hidden_states, router_logits=router_logits, sum_routes=sum_routes)
        return self.experts.routed_forward(hidden_states, *routes, sum_routes=sum_routes)

    def _forward_fused(self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None, routes) -> torch.Tensor:
        """A decode step with the shared expert's elementwise kernels inside its two projections,
        bit for bit the unfused statements of ``forward``."""
        from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_down_add, shared_expert_gate_up

        gate_up, down = self.shared_expert.gate_up_proj, self.shared_expert.down_proj
        # as in forward, everything that reads hidden_states runs before the routed experts
        act = shared_expert_gate_up(hidden_states, gate_up.weight, gate_up.weight_scale, gate_up.weight_global)
        gate = self.shared_expert_gate.forward(hidden_states).view(-1)
        # each route's output where the experts' decode kernel has it: the down kernel then sums the routes too
        routed = self._routed(hidden_states, router_logits, routes, sum_routes=False)
        return shared_expert_down_add(act, down.weight, down.weight_scale, down.weight_global, gate, routed)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        routes = self._routes_in_kernel(hidden_states)
        router_logits = self.gate.forward(hidden_states) if routes is None else None
        if self._fuses_step(hidden_states):
            return self._forward_fused(hidden_states, router_logits, routes).view(num_tokens, hidden_dim)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self._routed(hidden_states, router_logits, routes)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
