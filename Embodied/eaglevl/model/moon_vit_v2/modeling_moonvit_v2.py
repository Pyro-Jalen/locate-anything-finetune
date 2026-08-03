# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
#
# Extracted from moonshotai/Kimi-K3 for standalone MoonViT-V2 use.
# Licensed under the Kimi K3 License (see LICENSE in this repository).

"""MoonViT-V2: standalone native-resolution vision encoder from Kimi K3."""

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import activations

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh
    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available
from transformers.utils import logging as hf_logging

from .configuration_moonvit_v2 import MoonViTV2Config

logger = hf_logging.get_logger(__name__)

# Flash attention imports
try:
    if is_flash_attn_2_available():
        from flash_attn import flash_attn_varlen_func
    else:
        flash_attn_varlen_func = None
except Exception:
    flash_attn_varlen_func = None

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
except Exception:
    flash_attn4_varlen_func = None


def is_flash_attn_4_available() -> bool:
    return flash_attn4_varlen_func is not None


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Multi-head attention using flash attention 2.

    Aligned with moonshotai/MoonViT-SO-400M modeling_moonvit.py.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert (
        k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    ), "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """SDPA attention (same pattern as MoonViT-SO-400M)."""
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(
        q, k, v, attention_mask, dropout_p=0.0
    )
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight = attn_weight.masked_fill(
        ~attention_mask, torch.finfo(attn_weight.dtype).min
    )
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def multihead_attention_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Multi-head attention using FlashAttention-4 (flash_attn.cute). Same as LocAny MoonViT v1."""
    if flash_attn4_varlen_func is None:
        logger.warning_once(
            "flash-attn-4 is not available for MoonViT-V2; falling back to sdpa attention."
        )
        return sdpa_attention(
            q,
            k,
            v,
            q_cu_seqlens=q_cu_seqlens,
            k_cu_seqlens=k_cu_seqlens,
        )

    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert (
        k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    ), "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = int((q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item())
    max_seqlen_k = int((k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item())
    attn_out = flash_attn4_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=q_cu_seqlens,
        cu_seqlens_k=k_cu_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return attn_out.flatten(start_dim=-2)


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_4": multihead_attention_fa4,
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
@torch.compile(dynamic=True)
def get_rope_shape(org, interpolation_mode, shape):
    return (F.interpolate(
        org.permute((2, 0, 1)).unsqueeze(0),
        size=shape,
        mode=interpolation_mode,
    ).squeeze(0).permute((1, 2, 0)).flatten(end_dim=1))


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(
        -2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(
        -2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):

    def __init__(self,
                 height: int,
                 width: int,
                 num_frames: int,
                 dim: int,
                 interpolation_mode: str = 'bicubic') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        # Fixed 1D sincos temporal PE. Same as moonshotai/Kimi-K3:
        # registered with persistent=False, so it is NOT saved in the checkpoint
        # (absent from model.safetensors.index.json). Recomputed from the closed
        # form at init / after HF meta-load. Stored as float32 because it comes
        # from numpy float32 sincos (`.float()`), then cast to the running dtype
        # at use time (images use T=1 and never touch this buffer).
        self.register_buffer(
            "time_weight",
            torch.from_numpy(
                get_1d_sincos_pos_embed(self.dim, self.num_frames)
            ).float().unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def _materialize_time_weight(self) -> torch.Tensor:
        """Ensure time_weight matches the K3 sincos formula (HF meta-init safe)."""
        tw = getattr(self, "time_weight", None)
        need = (
            tw is None
            or tw.device.type == "meta"
            or tw.numel() == 0
            or (not torch.isfinite(tw).all())
        )
        if need:
            tw = torch.from_numpy(
                get_1d_sincos_pos_embed(self.dim, self.num_frames)
            ).float().unsqueeze(1)
            # Prefer in-place refresh when a real buffer already exists.
            if (
                hasattr(self, "time_weight")
                and self.time_weight is not None
                and self.time_weight.device.type != "meta"
                and self.time_weight.shape == tw.shape
            ):
                self.time_weight.copy_(tw.to(device=self.time_weight.device))
            else:
                self.register_buffer("time_weight", tw, persistent=False)
        return self.time_weight

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f't:{t} > self.num_frames:{self.num_frames}'
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                # Cast fp32 sincos buffer to activation dtype (bf16), same numerical
                # effect as K3 after `module.to(dtype=bfloat16)`.
                time_emb = self._materialize_time_weight()[0:t].to(
                    device=pos_emb_2d.device, dtype=pos_emb_2d.dtype)
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + time_emb

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):

    def __init__(self,
                 out_dim: int,
                 in_dim: int = 3,
                 patch_size: int | tuple[int, int] = (14, 14),
                 pos_emb_height: int = 14,
                 pos_emb_width: int = 14,
                 pos_emb_time: int = 4,
                 pos_emb_type: str = 'divided_fixed',
                 patch_embed_proj_bias: bool = True,
                 pos_emb_interpolation_mode: str = 'bicubic'):
        super().__init__()
        assert isinstance(
            patch_size,
            int | Sequence), f'Invalid patch_size type: {type(patch_size)}'
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (len(patch_size) == 2
                ), f'Expected patch_size to be a tuple of 2, got {patch_size}'
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim,
                              out_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=patch_embed_proj_bias)

        if pos_emb_type == 'divided_fixed':
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode)
        else:
            raise NotImplementedError(
                f'Not support pos_emb_type: {pos_emb_type}')

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self,
                 dim: int,
                 max_height: int,
                 max_width: int,
                 theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, 'dim must be divisible by 4'
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f'dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}'

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (torch.arange(0, self.dim,
                                  4)[:(self.dim // 4)].float().to(device)
                     )  # C/4
        freqs = 1.0 / (self.theta_base**(dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1),
             y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, 'freqs_cis'):
            self.register_buffer('freqs_cis',
                                 self._precompute_freqs_cis(device),
                                 persistent=False)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width
                   for t, h, w in shapes), (
                       shapes,
                       self.max_height,
                       self.max_width,
                   )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: int | None = None,
        norm_type: str = 'layernorm',
        mlp_type: str = 'mlp2',
        *,
        attn_implementation: str = 'flash_attention_2',
        activation=F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = True,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        if norm_type == "layernorm":
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.norm0 = nn.RMSNorm(hidden_dim)
            self.norm1 = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

        if mlp_type == "mlp2":
            self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim],
                            activation,
                            bias=linear_bias)
        else:
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")

        self.wqkv = nn.Linear(hidden_dim,
                              self.qkv_hidden_size * 3,
                              bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        # Same dispatch as MoonViT-SO-400M; fall back if transformers picks an unknown impl.
        attn_impl = self.attn_implementation
        if attn_impl not in VL_VISION_ATTENTION_FUNCTIONS:
            attn_impl = "sdpa" if "sdpa" in VL_VISION_ATTENTION_FUNCTIONS else "eager"
        attn_func = VL_VISION_ATTENTION_FUNCTIONS[attn_impl]
        attn_out = attn_func(
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens,
                                                 max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 block_cfg: dict,
                 use_deterministic_attn: bool = False) -> None:
        super().__init__()
        self.use_deterministic_attn = use_deterministic_attn

        qkv_hidden_size = block_cfg['hidden_dim'] if block_cfg.get(
            'qkv_hidden_size') is None else block_cfg['qkv_hidden_size']
        self.rope_2d = Rope2DPosEmbRepeated(
            qkv_hidden_size // block_cfg['num_heads'], 512, 512)
        self.blocks = nn.ModuleList([
            MoonViTEncoderLayer(
                **block_cfg,
                use_deterministic_attn=self.use_deterministic_attn)
            for _ in range(num_layers)
        ])
        norm_type = block_cfg.get('norm_type', 'layernorm')
        if norm_type == "layernorm":
            self.final_layernorm = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.final_layernorm = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device)

        lengths = torch.cat((
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        ))

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0,
                                                             dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states,
                                  cu_seqlens,
                                  max_seqlen,
                                  rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
        x: torch.Tensor,
        grid_thws: torch.Tensor,
        merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum:pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width,
                                kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1,
                                            3, 2, 4, 5).contiguous().mean(
                                                dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width,
                                       kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViTV2PretrainedModel(PreTrainedModel):
    config_class = MoonViTV2Config
    model_type = 'moonvit_v2'
    _no_split_modules = ['MoonViTEncoderLayer']
    _supports_flash_attn = True
    _supports_flash_attn_2 = True  # compat with older transformers
    _supports_sdpa = True

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        # HF does not know flash_attention_4; dispatch via VL_VISION_ATTENTION_FUNCTIONS.
        if getattr(config, "_attn_implementation", None) == "flash_attention_4":
            return config
        return super()._autoset_attn_implementation(config, *args, **kwargs)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):
        if attn_implementation == "flash_attention_4":
            return "flash_attention_4"
        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=getattr(config, 'patch_embed_proj_bias',
                                          True),
            pos_emb_interpolation_mode=getattr(
                config, 'pos_emb_interpolation_mode', 'bicubic'),
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                'num_heads': config.num_attention_heads,
                'hidden_dim': config.hidden_size,
                'qkv_hidden_size': getattr(config, 'qkv_hidden_size', None),
                'mlp_dim': config.intermediate_size,
                'norm_type': getattr(config, 'norm_type', 'layernorm'),
                'mlp_type': getattr(config, 'mlp_type', 'mlp2'),
                'activation': PytorchGELUTanh(),
                'attn_bias': getattr(config, 'attn_bias', True),
                'linear_bias': getattr(config, 'linear_bias', True),
                'attn_implementation': getattr(config, '_attn_implementation', None) or getattr(config, 'attn_implementation', 'eager'),
            },
            use_deterministic_attn=getattr(self, 'use_deterministic_attn',
                                           False))
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # HF low_cpu_mem / meta init can leave non-persistent buffers uninitialized;
        # restore the closed-form sincos temporal PE (same as Kimi-K3 __init__).
        pos_emb = getattr(getattr(model, "patch_embed", None), "pos_emb", None)
        if pos_emb is not None and hasattr(pos_emb, "_materialize_time_weight"):
            pos_emb._materialize_time_weight()
        return model

    def forward(self, pixel_values: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        # grid_thws = grid_thws.to('cpu')
        assert grid_thws.ndim == 2, f'grid_thws should be 2D, got {grid_thws.ndim}'
        assert grid_thws.size(1) == 3, f'No support for thw: {grid_thws}'
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == 'sd2_tpool':  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(
                hidden_states,
                grid_thws,
                merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f'Not support {self.merge_type}')

        return hidden_states

