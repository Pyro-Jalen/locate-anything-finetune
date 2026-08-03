# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
#
# Extracted from moonshotai/Kimi-K3 for standalone MoonViT-V2 use.
# Licensed under the Kimi K3 License (see LICENSE in this repository).

from transformers.configuration_utils import PretrainedConfig


class MoonViTV2Config(PretrainedConfig):
    r"""
    Configuration class for MoonViT-V2, the native-resolution vision encoder of Kimi K3.

    MoonViT-V2 is a ~0.4B-parameter Vision Transformer (27 layers, 12 heads, patch size 14)
    trained from scratch jointly with the language model (not initialized from SigLIP).
    """

    model_type = "moonvit_v2"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        pos_emb_interpolation_mode: str = "bilinear",
        num_attention_heads: int = 12,
        num_hidden_layers: int = 27,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        qkv_hidden_size: int = 1536,
        merge_kernel_size: tuple[int, int] = (2, 2),
        merge_type: str = "sd2_tpool",
        norm_type: str = "rmsnorm",
        mlp_type: str = "mlp2",
        attn_bias: bool = False,
        linear_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.qkv_hidden_size = qkv_hidden_size
        self.merge_kernel_size = list(merge_kernel_size) if not isinstance(merge_kernel_size, list) else merge_kernel_size
        self.merge_type = merge_type
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.attn_bias = attn_bias
        self.linear_bias = linear_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.activation_func = activation_func
