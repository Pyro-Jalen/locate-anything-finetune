# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
#
# Extracted from moonshotai/Kimi-K3 for standalone MoonViT-V2 use.
# Licensed under the Kimi K3 License (see LICENSE in this repository).

"""Image / video processor for MoonViT-V2 (NaViT-style native resolution)."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.utils import TensorType


def _navit_resize(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
):
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor
    token_height = (new_h + pad_height) // factor
    token_width = (new_w + pad_width) // factor
    return {
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "num_tokens": token_height * token_width,
    }


def _patchify(pixel_values: np.ndarray, patch_size: int) -> dict:
    """pixel_values: (t, h, w, c) -> patches + grid_thw."""
    T, H, W, C = pixel_values.shape
    assert C == 3
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size], dtype=np.int64)
    return {"pixel_values": patches, "grid_thw": grid_thw}


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


class MoonViTV2ImageProcessor(BaseImageProcessor):
    model_type = "moonvit_v2"

    def __init__(
        self,
        patch_size: int = 14,
        merge_kernel_size: int = 2,
        in_patch_limit: int = 65536,
        patch_limit_on_one_side: int = 512,
        max_num_frames: int = 4,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_kernel_size = merge_kernel_size
        self.in_patch_limit = in_patch_limit
        self.patch_limit_on_one_side = patch_limit_on_one_side
        # Matches MoonViT-V2 init_pos_emb_time / temporal pos-emb capacity.
        self.max_num_frames = max_num_frames
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)

    def _resize_config(self, image: Image.Image) -> dict:
        w, h = image.size
        return _navit_resize(
            w,
            h,
            self.patch_size,
            self.merge_kernel_size,
            self.in_patch_limit,
            self.patch_limit_on_one_side,
        )

    def _normalize_frame(self, image: Image.Image, cfg: dict) -> np.ndarray:
        image = image.resize(
            (cfg["new_width"], cfg["new_height"]), resample=Image.Resampling.BICUBIC
        )
        arr = np.asarray(image)
        if cfg["pad_height"] or cfg["pad_width"]:
            arr = np.pad(
                arr,
                ((0, cfg["pad_height"]), (0, cfg["pad_width"]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        mean = np.array(self.image_mean, dtype=np.float32)
        std_inv = 1.0 / np.array(self.image_std, dtype=np.float32)
        return (arr.astype(np.float32) / 255.0 - mean) * std_inv

    def _preprocess_one(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        image = image.convert("RGB")
        cfg = self._resize_config(image)
        arr = self._normalize_frame(image, cfg)
        packed = _patchify(np.expand_dims(arr, axis=0), self.patch_size)
        return packed["pixel_values"], packed["grid_thw"]

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """Preprocess one or more **images** (each becomes an independent sample with T=1)."""
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be PIL.Image.Image, numpy.ndarray, or torch.Tensor."
            )

        pixel_values, grid_thws = [], []
        for image in images:
            patches, grid_thw = self._preprocess_one(_as_pil(image))
            pixel_values.append(torch.from_numpy(patches))
            grid_thws.append(torch.from_numpy(grid_thw).unsqueeze(0))

        data = {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "grid_thws": torch.cat(grid_thws, dim=0),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess_video(
        self,
        frames: Sequence[ImageInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """Preprocess a **video** as an ordered list of frames (one sample with T=len(frames)).

        All frames share the resize/pad config of the first frame so spatial grids align.
        ``T`` must be in ``[1, max_num_frames]`` (default 4, matching temporal pos-emb).
        """
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("`frames` must be a non-empty list/tuple of images.")
        if len(frames) > self.max_num_frames:
            raise ValueError(
                f"Got {len(frames)} frames, but max_num_frames={self.max_num_frames} "
                f"(MoonViT-V2 temporal pos-emb capacity)."
            )

        pil_frames = [_as_pil(f) for f in frames]
        cfg = self._resize_config(pil_frames[0])
        arrs = [self._normalize_frame(fr, cfg) for fr in pil_frames]
        pixels = np.stack(arrs, axis=0)  # (T, H, W, C)
        packed = _patchify(pixels, self.patch_size)

        data = {
            "pixel_values": torch.from_numpy(packed["pixel_values"]),
            "grid_thws": torch.from_numpy(packed["grid_thw"]).unsqueeze(0),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)
