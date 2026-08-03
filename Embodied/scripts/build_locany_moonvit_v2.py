#!/usr/bin/env python3
"""Assemble LocateAnything-3B (LLM) + MoonViT-V2 (vision) with a fresh mlp1.

Image-only path: LocAny processor (H,W) + T=1 adapter lives in modeling code.
Saves a hybrid checkpoint usable as MODEL_PATH for Magi SFT.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn


def build_mlp1(vit_hidden: int, llm_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(vit_hidden * 4),
        nn.Linear(vit_hidden * 4, llm_hidden),
        nn.GELU(),
        nn.Linear(llm_hidden, llm_hidden),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--locany",
        type=Path,
        default=Path("/workspace/models/CommonModels/LocateAnything-3B"),
    )
    ap.add_argument(
        "--moonvit-v2",
        type=Path,
        default=Path("/workspace/models/CommonModels/MoonVit-v2"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/models/CommonModels/LocateAnything-3B-MoonViTV2"),
    )
    args = ap.parse_args()

    from transformers.utils import is_flash_attn_2_available

    from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
    from eaglevl.model.locany.modeling_locateanything import (
        LocateAnythingForConditionalGeneration,
    )
    from eaglevl.model.moon_vit_v2.configuration_moonvit_v2 import MoonViTV2Config
    from eaglevl.model.moon_vit_v2.modeling_moonvit_v2 import (
        MoonViTV2PretrainedModel,
        is_flash_attn_4_available,
    )

    if is_flash_attn_4_available():
        vision_attn = "flash_attention_4"
    elif is_flash_attn_2_available():
        vision_attn = "flash_attention_2"
    else:
        vision_attn = "sdpa"

    print(f"Loading LocateAnything from {args.locany}")
    # Load with original moonvit config so LLM weights map cleanly.
    config = LocateAnythingConfig.from_pretrained(str(args.locany))
    # Assemble on CPU: avoid "magi" (train-only) and FA4 (needs CUDA).
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_autoset = False
    model = LocateAnythingForConditionalGeneration.from_pretrained(
        str(args.locany),
        dtype=torch.bfloat16,
        config=config,
    )

    print(f"Loading MoonViT-V2 from {args.moonvit_v2} (attn={vision_attn})")
    v2_cfg = MoonViTV2Config.from_pretrained(str(args.moonvit_v2))
    # Persist preferred FA4→FA2→SDPA for later GPU training; load V2 with sdpa here.
    v2_cfg._attn_implementation = "sdpa"
    v2_cfg._attn_implementation_autoset = False
    vision = MoonViTV2PretrainedModel.from_pretrained(
        str(args.moonvit_v2),
        dtype=torch.bfloat16,
        config=v2_cfg,
    )

    vit_h = int(v2_cfg.hidden_size)
    llm_h = int(config.text_config.hidden_size)
    model.vision_model = vision
    model.mlp1 = build_mlp1(vit_h, llm_h).to(dtype=torch.bfloat16)
    model.config.vision_config = v2_cfg
    model.config.vision_config._attn_implementation = vision_attn

    print(
        f"vision={type(model.vision_model).__name__} "
        f"vit_hidden={vit_h} mlp_in={vit_h * 4} llm_hidden={llm_h}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving hybrid to {args.out}")
    model.save_pretrained(str(args.out), safe_serialization=True)
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = args.locany / name
        if src.is_file():
            shutil.copy2(src, args.out / name)

    repo = Path(__file__).resolve().parents[1]
    utils_locany = repo / "eaglevl" / "utils" / "locany"
    if utils_locany.is_dir():
        for f in utils_locany.iterdir():
            if f.is_file() and f.suffix in {".py", ".json", ".jinja"}:
                shutil.copy2(f, args.out / f.name)

    v2_dst = args.out / "moon_vit_v2"
    if v2_dst.exists():
        shutil.rmtree(v2_dst)
    shutil.copytree(repo / "eaglevl" / "model" / "moon_vit_v2", v2_dst)

    cfg_path = args.out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    # Ensure vision_config reflects V2 after save.
    v2_dict = v2_cfg.to_dict()
    v2_dict["model_type"] = "moonvit_v2"
    v2_dict["_attn_implementation"] = vision_attn
    cfg["vision_config"] = v2_dict
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print("done.")


if __name__ == "__main__":
    main()
