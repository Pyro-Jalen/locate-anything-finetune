# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Multi-GPU DDP inference for PCB LocAny tasks.

Tasks (``--task``)::
  - dimension: line↔value matching
  - pad_hole: SMD Pad / Locating Hole detection
  - both: run both prompts per image

Label / prediction formats follow ``Embodied/data/data-detect-rule.md``::

  Pad/Hole (class-major, one ref + many boxes)::
    <ref>SMD Pad</ref><box>...</box><box>...</box>
    <ref>Locating Hole</ref><box>...</box>...

  Dimension (adjacent value pairs)::
    <ref>dim_text:VALUE</ref><box>xyxy</box>
    <ref>dim_axis:VALUE</ref><box>xyxy</box>
    or dim_text → dim_target (2-point box)

Input JSONL (aligned with ``test_dimension_pad_hole.jsonl``)::

    {"ID", "image_path", "dimension_label": [...], "pad_hole_label": [...]}

Output JSONL keeps the same GT fields and adds predictions::

    {"ID", "image_path", "dimension_label", "pad_hole_label",
     "model_response", "model_result",           # dimension
     "pad_hole_response", "pad_hole_result",     # pad/hole [{type,bbox}]
     "label"(=dimension_label, reward compat)}

Coords stay in norm1000 [0, 1000].
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from inference_compat import (
    apply_chat_template,
    build_generate_kwargs,
    decode_generation_output,
    prepare_generation_inputs,
    process_vision_info,
)

os.environ.setdefault("NCCL_TIMEOUT", "7200")

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "pcb_dimension_locate.txt"
)
DEFAULT_PAD_HOLE_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "pcb_smd_hole_locate.txt"
)
TASK_CHOICES = ("dimension", "pad_hole", "both")

# Canonical LocAny chunk: one <ref> then one or more <box> (see data-detect-rule.md).
REF_CHUNK_RE = re.compile(
    r"<ref>\s*([^<]+?)\s*</ref>\s*((?:<box>.*?</box>\s*)+)",
    re.DOTALL | re.IGNORECASE,
)
BOX_TAG_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL | re.IGNORECASE)
COORD_RE = re.compile(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>")
DIM_REF_RE = re.compile(
    r"^(dim_text|dim_axis|dim_target)\s*:\s*(.+)$",
    re.IGNORECASE,
)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
PAD_HOLE_REF_TO_TYPE = {
    "smd pad": "pad",
    "pad": "pad",
    "rect": "pad",
    "locating hole": "hole",
    "hole": "hole",
    "circle": "hole",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="DDP inference for PCB dimension match (LocateAnything → pad_detect schema)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dimension",
        choices=list(TASK_CHOICES),
        help="dimension | pad_hole | both",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/models/CheckPoints/size_line_value_match/locateanything-3b-full-pcb-magi",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default="/workspace/PROJECTS/github/Eagle/Embodied/data/test/test.jsonl",
        help="test JSONL with ID / image_path / dimension_label / pad_hole_label",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="",
        help="Optional root prepended to relative image_path. Empty = use image_path as-is.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/workspace/PROJECTS/pad_detect/results/size_line_value_match/"
        "locateanything-3b-full-pcb-magi.jsonl",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="Dimension task prompt (pcb_dimension_locate.txt)",
    )
    parser.add_argument(
        "--pad_hole_prompt_path",
        type=str,
        default=str(DEFAULT_PAD_HOLE_PROMPT_PATH),
        help="Pad/hole task prompt (pcb_smd_hole_locate.txt)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override dimension prompt text")
    parser.add_argument(
        "--pad_hole_prompt",
        type=str,
        default=None,
        help="Override pad/hole prompt text",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N samples")
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="hybrid",
        choices=["fast", "slow", "hybrid"],
    )
    parser.add_argument(
        "--short_side_size",
        type=int,
        default=None,
        help="Optional short-side resize before inference (coords stay norm1000; no remap).",
    )
    parser.add_argument(
        "--print_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-sample token stats + model output to terminal (default: on).",
    )
    parser.add_argument(
        "--print_max_chars",
        type=int,
        default=0,
        help="Truncate printed output to N chars (0 = print full output).",
    )
    parser.add_argument(
        "--print_history",
        action="store_true",
        help="Also print MTP/AR step chunks from verbose generate history.",
    )
    parser.add_argument(
        "--reparse_jsonl",
        type=str,
        default=None,
        help="Reparse an existing prediction JSONL (model_response→model_result) "
        "without loading the model. Writes back to --save_path (or in-place).",
    )
    # DDP
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Not using distributed mode")
        return 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=2),
    )
    dist.barrier()
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def load_prompt_text(override: str | None, path: str | Path) -> str:
    if override:
        return override.strip()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def load_prompts(args) -> dict[str, str]:
    """Return prompts needed for ``args.task``."""
    out: dict[str, str] = {}
    if args.task in ("dimension", "both"):
        out["dimension"] = load_prompt_text(args.prompt, args.prompt_path)
    if args.task in ("pad_hole", "both"):
        out["pad_hole"] = load_prompt_text(args.pad_hole_prompt, args.pad_hole_prompt_path)
    return out


def load_test_data(test_jsonl_path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
            if limit is not None and len(rows) >= limit:
                break
    return rows


def clamp_norm(v: float) -> float:
    return max(0.0, min(1000.0, float(v)))


def _parse_box_coords(box_inner: str) -> list[float] | None:
    """Parse LocAny box body into 2 (point) or 4 (xyxy) floats."""
    nums = [float(x) for x in COORD_RE.findall(box_inner or "")]
    if len(nums) in (2, 4):
        return nums
    return None


def iter_ref_chunks(text: str) -> list[tuple[str, list[list[float]]]]:
    """Yield ``(ref_name, [coords, ...])`` for each ``<ref>…</ref><box>…</box>+`` chunk."""
    chunks: list[tuple[str, list[list[float]]]] = []
    for ref_raw, boxes_str in REF_CHUNK_RE.findall(text or ""):
        coords_list: list[list[float]] = []
        for box_inner in BOX_TAG_RE.findall(boxes_str):
            coords = _parse_box_coords(box_inner)
            if coords is not None:
                coords_list.append(coords)
        if coords_list:
            chunks.append((ref_raw.strip(), coords_list))
    return chunks


def _norm_xyxy(coords: list[float]) -> list[float]:
    x1, y1, x2, y2 = (clamp_norm(v) for v in coords)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _norm_point(coords: list[float]) -> list[float]:
    if len(coords) == 2:
        return [clamp_norm(coords[0]), clamp_norm(coords[1])]
    x1, y1, x2, y2 = (clamp_norm(v) for v in coords[:4])
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def parse_dim_ref_name(name: str) -> tuple[str, str] | None:
    """Parse ``dim_text|dim_axis|dim_target:VALUE`` → ``(kind, value_text)``."""
    m = DIM_REF_RE.match(name.strip())
    if not m:
        return None
    kind = m.group(1).lower()
    value = m.group(2).strip()
    return (kind, value) if value else None


def parse_locany_dim_atoms(text: str) -> list[dict[str, Any]]:
    """Flatten dimension chunks into ordered atoms (one box each).

    Each atom: ``{kind, value_text, coords, is_point}``.
    Canonical labels use one box per ref; multiple boxes under one ref are
    expanded in order (same kind/value).
    """
    atoms: list[dict[str, Any]] = []
    for ref_name, coords_list in iter_ref_chunks(text):
        parsed = parse_dim_ref_name(ref_name)
        if parsed is None:
            continue
        kind, value_text = parsed
        for coords in coords_list:
            if kind == "dim_target":
                atoms.append(
                    {
                        "kind": kind,
                        "value_text": value_text,
                        "coords": _norm_point(coords),
                        "is_point": True,
                    }
                )
            elif len(coords) == 4:
                # Keep axis endpoints as-is (may be a degenerate line).
                if kind == "dim_axis":
                    xyxy = [clamp_norm(v) for v in coords]
                else:
                    xyxy = _norm_xyxy(coords)
                atoms.append(
                    {
                        "kind": kind,
                        "value_text": value_text,
                        "coords": xyxy,
                        "is_point": False,
                    }
                )
            elif len(coords) == 2 and kind == "dim_axis":
                x, y = _norm_point(coords)
                atoms.append(
                    {
                        "kind": kind,
                        "value_text": value_text,
                        "coords": [x, y, x, y],
                        "is_point": False,
                    }
                )
    return atoms


def _emit_dimension_match(
    value_text: str,
    value_bbox: list[float],
    geom_kind: str,
    geom: list[float],
) -> dict[str, Any] | None:
    item: dict[str, Any] = {
        "value_bbox": value_bbox,
        "value_text": value_text,
        "orientation": "horizontal",
    }
    if geom_kind == "dim_target":
        item["orientation"] = "size_value"
        if len(geom) == 2:
            item["target_point"] = geom
        elif len(geom) == 4:
            item["target_point"] = _norm_point(geom)
        else:
            return None
    else:
        if len(geom) != 4:
            return None
        dx = abs(geom[2] - geom[0])
        dy = abs(geom[3] - geom[1])
        item["orientation"] = "horizontal" if dx >= dy else "vertical"
        item["line_xyxy"] = geom
    return item


def refs_to_matches(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair adjacent ``dim_text`` → ``dim_axis|dim_target`` with the same VALUE.

    Matches ``data-detect-rule.md``: do not globally re-pair by value string.
    """
    matches: list[dict[str, Any]] = []
    i = 0
    n = len(atoms)
    while i < n:
        cur = atoms[i]
        if (
            cur["kind"] == "dim_text"
            and i + 1 < n
            and atoms[i + 1]["kind"] in ("dim_axis", "dim_target")
            and atoms[i + 1]["value_text"] == cur["value_text"]
        ):
            mate = atoms[i + 1]
            item = _emit_dimension_match(
                cur["value_text"],
                list(cur["coords"]),
                mate["kind"],
                list(mate["coords"]),
            )
            if item is not None:
                matches.append(item)
            i += 2
            continue
        i += 1
    return matches


def parse_dimension_matches_from_locany(
    text: str,
    label: list[dict[str, Any]] | None = None,
    **_kwargs,
) -> list[dict[str, Any]]:
    del label  # unused; kept for call-site compatibility
    cleaned = SPECIAL_TOKEN_RE.sub("", text or "").strip()
    return refs_to_matches(parse_locany_dim_atoms(cleaned))


def parse_pad_hole_from_locany(text: str) -> list[dict[str, Any]]:
    """Parse class-major LocAny pad/hole output → ``[{type,bbox},...]``.

    Canonical form (``data-detect-rule.md``)::

        <ref>SMD Pad</ref><box>...</box><box>...</box>
        <ref>Locating Hole</ref><box>...</box>...
    """
    cleaned = SPECIAL_TOKEN_RE.sub("", text or "").strip()
    items: list[dict[str, Any]] = []
    for ref_raw, coords_list in iter_ref_chunks(cleaned):
        typ = PAD_HOLE_REF_TO_TYPE.get(ref_raw.strip().lower())
        if typ is None:
            continue
        for coords in coords_list:
            if len(coords) != 4:
                continue
            items.append({"type": typ, "bbox": _norm_xyxy(coords)})
    return items


def reparse_prediction_jsonl(
    input_path: str,
    output_path: str | None = None,
    **_kwargs,
) -> dict[str, int]:
    """Re-parse model_response / pad_hole_response on an existing pred JSONL."""
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_dim = n_ph = n_empty_dim = n_empty_ph = 0
    rows_out: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            response = row.get("model_response") or row.get("raw_response") or ""
            if response or "model_result" in row:
                model_result = parse_dimension_matches_from_locany(response)
                row["model_result"] = model_result
                n_dim += len(model_result)
                if not model_result:
                    n_empty_dim += 1
            ph_resp = row.get("pad_hole_response") or ""
            if ph_resp or "pad_hole_result" in row:
                ph_result = parse_pad_hole_from_locany(ph_resp)
                row["pad_hole_result"] = ph_result
                n_ph += len(ph_result)
                if not ph_result:
                    n_empty_ph += 1
            # Keep GT field names aligned with test_dimension_pad_hole.jsonl
            if "dimension_label" not in row and "label" in row:
                row["dimension_label"] = row.get("label") or []
            row.setdefault("pad_hole_label", row.get("pad_hole_label") or [])
            rows_out.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "rows": n_rows,
        "dim_pairs": n_dim,
        "dim_empty": n_empty_dim,
        "pad_hole_boxes": n_ph,
        "pad_hole_empty": n_empty_ph,
    }
    print(
        f"[reparse] {in_path} → {out_path} | "
        f"rows={n_rows} dim_pairs={n_dim} pad_hole_boxes={n_ph}"
    )
    return stats


# Intermediate HF Trainer checkpoints often miss LocAny remote-code .py files.
# Final output_dir has them; copy from parent (or repo utils) before from_pretrained.
LOCANY_REMOTE_CODE_FILES = (
    "configuration_locateanything.py",
    "configuration_qwen2.py",
    "modeling_locateanything.py",
    "modeling_qwen2.py",
    "modeling_vit.py",
    "processing_locateanything.py",
    "image_processing_locateanything.py",
    "generate_utils.py",
    "attn_mask_utils.py",
    "mask_magi_utils.py",
    "mask_sdpa_utils.py",
)
LOCANY_AUTO_MAP = {
    "AutoConfig": "configuration_locateanything.LocateAnythingConfig",
    "AutoModel": "modeling_locateanything.LocateAnythingForConditionalGeneration",
    "AutoModelForCausalLM": "modeling_locateanything.LocateAnythingForConditionalGeneration",
    "AutoImageProcessor": "image_processing_locateanything.LocateAnythingImageProcessor",
    "AutoProcessor": "processing_locateanything.LocateAnythingProcessor",
}


# Always overwrite these from repo so mid-checkpoints pick up FA4 / magi loader fixes.
LOCANY_FORCE_REFRESH_FROM_REPO = (
    "modeling_vit.py",
    "modeling_locateanything.py",
)


def ensure_locany_remote_code(model_path: str) -> str:
    """Make a checkpoint-* dir loadable via trust_remote_code.

    Trainer mid-checkpoints keep weights but often omit custom modeling/config
    modules that only get copied into the final output_dir. If required files are
    missing, copy them from the parent run dir (preferred) or Embodied utils.

    Also force-refresh modeling_vit / modeling_locateanything from repo utils so
    checkpoints saved with vision ``flash_attention_4`` can load under current HF.
    """
    import shutil

    path = Path(model_path).resolve()
    if not path.is_dir():
        return model_path

    repo_utils = Path(__file__).resolve().parents[1] / "eaglevl" / "utils" / "locany"
    refreshed = []
    if repo_utils.is_dir():
        for name in LOCANY_FORCE_REFRESH_FROM_REPO:
            src = repo_utils / name
            if src.is_file():
                shutil.copy2(src, path / name)
                refreshed.append(name)

    missing = [name for name in LOCANY_REMOTE_CODE_FILES if not (path / name).is_file()]
    candidates = []
    if path.name.startswith("checkpoint-") and path.parent.is_dir():
        candidates.append(path.parent)
    if repo_utils.is_dir():
        candidates.append(repo_utils)

    copied = []
    still_missing = []
    for name in missing:
        src = next((c / name for c in candidates if (c / name).is_file()), None)
        if src is None:
            still_missing.append(name)
            continue
        shutil.copy2(src, path / name)
        copied.append(name)

    if still_missing:
        raise FileNotFoundError(
            f"{path} is missing LocAny remote-code files: {still_missing}. "
            f"Point --model_path at the final output_dir, or ensure parent has these files."
        )
    if refreshed:
        print(f"[ensure_locany_remote_code] Refreshed from repo into {path.name}: {', '.join(refreshed)}")
    if copied:
        print(f"[ensure_locany_remote_code] Copied into {path.name}: {', '.join(copied)}")

    _ensure_locany_auto_map(path)
    return str(path)


def _patch_hf_attn_implementation_allowlist() -> None:
    """Allow LocAny custom attn names that stock HF rejects (magi / flash_attention_4)."""
    import transformers.modeling_utils as _hf_modeling_utils

    if getattr(_hf_modeling_utils.PreTrainedModel, "_locany_attn_allowlist_patched", False):
        return
    _orig = _hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

    def _patched(self, attn_implementation, is_init_check=False):
        if attn_implementation in ("magi", "flash_attention_4"):
            return attn_implementation
        return _orig(self, attn_implementation, is_init_check)

    _hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = _patched
    _hf_modeling_utils.PreTrainedModel._locany_attn_allowlist_patched = True


def _ensure_locany_auto_map(path: Path) -> None:
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return
    auto_map = dict(cfg.get("auto_map") or {})
    updated = False
    for k, v in LOCANY_AUTO_MAP.items():
        if auto_map.get(k) != v:
            auto_map[k] = v
            updated = True
    if updated:
        cfg["auto_map"] = auto_map
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[ensure_locany_remote_code] Updated auto_map in {cfg_path}")


def _count_output_tokens(processor, text: str) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:
        return 0


def _format_sample_log(
    *,
    rank: int,
    sample_id: str,
    gen: dict[str, Any],
    model_result: list[dict[str, Any]],
    label: list,
    print_max_chars: int = 0,
    print_history: bool = False,
) -> str:
    output = gen.get("text") or ""
    n_gt = len(label) if isinstance(label, list) else 0
    n_pred = len(model_result)
    n_ref = output.count("<ref>")
    n_box = output.count("<box>")
    lines = [
        "",
        "=" * 72,
        f"[Rank {rank}] ID={sample_id}",
        (
            f"  tokens={gen.get('num_tokens')}  steps={gen.get('num_steps')}  "
            f"refs={n_ref}  boxes={n_box}  pred={n_pred}  gt={n_gt}"
        ),
    ]
    if gen.get("gen_info"):
        lines.append(f"  gen_info: {str(gen['gen_info']).strip()}")
    if print_history and gen.get("history"):
        lines.append("  history:")
        for i, item in enumerate(gen["history"]):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                mode, chunk = item[0], item[1]
            else:
                mode, chunk = "?", item
            chunk_s = str(chunk).replace("\n", "\\n")
            if len(chunk_s) > 120:
                chunk_s = chunk_s[:117] + "..."
            lines.append(f"    [{i:03d}] {mode}: {chunk_s}")
    shown = output
    if print_max_chars and print_max_chars > 0 and len(shown) > print_max_chars:
        shown = shown[:print_max_chars] + f"...<truncated {len(output) - print_max_chars} chars>"
    lines.append("  output:")
    lines.append(shown)
    lines.append("=" * 72)
    return "\n".join(lines)


class LocateAnythingMatchWorker:
    def __init__(self, model_path: str, device: str = "cuda", generation_mode: str = "hybrid"):
        self.device = device
        self.generation_mode = generation_mode
        _patch_hf_attn_implementation_allowlist()
        model_path = ensure_locany_remote_code(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if hasattr(self.processor, "tokenizer"):
            try:
                self.processor.tokenizer.padding_side = "left"
            except Exception:
                pass
        self.model = self.model.to(device)
        self.model.eval()

    def build_messages(self, image, prompt: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    @torch.inference_mode()
    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 8192,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Generate and return text plus token/step stats for terminal logging."""
        messages = self.build_messages(image, prompt)
        text_list = [apply_chat_template(self.processor, messages)]
        image_inputs, video_inputs = process_vision_info(self.processor, messages)
        processor_inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        prepared_inputs = prepare_generation_inputs(processor_inputs, self.device)
        generate_kwargs = build_generate_kwargs(
            prepared_inputs,
            self.processor,
            generation_mode=self.generation_mode,
            max_new_tokens=max_new_tokens,
            include_eos_token=True,
        )
        # LocAny generate(verbose=True) returns (text, sampling_history, stats_str)
        generate_kwargs["verbose"] = bool(verbose)
        raw_output = self.model.generate(**generate_kwargs)

        history = None
        gen_info = None
        if isinstance(raw_output, tuple):
            text = raw_output[0] if raw_output else ""
            if len(raw_output) >= 2:
                history = raw_output[1]
            if len(raw_output) >= 3:
                gen_info = raw_output[2]
            if not isinstance(text, str):
                text = decode_generation_output(
                    text, prepared_inputs["input_ids"], self.processor
                )
        else:
            text = decode_generation_output(
                raw_output,
                prepared_inputs["input_ids"],
                self.processor,
            )

        num_tokens = _count_output_tokens(self.processor, text)
        # Prefer token count parsed from model verbose stats when present.
        if isinstance(gen_info, str):
            m = re.search(r"num_tokens\s*=\s*(\d+)", gen_info)
            if m:
                num_tokens = int(m.group(1))
        num_steps = len(history) if isinstance(history, list) else None
        return {
            "text": text if isinstance(text, str) else str(text),
            "num_tokens": num_tokens,
            "num_steps": num_steps,
            "history": history,
            "gen_info": gen_info,
        }


def _gt_fields_from_sample(sample: dict) -> dict[str, Any]:
    """GT fields for prediction rows (reward keeps ``label`` alias)."""
    dimension_label = sample.get("dimension_label") or sample.get("label") or []
    return {
        "dimension_label": dimension_label,
        "pad_hole_label": sample.get("pad_hole_label") or [],
        "label": dimension_label,
    }


class MatchDataset(Dataset):
    def __init__(self, test_data: list[dict], image_root_dir: str):
        self.test_data = test_data
        self.image_root_dir = image_root_dir or ""

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        entry = self.test_data[idx]
        image_path = entry.get("image_path") or entry.get("image") or ""
        sample_id = entry.get("ID") or entry.get("id") or Path(image_path).name
        # Prefer dimension_label; fall back to legacy label.
        dimension_label = entry.get("dimension_label")
        if dimension_label is None:
            dimension_label = entry.get("label") or []
        pad_hole_label = entry.get("pad_hole_label") or []
        if self.image_root_dir and image_path and not os.path.isabs(image_path):
            full_image_path = os.path.join(self.image_root_dir, image_path)
        else:
            full_image_path = image_path
        return {
            "ID": sample_id,
            "image_path": image_path,
            "full_image_path": full_image_path,
            "dimension_label": dimension_label,
            "pad_hole_label": pad_hole_label,
            # Alias for older call sites / reward --gt-key label
            "label": dimension_label,
            "idx": idx,
        }


def resize_image_short_side(image: Image.Image, short_side_size: int):
    w, h = image.size
    if w <= h:
        new_w = short_side_size
        scale = new_w / w
        new_h = int(h * scale)
    else:
        new_h = short_side_size
        scale = new_h / h
        new_w = int(w * scale)
    return image.resize((new_w, new_h), Image.BILINEAR), scale


def main():
    args = get_args()

    # Offline reparse path: no model / no DDP needed.
    if args.reparse_jsonl:
        out = args.save_path or args.reparse_jsonl
        reparse_prediction_jsonl(args.reparse_jsonl, out)
        return

    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
    prompts = load_prompts(args)
    run_dimension = args.task in ("dimension", "both")
    run_pad_hole = args.task in ("pad_hole", "both")

    if is_main_process():
        print("=== PCB LocAny DDP Inference ===")
        print(f"Task: {args.task}")
        print(f"World Size: {world_size}")
        print(f"Model Path: {args.model_path}")
        print(f"Test JSONL: {args.test_jsonl_path}")
        print(f"Save Path: {args.save_path}")
        print(f"Generation Mode: {args.generation_mode}")
        for name, text in prompts.items():
            print(f"Prompt[{name}]:\n{text}\n")

    save_dir = os.path.dirname(args.save_path)
    if is_main_process() and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    worker = LocateAnythingMatchWorker(
        args.model_path, device=device, generation_mode=args.generation_mode
    )

    test_data = load_test_data(args.test_jsonl_path, limit=args.limit)
    if is_main_process():
        print(f"Loaded {len(test_data)} test entries")

    dataset = MatchDataset(test_data, args.image_root_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    local_predictions: list[dict] = []
    iterator = tqdm(dataloader, desc=f"Rank {rank}", disable=not is_main_process())

    def _empty_pred(sample: dict, error: str | None = None) -> dict:
        row = {
            "ID": sample["ID"],
            "image_path": sample["image_path"],
            **_gt_fields_from_sample(sample),
            "model_response": "",
            "model_result": [],
            "pad_hole_response": "",
            "pad_hole_result": [],
        }
        if error:
            row["error"] = error
        return row

    for sample in iterator:
        full_image_path = sample["full_image_path"]
        if not full_image_path or not os.path.exists(full_image_path):
            print(f"[Rank {rank}] Warning: Image not found: {full_image_path}")
            local_predictions.append(
                _empty_pred(sample, error=f"image_not_found: {full_image_path}")
            )
            continue

        try:
            image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"[Rank {rank}] Error loading image {full_image_path}: {e}")
            local_predictions.append(_empty_pred(sample, error=str(e)))
            continue

        if args.short_side_size is not None:
            image, _ = resize_image_short_side(image, args.short_side_size)

        row = _empty_pred(sample)
        total_tokens = 0
        total_steps = 0

        if run_dimension:
            try:
                gen = worker.generate(
                    image,
                    prompts["dimension"],
                    max_new_tokens=args.max_new_tokens,
                    verbose=True,
                )
                output = gen["text"]
                row["model_response"] = output if isinstance(output, str) else str(output)
                try:
                    row["model_result"] = parse_dimension_matches_from_locany(output)
                except Exception as e:
                    print(f"[Rank {rank}] Dimension parse failed for {sample['ID']}: {e}")
                    row["model_result"] = []
                total_tokens += int(gen.get("num_tokens") or 0)
                total_steps += int(gen.get("num_steps") or 0)
                if args.print_sample:
                    log_block = _format_sample_log(
                        rank=rank,
                        sample_id=f"{sample['ID']}[dimension]",
                        gen=gen,
                        model_result=row["model_result"],
                        label=sample["dimension_label"],
                        print_max_chars=args.print_max_chars,
                        print_history=args.print_history,
                    )
                    try:
                        iterator.write(log_block)
                    except Exception:
                        print(log_block, flush=True)
            except Exception as e:
                print(f"[Rank {rank}] Dimension generate failed for {sample['ID']}: {e}")
                row["error"] = f"dimension_generate_failed: {e}"

        if run_pad_hole:
            try:
                gen = worker.generate(
                    image,
                    prompts["pad_hole"],
                    max_new_tokens=args.max_new_tokens,
                    verbose=True,
                )
                output = gen["text"]
                row["pad_hole_response"] = output if isinstance(output, str) else str(output)
                try:
                    row["pad_hole_result"] = parse_pad_hole_from_locany(output)
                except Exception as e:
                    print(f"[Rank {rank}] Pad/hole parse failed for {sample['ID']}: {e}")
                    row["pad_hole_result"] = []
                total_tokens += int(gen.get("num_tokens") or 0)
                total_steps += int(gen.get("num_steps") or 0)
                if args.print_sample:
                    log_block = _format_sample_log(
                        rank=rank,
                        sample_id=f"{sample['ID']}[pad_hole]",
                        gen=gen,
                        model_result=row["pad_hole_result"],
                        label=sample.get("pad_hole_label") or [],
                        print_max_chars=args.print_max_chars,
                        print_history=args.print_history,
                    )
                    try:
                        iterator.write(log_block)
                    except Exception:
                        print(log_block, flush=True)
            except Exception as e:
                print(f"[Rank {rank}] Pad/hole generate failed for {sample['ID']}: {e}")
                prev = row.get("error")
                row["error"] = (
                    f"{prev}; pad_hole_generate_failed: {e}"
                    if prev
                    else f"pad_hole_generate_failed: {e}"
                )

        row["num_tokens"] = total_tokens or None
        row["num_steps"] = total_steps or None
        local_predictions.append(row)

    print(f"[Rank {rank}] Finished {len(local_predictions)} samples")

    base_name = os.path.basename(args.save_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1] or ".jsonl"
    rank_save_path = os.path.join(save_dir or ".", f"{name_without_ext}_rank{rank}{ext}")
    with open(rank_save_path, "w", encoding="utf-8") as f:
        for pred in local_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"[Rank {rank}] Saved to {rank_save_path}")

    if dist.is_initialized():
        dist.barrier()

    if is_main_process():
        all_predictions: list[dict] = []
        for r in range(world_size):
            r_path = os.path.join(save_dir or ".", f"{name_without_ext}_rank{r}{ext}")
            if not os.path.exists(r_path):
                continue
            with open(r_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_predictions.append(json.loads(line))
            os.remove(r_path)

        id_order = {
            (row.get("ID") or row.get("id") or Path(row.get("image_path", "")).name): i
            for i, row in enumerate(test_data)
        }
        all_predictions.sort(key=lambda p: id_order.get(p.get("ID"), 10**9))

        with open(args.save_path, "w", encoding="utf-8") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"Saved {len(all_predictions)} predictions to {args.save_path}")

        if run_dimension:
            n_empty = sum(1 for p in all_predictions if not p.get("model_result"))
            n_pairs = sum(len(p.get("model_result") or []) for p in all_predictions)
            n_gt = sum(
                len(p.get("dimension_label") or p.get("label") or [])
                for p in all_predictions
            )
            print(
                f"Dimension summary: samples={len(all_predictions)} empty_pred={n_empty} "
                f"pred_pairs={n_pairs} gt_pairs={n_gt}"
            )

        if run_pad_hole:
            from pad_hole_metrics import build_pad_hole_summary, print_pad_hole_table

            n_empty = sum(1 for p in all_predictions if not p.get("pad_hole_result"))
            n_pred = sum(len(p.get("pad_hole_result") or []) for p in all_predictions)
            n_gt = sum(len(p.get("pad_hole_label") or []) for p in all_predictions)
            print(
                f"Pad/hole summary: samples={len(all_predictions)} empty_pred={n_empty} "
                f"pred_boxes={n_pred} gt_boxes={n_gt}"
            )
            summary = build_pad_hole_summary(all_predictions, iou_report=0.5)
            print("=== Pad/Hole detection metrics (norm1000, IoU@0.5 for Box P/R) ===")
            print_pad_hole_table(summary)
            metrics_path = Path(args.save_path).with_suffix(".pad_hole_metrics.json")
            metrics_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote pad/hole metrics: {metrics_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
