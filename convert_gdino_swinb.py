#!/usr/bin/env python3
"""Convert the mmdetection MM-GroundingDINO Swin-B checkpoint to HF format, offline.

Wraps transformers' official convert_mm_grounding_dino_to_hf.py but reads the .pth
from disk instead of downloading it, and builds the tokenizer from a local vocab.

    python convert_gdino_swinb.py \
        --ckpt /tmp/gdino_probe/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth \
        --tokenizer-dir ~/models/gdino_large \
        --out ~/models/mm-gdino-swinb-hf
"""

import argparse
import pathlib
import sys

import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import MMGroundingDinoForObjectDetection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mmengine_stub  # noqa: E402,F401  (registers stubs so torch.load can open the .pth)

# The upstream script imports httpx only to fetch a COCO test image for its optional
# output verification, and this huggingface_hub build doesn't re-export it. We convert
# offline and never call that path, so a placeholder is enough.
import huggingface_hub.utils as _hf_utils  # noqa: E402

if not hasattr(_hf_utils, "httpx"):
    try:
        import httpx as _httpx
    except ImportError:
        _httpx = None
    _hf_utils.httpx = _httpx

from convert_mm_grounding_dino_to_hf import (  # noqa: E402
    convert_mm_to_hf_state,
    get_mm_grounding_dino_config,
)

# The checkpoint this repo is built around; the official mapping ties this exact
# filename to the "base_o365v1_goldg_v3det" config.
MODEL_NAME = "mm_grounding_dino_base_o365v1_goldg_v3det"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="mmdet .pth checkpoint")
    ap.add_argument("--tokenizer-dir", required=True, help="dir holding bert vocab.txt")
    ap.add_argument("--out", required=True, help="output HF model dir")
    ap.add_argument("--model-name", default=MODEL_NAME)
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mm_state = ckpt["state_dict"]
    print(f"  {len(mm_state)} tensors")

    cfg = get_mm_grounding_dino_config(args.model_name)
    hf_state = convert_mm_to_hf_state(mm_state, cfg)
    print(f"  -> {len(hf_state)} hf tensors")

    model = MMGroundingDinoForObjectDetection(cfg).eval()
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:10])
    if unexpected:
        print("  unexpected sample:", unexpected[:10])
    if missing or unexpected:
        raise SystemExit("state_dict mismatch - refusing to write a silently broken model")

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
    processor = GroundingDinoProcessor(GroundingDinoImageProcessor(), tokenizer)

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
