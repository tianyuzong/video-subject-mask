#!/usr/bin/env python3
"""Colour a video's subjects by importance rank, leaving everything else untouched.

Rank 1 (the main subject) is painted red, rank 2 green, rank 3 blue. Pixels outside
every subject keep their original value. A JSON sidecar describes each subject.

Pipeline:  shot split -> Grounding DINO on keyframes -> NMS -> subject ranking ->
SAM 2.1 multi-object box prompt + propagation -> ranked fill -> ffmpeg mux.

Grounding DINO is a detector only (no mask branch), so it supplies the boxes that tell
SAM 2 *which* objects matter; SAM 2 produces the pixels and carries them across frames.

Example:
    python subject_mask.py \
        --video ~/datasets/VMQ/cases/single_0000_s8_walkthrough/video.mp4 \
        --detector ~/models/mm-gdino-swinb-hf \
        --sam ~/models/sam2.1-hiera-large \
        --out ~/outputs/subject_mask --device cpu --max-frames 12
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import cv2
import numpy as np
import torch
import torchvision
from transformers import (
    AutoProcessor,
    MMGroundingDinoForObjectDetection,
    Sam2VideoModel,
    Sam2VideoProcessor,
)

# Open-vocabulary candidates used when subjects are picked automatically. Grounding
# DINO wants a flat phrase list; broad nouns at the end act as a catch-all.
DEFAULT_CANDIDATES = [
    "person",
    "man",
    "woman",
    "child",
    "dog",
    "cat",
    "bird",
    "horse",
    "cow",
    "sheep",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "boat",
    "airplane",
    "animal",
    "object",
]

DEFAULT_PALETTE = "255,0,0:0,255,0:0,0,255"
RANK_NAMES = ["primary", "secondary", "tertiary"]



# --------------------------------------------------------------------------- video io


def probe_video(path: str) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "nb_frames": int(stream.get("nb_frames", 0)),
        "duration": float(info["format"]["duration"]),
    }


def has_audio(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def count_frames(path) -> int:
    """Exact decoded frame count - container metadata can lie."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().rstrip(","))


def read_frames(path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Decode the whole video to a list of RGB uint8 frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise SystemExit(f"decoded 0 frames from {path}")
    return frames


def detect_shots(path: str, num_frames: int, threshold: float = 0.4, fps: float = 30.0) -> list[int]:
    """Return shot start indices. A single shot yields [0]."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-filter:v",
         f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = {0}
    for line in proc.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            t = float(line.split("pts_time:")[1].split()[0])
        except (IndexError, ValueError):
            continue
        idx = int(round(t * fps))
        if 0 < idx < num_frames:
            starts.add(idx)
    return sorted(starts)


# ------------------------------------------------------------------------- detection


def detect_boxes(model, processor, frame: np.ndarray, phrases: list[str],
                 box_threshold: float, text_threshold: float, device: str):
    """Run Grounding DINO on one RGB frame. Returns (boxes_xyxy, scores, labels)."""
    text = [[p for p in phrases]]
    inputs = processor(images=frame, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(frame.shape[0], frame.shape[1])],
    )[0]
    return (
        results["boxes"].cpu().numpy(),
        results["scores"].cpu().numpy(),
        results["text_labels"],
    )


def subject_score(box: np.ndarray, score: float, h: int, w: int) -> float:
    """Rank a box by how much it reads as 'the subject': confident, big, centred."""
    x0, y0, x1, y1 = box
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_ratio = area / float(h * w)
    if area_ratio <= 0:
        return 0.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dist = np.hypot(cx - w / 2.0, cy - h / 2.0)
    centrality = max(0.0, 1.0 - 2.0 * dist / np.hypot(w, h))
    return float(score) ** 0.5 * area_ratio**0.6 * (0.2 + 0.8 * centrality)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def rank_candidates(boxes: np.ndarray, scores: np.ndarray, labels: list[str],
                    h: int, w: int, nms_iou: float, top_k: int,
                    min_rel_score: float = 0.0) -> list[dict]:
    """Deduplicate one frame's detections and return the top_k ranked subjects.

    The candidate vocabulary deliberately overlaps ("person"/"man"/"woman" all fire on
    the same human), so without class-agnostic NMS the top two "subjects" would be two
    boxes around one object and would get painted two different colours.

    min_rel_score drops trailing ranks that score far below rank 1. Without it a coat
    hanger or a distant car fragment gets promoted to "secondary subject" purely because
    nothing else was detected; rank 1 is always kept regardless.
    """
    if len(boxes) == 0:
        return []
    keep = torchvision.ops.nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        nms_iou,
    ).numpy()
    ranked = sorted(
        (
            {
                "subject_score": subject_score(boxes[i], scores[i], h, w),
                "box": boxes[i],
                "det_score": float(scores[i]),
                "text_label": labels[i] if i < len(labels) else "",
            }
            for i in keep
        ),
        key=lambda c: -c["subject_score"],
    )[:top_k]
    if min_rel_score > 0 and ranked:
        floor = ranked[0]["subject_score"] * min_rel_score
        ranked = [ranked[0]] + [c for c in ranked[1:] if c["subject_score"] >= floor]
    return ranked


# ------------------------------------------------------------------------ segmentation


def segment_shot(frames: list[np.ndarray], seed_idx: int, seed_boxes: list[np.ndarray],
                 sam_model, sam_processor, device: str,
                 rechecks: dict[int, list[np.ndarray | None]] | None = None,
                 iou_reseed: float = 0.3) -> np.ndarray:
    """Box-prompt SAM 2 with K subjects at seed_idx and propagate both ways.

    All K objects share one session and one propagation pass. rechecks maps
    frame_idx -> per-rank detector box (None where that rank had no detection); when a
    propagated mask drifts away from its detector box the box is re-injected for that
    object alone and propagation restarts from there. The other objects resume from
    their own memory, unaffected.

    Returns a (K, T, H, W) bool array indexed by rank.
    """
    num_objs = len(seed_boxes)
    obj_ids = list(range(1, num_objs + 1))  # obj_id == rank + 1
    h, w = frames[0].shape[:2]
    session = sam_processor.init_video_session(
        video=frames,
        inference_device=device,
        video_storage_device="cpu",
        dtype=torch.float32,
    )
    masks = np.zeros((num_objs, len(frames), h, w), dtype=bool)

    def add_boxes(frame_idx: int, boxes: dict[int, np.ndarray]) -> None:
        """boxes maps obj_id -> xyxy. Seeds them in one call."""
        ids = sorted(boxes)
        sam_processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=ids,
            input_boxes=[[[float(v) for v in boxes[i]] for i in ids]],
        )

    def store(output) -> dict[int, np.ndarray]:
        # (num_objects, 1, H, W) at original resolution; rows follow output.object_ids,
        # which is insertion order and NOT necessarily sorted.
        batch = sam_processor.post_process_masks(
            [output.pred_masks], original_sizes=[[h, w]], binarize=True,
            apply_non_overlapping_constraints=True,
        )[0]
        per_obj = {}
        for row, obj_id in enumerate(output.object_ids):
            m = batch[row, 0].cpu().numpy().astype(bool)
            masks[obj_id - 1, output.frame_idx] = m
            per_obj[obj_id] = m
        return per_obj

    add_boxes(seed_idx, {oid: seed_boxes[oid - 1] for oid in obj_ids})

    # forward, re-seeding whichever objects drift at a recheck frame
    cursor = seed_idx
    while True:
        reseed_at = None
        reseed_boxes: dict[int, np.ndarray] = {}
        for output in sam_model.propagate_in_video_iterator(session, start_frame_idx=cursor):
            per_obj = store(output)
            idx = output.frame_idx
            if not rechecks or idx not in rechecks or idx <= cursor:
                continue
            drifted = {}
            for oid in obj_ids:
                det_box = rechecks[idx][oid - 1]
                if det_box is None:
                    continue
                cur_box = mask_to_box(per_obj.get(oid, masks[oid - 1, idx]))
                if cur_box is None or box_iou(cur_box, det_box) < iou_reseed:
                    drifted[oid] = det_box
            if drifted:
                reseed_at, reseed_boxes = idx, drifted
                break
        if reseed_at is None:
            break
        add_boxes(reseed_at, reseed_boxes)
        cursor = reseed_at

    # backward from the seed
    if seed_idx > 0:
        for output in sam_model.propagate_in_video_iterator(
            session, start_frame_idx=seed_idx, reverse=True
        ):
            store(output)

    session.reset_inference_session()
    return masks


def clean_mask(mask: np.ndarray, close_kernel: int = 3) -> np.ndarray:
    if close_kernel <= 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return closed.astype(bool)


# ----------------------------------------------------------------------------- output


def open_writer(out_path: pathlib.Path, w: int, h: int, fps: float,
                src: str, lossless: bool, keep_audio: bool):
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-framerate", f"{fps}", "-i", "pipe:0",
    ]
    if keep_audio:
        # No -shortest: it truncates the video to whatever the encoded audio stream
        # happens to end at, silently dropping trailing frames.
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]
    if lossless:
        # ffv1 over rgb24 is bit-exact: no colourspace conversion, no quantisation.
        cmd += ["-c:v", "ffv1", "-level", "3"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd.append(str(out_path))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ------------------------------------------------------------------------------- main


def load_models(detector: str, sam: str, device: str):
    """Build the detector and segmenter once.

    Split out of process_video because these are 232.81M + 224.45M parameters
    (1.8 GB of weights). A pipeline that processes many clips must load them
    once per worker process, not once per clip; conductor's Processor.setup()
    is the right home for that.
    """
    print(f"[load] detector {detector}")
    det_processor = AutoProcessor.from_pretrained(detector)
    det_model = MMGroundingDinoForObjectDetection.from_pretrained(detector).to(device).eval()

    print(f"[load] sam {sam}")
    sam_processor = Sam2VideoProcessor.from_pretrained(sam)
    sam_model = Sam2VideoModel.from_pretrained(sam).to(device).eval()
    return det_processor, det_model, sam_processor, sam_model


def process_video(video: str, out_dir, models=None, detector=None, sam=None,
                  device=None, **opts) -> dict:
    """Segment one video. Returns a summary dict.

    ``models`` is the tuple from :func:`load_models`. Pass it to reuse weights
    across clips. Leave it None and the weights are loaded here, from
    ``detector`` and ``sam`` — that is the standalone path, and it costs a
    1.8 GB read each call, so a loop over many clips should pass ``models``.

    ``opts`` mirrors the CLI flags (``max_subjects``, ``prompt``, ``keyframe_stride``,
    ``save_masks``, ...). Defaults match the argparse defaults so behaviour is
    identical whichever way it is called.
    """
    g = {
        "max_subjects": 3, "prompt": None, "palette": DEFAULT_PALETTE,
        "nms_iou": 0.55, "min_rel_score": 0.10, "keyframe_stride": 15,
        "recheck_stride": 30, "iou_reseed": 0.3, "box_threshold": 0.3,
        "text_threshold": 0.25, "scene_threshold": 0.4, "no_shot_split": False,
        "close_kernel": 3, "max_frames": None, "lossless": False,
        "no_audio": False, "save_masks": False, "debug_overlay": False,
        "no_json": False,
    }
    unknown = set(opts) - set(g)
    if unknown:
        raise TypeError(f"process_video got unexpected options: {sorted(unknown)}")
    g.update(opts)
    video = str(pathlib.Path(video).expanduser())
    out_dir = pathlib.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(video).stem

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if models is None:
        if not detector or not sam:
            raise ValueError(
                "process_video needs either models=... or both detector=... and "
                "sam=...; without them the weights cannot be located"
            )
        models = load_models(detector, sam, device)
    det_processor, det_model, sam_processor, sam_model = models

    meta = probe_video(video)
    print(f"[video] {stem} {meta['width']}x{meta['height']} "
          f"{meta['fps']:.3f}fps {meta['nb_frames']} frames")

    frames = read_frames(video, g["max_frames"])
    num_frames = len(frames)
    h, w = frames[0].shape[:2]
    print(f"[decode] {num_frames} frames in memory")

    palette = [tuple(int(v) for v in c.split(",")) for c in g["palette"].split(":")]
    if len(palette) < g["max_subjects"]:
        raise ValueError(f"palette has {len(palette)} colours but max_subjects is "
                         f"{g['max_subjects']}")
    palette = palette[:g["max_subjects"]]
    phrases = [g["prompt"]] if g["prompt"] else DEFAULT_CANDIDATES

    if g["no_shot_split"] or num_frames < 2:
        shot_starts = [0]
    else:
        shot_starts = detect_shots(video, num_frames, g["scene_threshold"], meta["fps"])
        shot_starts = [s for s in shot_starts if s < num_frames]
    shot_bounds = list(zip(shot_starts, shot_starts[1:] + [num_frames]))

    rank_masks = np.zeros((g["max_subjects"], num_frames, h, w), dtype=bool)
    shot_records = []

    for shot_i, (start, end) in enumerate(shot_bounds):
        shot_frames = frames[start:end]
        n = len(shot_frames)
        keyframes = list(range(0, n, max(1, g["keyframe_stride"])))
        if keyframes[-1] != n - 1:
            keyframes.append(n - 1)

        per_frame: dict = {}
        best_seed, best_total = None, -1.0
        for k in keyframes:
            boxes, scores, labels = detect_boxes(
                det_model, det_processor, shot_frames[k], phrases,
                g["box_threshold"], g["text_threshold"], device)
            cands = rank_candidates(boxes, scores, labels, h, w, g["nms_iou"],
                                    g["max_subjects"], g["min_rel_score"])
            if not cands:
                continue
            per_frame[k] = cands
            total = sum(c["subject_score"] for c in cands)
            if total > best_total:
                best_seed, best_total = k, total

        if best_seed is None:
            print(f"[shot {shot_i}] no detection in {n} frames - left untouched")
            shot_records.append({"shot_index": shot_i, "start_frame": start,
                                 "end_frame": end - 1, "subjects": []})
            continue

        seeds = per_frame[best_seed]
        k_objs = len(seeds)
        rechecks = {}
        if g["recheck_stride"] > 0:
            for k, cands in per_frame.items():
                if k % g["recheck_stride"]:
                    continue
                rechecks[k] = [cands[r]["box"] if r < len(cands) else None
                               for r in range(k_objs)]

        shot_masks = segment_shot(shot_frames, best_seed, [c["box"] for c in seeds],
                                  sam_model, sam_processor, device, rechecks,
                                  g["iou_reseed"])

        subjects = []
        for r in range(k_objs):
            rows, areas, present_idx = [], [], []
            for i in range(n):
                m = clean_mask(shot_masks[r, i], g["close_kernel"])
                rank_masks[r, start + i] = m
                area = int(m.sum())
                bb = mask_to_box(m)
                present = bb is not None
                if present:
                    present_idx.append(i)
                    areas.append(area / float(h * w))
                rows.append({
                    "frame": start + i, "present": present,
                    "bbox_xyxy": [round(float(v), 1) for v in bb] if present else None,
                    "area_px": area, "area_ratio": round(area / float(h * w), 6),
                })
            subjects.append({
                "rank": r + 1,
                "rank_name": RANK_NAMES[r] if r < len(RANK_NAMES) else f"rank{r+1}",
                "rgb": list(palette[r]), "obj_id": r + 1,
                "text_label": seeds[r]["text_label"],
                "det_score": round(seeds[r]["det_score"], 4),
                "subject_score": round(seeds[r]["subject_score"], 6),
                "seed_frame": start + best_seed,
                "seed_box_xyxy": [round(float(v), 1) for v in seeds[r]["box"]],
                "first_frame": start + present_idx[0] if present_idx else None,
                "last_frame": start + present_idx[-1] if present_idx else None,
                "num_present_frames": len(present_idx),
                "mean_area_ratio": round(float(np.mean(areas)), 6) if areas else 0.0,
                "per_frame": rows,
            })
        shot_records.append({"shot_index": shot_i, "start_frame": start,
                             "end_frame": end - 1, "subjects": subjects})

    any_mask = rank_masks.any(axis=0)
    covered = int(any_mask.reshape(num_frames, -1).any(axis=1).sum())

    ext = "mkv" if g["lossless"] else "mp4"
    keep_audio = (not g["no_audio"]) and has_audio(video) and g["max_frames"] is None
    out_video = out_dir / f"{stem}_labeled.{ext}"
    writer = open_writer(out_video, w, h, meta["fps"], video, g["lossless"], keep_audio)
    for i, frame in enumerate(frames):
        out = frame.copy()
        for r in reversed(range(g["max_subjects"])):
            m = rank_masks[r, i]
            if m.any():
                out[m] = palette[r]
        writer.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg exited non-zero")
    written = count_frames(out_video)
    if written != num_frames:
        raise RuntimeError(f"{out_video} has {written} frames but {num_frames} were fed in")

    if not g["no_json"]:
        labels_doc = {
            "video": {"path": video, "width": w, "height": h, "fps": meta["fps"],
                      "num_frames": num_frames, "duration_sec": meta["duration"]},
            "params": {k: g[k] for k in (
                "max_subjects", "box_threshold", "text_threshold", "nms_iou",
                "min_rel_score", "keyframe_stride", "recheck_stride", "iou_reseed",
                "scene_threshold", "close_kernel", "prompt")},
            "palette": [{"rank": r + 1,
                         "name": RANK_NAMES[r] if r < len(RANK_NAMES) else f"rank{r+1}",
                         "rgb": list(c)} for r, c in enumerate(palette)],
            "notes": [
                "Frame indices are global (0-based into the source video).",
                "Ranks are assigned independently per shot: rank 1 in one shot is not "
                "guaranteed to be the same real-world object as rank 1 in another.",
            ],
            "shots": shot_records,
        }
        (out_dir / f"{stem}_labels.json").write_text(
            json.dumps(labels_doc, ensure_ascii=False, indent=2))

    if g["save_masks"]:
        np.savez_compressed(out_dir / f"{stem}_masks.npz", masks=rank_masks)

    if g["debug_overlay"]:
        idx = int(np.argmax(any_mask.reshape(num_frames, -1).sum(axis=1)))
        overlay = frames[idx].copy()
        for r in reversed(range(g["max_subjects"])):
            m = rank_masks[r, idx]
            if m.any():
                overlay[m] = (0.5 * overlay[m] + 0.5 * np.array(palette[r])).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_overlay_f{idx}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    n_subj = sum(len(s["subjects"]) for s in shot_records)
    print(f"[mask] non-empty on {covered}/{num_frames} frames, "
          f"coverage {any_mask.mean() * 100:.2f}%")
    return {"stem": stem, "labeled_path": str(out_video),
            "labels_path": str(out_dir / f"{stem}_labels.json"),
            "masks_path": str(out_dir / f"{stem}_masks.npz") if g["save_masks"] else None,
            "num_subjects": n_subj, "num_frames": num_frames,
            "coverage": round(float(any_mask.mean()), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--detector", required=True, help="HF MM-GroundingDINO dir")
    ap.add_argument("--sam", required=True, help="HF SAM 2.1 video dir")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", default=None,
                    help="explicit subject phrase, e.g. 'person'. Default: auto-rank "
                         "detections from a candidate vocabulary.")
    ap.add_argument("--max-subjects", type=int, default=3,
                    help="how many ranked subjects to colour (default 3)")
    ap.add_argument("--palette", default=DEFAULT_PALETTE,
                    help="colon-separated RGB per rank, e.g. '255,0,0:0,255,0:0,0,255'")
    ap.add_argument("--nms-iou", type=float, default=0.55,
                    help="class-agnostic NMS IoU; dedups overlapping vocabulary hits")
    ap.add_argument("--min-rel-score", type=float, default=0.10,
                    help="drop rank 2+ scoring below this fraction of rank 1's score, so "
                         "clutter is not promoted to 'secondary subject'. 0 disables.")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--keyframe-stride", type=int, default=15)
    ap.add_argument("--recheck-stride", type=int, default=30)
    ap.add_argument("--iou-reseed", type=float, default=0.3)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--scene-threshold", type=float, default=0.4)
    ap.add_argument("--no-shot-split", action="store_true")
    ap.add_argument("--close-kernel", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--lossless", action="store_true",
                    help="write ffv1/mkv so non-subject pixels stay bit-identical")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--debug-overlay", action="store_true")
    args = ap.parse_args()

    process_video(
        video=args.video,
        out_dir=args.out,
        detector=args.detector,
        sam=args.sam,
        device=args.device,
        prompt=args.prompt,
        max_subjects=args.max_subjects,
        palette=args.palette,
        nms_iou=args.nms_iou,
        min_rel_score=args.min_rel_score,
        no_json=args.no_json,
        keyframe_stride=args.keyframe_stride,
        recheck_stride=args.recheck_stride,
        iou_reseed=args.iou_reseed,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        scene_threshold=args.scene_threshold,
        no_shot_split=args.no_shot_split,
        close_kernel=args.close_kernel,
        max_frames=args.max_frames,
        lossless=args.lossless,
        no_audio=args.no_audio,
        save_masks=args.save_masks,
        debug_overlay=args.debug_overlay,
    )
    print("done")


if __name__ == "__main__":
    sys.exit(main())
