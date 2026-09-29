#!/usr/bin/env python3
"""Randomly convex-blur subject masks so a DiT cannot read the subject off the mask.

An exact silhouette *is* the answer: a person-shaped hole tells the model to draw a
person without understanding anything about the scene. This pushes each mask toward its
convex hull, dilates it, and feathers the edge, turning the silhouette into a generic
blob. Every hyper-parameter is sampled automatically per (video, shot, rank).

Two invariants the implementation must never break:

  1. The alpha==1 core is a strict SUPERSET of the original mask. Otherwise subject
     pixels survive into the masked video and the model sees them directly.
  2. The alpha==1 core is NOT EQUAL to the original mask. Otherwise thresholding alpha
     at 1.0 recovers the exact silhouette and the whole exercise is cosmetic.

Both are checked at runtime (see _perturb_frame) and asserted by run_perturb_smoke.sh.

Usage:
    python mask_perturb.py --masks-dir ~/outputs/subject_mask_multi \
                           --out ~/outputs/perturbed --workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from subject_mask import count_frames, has_audio, probe_video, read_frames  # noqa: E402

# Sampling ranges. These are the tuning surface: change them here rather than threading
# a dozen CLI flags through every batch invocation.
RANGES = {
    # Tuned for "cover the subject, don't overshoot". Shrinking these has diminishing
    # returns because the shape itself is the floor: measured, going from
    # (0.06, 0.05, 0.05) down to (0.01, 0.015, 0.025) only moves hull from 1.59x to
    # 1.34x. The bigger lever on area is SHAPE_WEIGHTS below.
    "shape_jitter": (0.01, 0.03),  # outward push, as a fraction of sqrt(area)
    "dilate": (0.01, 0.03),        # core growth on top of the shape, same units
    "feather": (0.02, 0.04),       # alpha ramp width, same units
    "aniso": (1.0, 1.15),          # stretch factor so blobs are not always isotropic
    "drift_amp": (0.04, 0.12),     # slow temporal wobble on jitter/dilate/feather
    "drift_period": (1.5, 4.0),    # seconds
}

# Drawn per (video, shot, rank). Circle and triangle were measured and rejected: a
# circle around an elongated person averages 2.22x the mask area and peaks at 4.60x,
# blanking a large slab of background for no extra ambiguity.
#
# Weighted toward the hull, which at these ranges covers the subject in 1.37x the mask
# area against 1.83-2.08x for the others. Shape variety and tight coverage pull in
# opposite directions; this trades a little variety for noticeably less blanked-out
# background. Set all four equal for maximum variety.
SHAPES = ("hull", "rotrect", "bbox", "ellipse")
SHAPE_WEIGHTS = (0.40, 0.25, 0.15, 0.20)

# Per-shape area ceilings, just above each shape's measured p99 at the ranges above, so
# the bisection only fires on genuine outliers instead of routinely clawing every shape
# back toward the hull. Scaled by --max-growth.
SHAPE_CAPS = {"hull": 2.3, "rotrect": 2.8, "bbox": 2.9, "ellipse": 3.0}

# Grey is the default because 127 maps to ~0 once pixels are normalised to [-1,1], i.e.
# the neutral "no information" value. Black or white sit at the extremes and inject a
# strong signal the model then has to compensate for. "noise" fills with per-frame
# uniform noise instead of a constant, so the occluded region carries no flat-colour
# prior at all. "ranked" paints each rank in its own palette colour from the labels
# file - the blurred counterpart of subject_mask.py's labelled video.
FILLS = {"grey": (127, 127, 127), "black": (0, 0, 0), "white": (255, 255, 255),
         "noise": None, "ranked": None, "random": None}


def sample_colour(rng: np.random.Generator, taken: list[tuple[int, int, int]],
                  min_dist: float = 90.0) -> tuple[int, int, int]:
    """A saturated RGB, kept clear of colours already used in this shot.

    Sampled in HSV and kept away from the low-saturation region so the fill never
    reads as "just a grey patch" - that would reintroduce the flat-colour prior the
    randomisation is meant to remove. Colours are also pushed apart within a shot,
    otherwise rank 1 and rank 2 can land on near-identical hues and become impossible
    to tell apart, both for a human checking output and for anything downstream.
    """
    best, best_gap = None, -1.0
    for _ in range(16):
        hsv = np.uint8([[[int(rng.integers(0, 180)),
                          int(rng.integers(120, 256)),
                          int(rng.integers(90, 231))]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        gap = min((float(np.linalg.norm(np.array(rgb) - np.array(t))) for t in taken),
                  default=1e9)
        if gap >= min_dist:
            return rgb
        if gap > best_gap:
            best, best_gap = rgb, gap
    return best


def unit_rng(run_seed: int, stem: str, shot: int, rank: int) -> np.random.Generator:
    """Derive an independent RNG per (video, shot, rank).

    Hash-derived rather than time- or PID-seeded: parallel workers starting in the same
    millisecond would otherwise collide and hand different videos identical parameters,
    which fails silently and at scale.
    """
    digest = hashlib.blake2b(f"{run_seed}|{stem}|{shot}|{rank}".encode(),
                             digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def sample_params(rng: np.random.Generator, fps: float) -> dict:
    u = rng.uniform
    return {
        "shape_jitter": float(u(*RANGES["shape_jitter"])),
        "dilate": float(u(*RANGES["dilate"])),
        "feather": float(u(*RANGES["feather"])),
        "aniso": float(u(*RANGES["aniso"])),
        "aniso_angle": float(u(0, np.pi)),
        "drift_amp": float(u(*RANGES["drift_amp"])),
        "drift_period_frames": float(u(*RANGES["drift_period"]) * max(fps, 1.0)),
        "drift_phase": [float(u(0, 2 * np.pi)) for _ in range(3)],
    }


def drifted(p: dict, t: int) -> tuple[float, float, float]:
    """Slow sinusoidal wobble. Per-frame independent randomness would flicker, and
    averaging many frames of it would reconstruct the true silhouette."""
    w = 2 * np.pi * t / max(p["drift_period_frames"], 1.0)
    j, d, f = (p["shape_jitter"], p["dilate"], p["feather"])
    amp = p["drift_amp"]
    j = float(max(0.0, j * (1 + amp * np.sin(w + p["drift_phase"][0]))))
    d = float(d * (1 + amp * np.sin(w + p["drift_phase"][1])))
    f = float(f * (1 + amp * np.sin(w + p["drift_phase"][2])))
    return j, d, f


_ANISO_CACHE: dict = {}


def _aniso_weight(shape: tuple[int, int], aniso: float, angle: float) -> np.ndarray:
    """Cached direction weight. aniso/angle are fixed per (video, shot, rank), so this
    is a hit on every frame after the first - rebuilding the grid per frame was costing
    ~3ms of the ~5.5ms budget."""
    key = (shape, round(aniso, 4), round(angle, 4))
    hit = _ANISO_CACHE.get(key)
    if hit is not None:
        return hit
    c, s = np.cos(angle), np.sin(angle)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    proj = np.abs((xx - cx) * c + (yy - cy) * s)
    norm = np.hypot(xx - cx, yy - cy) + 1e-6
    weight = (1.0 + (aniso - 1.0) * (proj / norm)).astype(np.float32)
    if len(_ANISO_CACHE) > 64:
        _ANISO_CACHE.clear()
    _ANISO_CACHE[key] = weight
    return weight


def _anisotropic(dist: np.ndarray, aniso: float, angle: float) -> np.ndarray:
    """Squash the distance field along one axis so the level sets are ellipses.

    Scaling a distance field by <1 along an axis makes the thresholded region grow
    further along it. Applied to the *distance*, never to the mask, so the superset
    property is untouched.
    """
    if aniso <= 1.001:
        return dist
    return dist / _aniso_weight(dist.shape, aniso, angle)


def _shape_poly(kind: str, contour: np.ndarray, jitter_px: float,
                rng: np.random.Generator, area_budget: float = 0.0) -> np.ndarray:
    """A convex polygon enclosing one contour, jittered in a way that keeps its family.

    Each shape gets its own jitter: pushing individual vertices would sand a rectangle
    down into an octagon and lose the point of having rectangles in the pool.

    area_budget, when positive, is the largest polygon area tolerated for this
    component; a shape that overshoots falls back to the hull.
    """
    if kind == "hull":
        pts = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
        if jitter_px > 0 and len(pts) >= 3:
            centre = pts.mean(axis=0)
            radial = pts - centre
            norm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6
            push = rng.uniform(0.0, jitter_px, size=(len(pts), 1)).astype(np.float32)
            pts = np.vstack([pts, pts + radial / norm * push])
        return cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)

    if kind in ("rotrect", "bbox"):
        if kind == "rotrect":
            (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy, angle = x + w / 2.0, y + h / 2.0, 0.0
        # widen each side; +2 covers boxPoints rounding
        w += float(rng.uniform(0.0, 2 * jitter_px)) + 2.0
        h += float(rng.uniform(0.0, 2 * jitter_px)) + 2.0
        return np.int32(cv2.boxPoints(((cx, cy), (w, h), angle))).reshape(-1, 2)

    if kind == "ellipse":
        if len(contour) < 5:
            return _shape_poly("hull", contour, jitter_px, rng)
        (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
        # fitEllipse *fits*, it does not enclose. Scale until every contour point is
        # inside, then jitter outward from there.
        pts = contour.reshape(-1, 2).astype(np.float64)
        th = np.deg2rad(angle)
        cos_t, sin_t = np.cos(th), np.sin(th)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = (dx * cos_t + dy * sin_t) / max(major / 2, 1e-6)
        v = (-dx * sin_t + dy * cos_t) / max(minor / 2, 1e-6)
        k = max(float(np.sqrt(np.max(u * u + v * v))), 1.0) * 1.02
        a = major / 2 * k + float(rng.uniform(0.0, jitter_px)) + 1.0
        b = minor / 2 * k + float(rng.uniform(0.0, jitter_px)) + 1.0
        # A poor fit (crescents, L-shapes) makes k explode - measured up to 9x the mask
        # area before this guard. Fall back to the hull rather than blanking the frame.
        if area_budget > 0 and np.pi * a * b > area_budget:
            return _shape_poly("hull", contour, jitter_px, rng)
        return cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                (int(np.ceil(a)), int(np.ceil(b))),
                                int(round(angle)), 0, 360, 10)

    raise ValueError(f"unknown shape {kind!r}")


def _shape_union(mask_small: np.ndarray, kind: str, jitter_px: float,
                 min_comp_frac: float, jitter_seed: int,
                 area_cap: float = 0.0) -> np.ndarray:
    """One convex primitive per connected component, unioned, then unioned with the mask.

    Per component, not one shape over everything: measured over 783 real rank-frames,
    a single global hull grows the area 17.5x in the worst case because it swallows the
    gaps between disconnected fragments (a crowd of legs can be 40 components). Per
    component the worst case is 2.15x, and each blob stays a strict convex polygon.

    The final union with the mask is not cosmetic. Rasterising a primitive from rounded
    vertices can fall a pixel short: measured bare, rotrect enclosed the mask on only
    63/198 rank-frames (circle 54, triangle 50). With the union it is 198/198 for every
    shape. Relying on the `alpha[mask] = 1.0` line at the end instead would fix the
    alpha values while leaving the core region itself wrong.

    jitter_seed is re-seeded identically on every frame, so a given vertex gets the same
    push throughout a shot. Fresh randomness per frame would make the boundary crawl,
    and averaging the crawl would expose the true silhouette; the magnitude still varies
    smoothly because jitter_px carries the temporal drift.
    """
    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_small)
    areas = [cv2.contourArea(c) for c in contours]
    biggest = max(areas) if areas else 0.0
    rng = np.random.default_rng(jitter_seed)

    out = np.zeros_like(mask_small)
    for contour, area in zip(contours, areas):
        if biggest > 0 and area < min_comp_frac * biggest:
            continue                       # SAM speckle, not a real part of the subject
        budget = area_cap * area if area_cap > 0 else 0.0
        poly = _shape_poly(kind, contour, jitter_px, rng, budget)
        if len(poly) >= 3:
            cv2.fillConvexPoly(out, poly.astype(np.int32), 1)
    out |= mask_small                      # enclosure by construction, not by luck
    return out


def _perturb_frame(mask: np.ndarray, kind: str, jitter: float, d_frac: float,
                   f_frac: float, aniso: float, angle: float, work_res: int,
                   max_growth: float, min_comp_frac: float,
                   jitter_seed: int) -> tuple[np.ndarray, float]:
    """Return (alpha float32 in [0,1] at full res, achieved core/mask area ratio)."""
    full_h, full_w = mask.shape
    if not mask.any():
        return np.zeros((full_h, full_w), np.float32), 0.0

    scale = min(1.0, work_res / float(max(full_h, full_w)))
    if scale < 1.0:
        small = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_NEAREST)
    else:
        small = mask.astype(np.uint8)
    if not small.any():                       # tiny object lost by downscaling
        small = mask.astype(np.uint8)

    area = float(small.sum())
    radius = np.sqrt(area)
    base = _shape_union(small, kind, jitter * radius, min_comp_frac, jitter_seed,
                        max_growth)
    shape_ratio = float(base.sum()) / max(area, 1.0)

    dist = cv2.distanceTransform(1 - base, cv2.DIST_L2, 3)
    dist = _anisotropic(dist, aniso, angle)

    # The shape is the floor: at d=0 the core *is* the shape, so the cap can never push
    # below shape_ratio. Bisect the dilation only.
    d = max(d_frac * radius, 1.0)
    core = (dist <= d).astype(np.uint8)
    ratio = float(core.sum()) / max(area, 1.0)
    lo, hi = 0.0, d
    for _ in range(8):
        if ratio <= max_growth or shape_ratio >= max_growth:
            break
        hi = (lo + hi) / 2.0
        core = (dist <= hi).astype(np.uint8)
        ratio = float(core.sum()) / max(area, 1.0)

    feather = max(f_frac * radius, 1.0)
    alpha_small = np.clip(1.0 - cv2.distanceTransform(1 - core, cv2.DIST_L2, 3) / feather,
                          0.0, 1.0).astype(np.float32)

    if alpha_small.shape != mask.shape:
        alpha = cv2.resize(alpha_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    else:
        alpha = alpha_small
    # Defensive: absorb 1-2px resampling error at the boundary. A no-op for sane
    # parameters, but a silent subject leak if it ever were not.
    alpha[mask] = 1.0
    return alpha, ratio


def open_gray_writer(path: pathlib.Path, w: int, h: int, fps: float):
    """Lossless grayscale ffv1. Alpha is a conditioning signal - h264 would smear both
    the flat core and the ramp, and the exact values are what the DiT consumes."""
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
           "-framerate", f"{fps}", "-i", "pipe:0",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def open_color_writer(path: pathlib.Path, w: int, h: int, fps: float,
                      src: str, lossless: bool, keep_audio: bool):
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", f"{fps}", "-i", "pipe:0"]
    if keep_audio:
        # No -shortest: it truncates video to the encoded audio length, dropping frames.
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]
    if lossless:
        cmd += ["-c:v", "ffv1", "-level", "3"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd.append(str(path))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_video(stem: str, masks_dir: str, out_dir: str, run_seed: int,
                  work_res: int, max_growth: float, fill_name: str,
                  lossless: bool, save_npz: bool, no_audio: bool,
                  min_comp_frac: float) -> dict:
    masks_dir, out_dir = pathlib.Path(masks_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    masks = np.load(masks_dir / f"{stem}_masks.npz")["masks"]          # (K,T,H,W) bool
    video = labels["video"]["path"]
    fps = labels["video"]["fps"]
    K, T, H, W = masks.shape

    frames = read_frames(video, T)
    if len(frames) != T:
        raise SystemExit(f"{stem}: decoded {len(frames)} frames but masks hold {T}")

    # Which shot each frame belongs to, so parameters change on shot boundaries.
    shot_of = np.zeros(T, dtype=int)
    for sh in labels["shots"]:
        shot_of[sh["start_frame"]:sh["end_frame"] + 1] = sh["shot_index"]

    params_by_unit = {}
    jitter_seed_by_unit = {}
    shape_by_unit = {}
    colour_by_unit = {}
    for sh in labels["shots"]:
        taken: list[tuple[int, int, int]] = []       # colours already used in this shot
        for subj in sh["subjects"]:
            r = subj["rank"] - 1
            key = (sh["shot_index"], r)
            rng = unit_rng(run_seed, stem, sh["shot_index"], subj["rank"])
            params_by_unit[key] = sample_params(rng, fps)
            shape_by_unit[key] = str(rng.choice(SHAPES, p=SHAPE_WEIGHTS))
            colour = sample_colour(rng, taken)
            taken.append(colour)
            colour_by_unit[key] = colour
            # A per-unit constant, replayed identically on every frame so the shape's
            # jittered boundary stays put instead of crawling.
            jitter_seed_by_unit[key] = int(rng.integers(0, 2**63 - 1))

    alpha_all = np.zeros((K, T, H, W), np.uint8)
    ratios = {r: [] for r in range(K)}
    t0 = time.perf_counter()
    rank_frames = 0

    for r in range(K):
        for t in range(T):
            m = masks[r, t]
            if not m.any():
                continue
            key = (int(shot_of[t]), r)
            p = params_by_unit.get(key)
            if p is None:
                continue
            j, d, f = drifted(p, t)
            kind = shape_by_unit[key]
            alpha, ratio = _perturb_frame(m, kind, j, d, f, p["aniso"], p["aniso_angle"],
                                          work_res, SHAPE_CAPS[kind] * max_growth,
                                          min_comp_frac, jitter_seed_by_unit[key])
            alpha_all[r, t] = np.round(alpha * 255).astype(np.uint8)
            ratios[r].append(ratio)
            rank_frames += 1
    perturb_ms = (time.perf_counter() - t0) / max(rank_frames, 1) * 1000

    # --- write alpha tracks (lossless gray) ---
    alpha_writers = {}
    for r in range(K):
        if alpha_all[r].any():
            alpha_writers[r] = open_gray_writer(
                out_dir / f"{stem}_alpha_rank{r + 1}.mkv", W, H, fps)
    for t in range(T):
        for r, wr in alpha_writers.items():
            wr.stdin.write(alpha_all[r, t].tobytes())
    for r, wr in alpha_writers.items():
        wr.stdin.close()
        if wr.wait() != 0:
            raise SystemExit(f"{stem}: alpha rank{r + 1} encode failed")

    # --- write masked video ---
    const_fill = FILLS[fill_name]
    fill = None if const_fill is None else np.array(const_fill, dtype=np.float32)
    # Noise fill draws from the same hash chain, so it stays reproducible under --seed
    # and distinct across parallel workers.
    noise_rng = unit_rng(run_seed, stem, -1, 0) if fill is None else None
    ext = "mkv" if lossless else "mp4"
    masked_path = out_dir / f"{stem}_masked.{ext}"
    keep_audio = (not no_audio) and has_audio(video)
    cw = open_color_writer(masked_path, W, H, fps, video, lossless, keep_audio)
    union_alpha = alpha_all.max(axis=0).astype(np.float32) / 255.0
    palette = [p["rgb"] for p in labels.get("palette", [])]
    for t in range(T):
        if fill_name in ("ranked", "random"):
            # Composite low rank first so rank 1 wins contested pixels, matching the
            # painting order in subject_mask.py. "ranked" uses the labels palette,
            # "random" the per-unit colour drawn above.
            out = frames[t].astype(np.float32)
            for r in reversed(range(K)):
                if not alpha_all[r, t].any():
                    continue
                a = alpha_all[r, t].astype(np.float32)[:, :, None] / 255.0
                if fill_name == "random":
                    rgb = colour_by_unit.get((int(shot_of[t]), r), (255, 255, 255))
                else:
                    rgb = palette[r] if r < len(palette) else (255, 255, 255)
                out = out * (1.0 - a) + np.array(rgb, dtype=np.float32) * a
        else:
            a = union_alpha[t][:, :, None]
            f = (noise_rng.integers(0, 256, size=(H, W, 3)).astype(np.float32)
                 if fill is None else fill)
            out = frames[t].astype(np.float32) * (1.0 - a) + f * a
        cw.stdin.write(np.ascontiguousarray(np.round(out), dtype=np.uint8).tobytes())
    cw.stdin.close()
    if cw.wait() != 0:
        raise SystemExit(f"{stem}: masked video encode failed")
    written = count_frames(masked_path)
    if written != T:
        raise SystemExit(f"{stem}: masked video has {written} frames, expected {T}")

    if save_npz:
        np.savez_compressed(out_dir / f"{stem}_alpha.npz", alpha=alpha_all)

    record = {
        "video": {"path": video, "width": W, "height": H, "fps": fps, "num_frames": T},
        "run_seed": run_seed,
        "settings": {"work_res": work_res, "max_growth_scale": max_growth,
                     "shape_pool": list(SHAPES), "shape_weights": list(SHAPE_WEIGHTS),
                     "shape_caps": dict(SHAPE_CAPS),
                     "min_comp_frac": min_comp_frac,
                     "fill": fill_name, "fill_rgb": list(const_fill) if const_fill else None,
                     "lossless": lossless},
        "sampling_ranges": {k: list(v) for k, v in RANGES.items()},
        "notes": [
            "alpha==1 core is a strict superset of the original mask (subject hidden).",
            "alpha==1 core differs from the original mask, so thresholding alpha at 1.0 "
            "does not recover the silhouette.",
            "Parameters are drawn per (video, shot, rank) from a hash of "
            "run_seed|stem|shot|rank, so parallel workers never collide.",
        ],
        "units": [
            {"shot_index": sh, "rank": r + 1,
             "shape": shape_by_unit[(sh, r)],
             "shape_cap": round(SHAPE_CAPS[shape_by_unit[(sh, r)]] * max_growth, 4),
             "fill_rgb": list(colour_by_unit[(sh, r)]) if fill_name == "random" else None,
             "params": p,
             "mean_growth": round(float(np.mean(ratios[r])), 4) if ratios[r] else None,
             "max_growth_seen": round(float(np.max(ratios[r])), 4) if ratios[r] else None}
            for (sh, r), p in sorted(params_by_unit.items())
        ],
        "perturb_ms_per_rank_frame": round(perturb_ms, 3),
        "rank_frames_processed": rank_frames,
    }
    (out_dir / f"{stem}_perturb.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    return {"stem": stem, "rank_frames": rank_frames, "ms": perturb_ms,
            "masked": str(masked_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks-dir", required=True,
                    help="directory holding <stem>_masks.npz and <stem>_labels.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="limit to these stems (default: every one found)")
    ap.add_argument("--seed", type=int, default=None,
                    help="replay a whole batch bit-for-bit; omitted = fresh randomness")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--work-res", type=int, default=512,
                    help="longest side the perturbation is computed at")
    ap.add_argument("--max-growth", type=float, default=1.0,
                    help="multiplier on the per-shape area caps "
                         f"({SHAPE_CAPS}). The shape itself is the floor - this "
                         "cannot push the core below it")
    ap.add_argument("--min-comp-frac", type=float, default=0.02,
                    help="drop connected components smaller than this fraction of "
                         "the largest one (SAM speckle)")
    ap.add_argument("--fill", choices=sorted(FILLS), default="random")
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    masks_dir = pathlib.Path(args.masks_dir).expanduser()
    stems = args.stems or sorted(p.name[:-len("_masks.npz")]
                                 for p in masks_dir.glob("*_masks.npz"))
    if not stems:
        raise SystemExit(f"no *_masks.npz under {masks_dir}")

    run_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    print(f"[seed] run_seed={run_seed}"
          f"{' (from --seed)' if args.seed is not None else ' (random)'}")
    print(f"[batch] {len(stems)} video(s), {args.workers} worker(s)")

    job = dict(masks_dir=str(masks_dir), out_dir=str(pathlib.Path(args.out).expanduser()),
               run_seed=run_seed, work_res=args.work_res, max_growth=args.max_growth,
               fill_name=args.fill, lossless=args.lossless, save_npz=args.save_npz,
               no_audio=args.no_audio, min_comp_frac=args.min_comp_frac)

    t0 = time.perf_counter()
    if args.workers <= 1:
        for s in stems:
            r = process_video(s, **job)
            print(f"[done] {r['stem']} {r['rank_frames']} rank-frames "
                  f"{r['ms']:.2f} ms/rank-frame")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_video, s, **job): s for s in stems}
            for fut in as_completed(futs):
                r = fut.result()
                print(f"[done] {r['stem']} {r['rank_frames']} rank-frames "
                      f"{r['ms']:.2f} ms/rank-frame")
    print(f"[batch] finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
