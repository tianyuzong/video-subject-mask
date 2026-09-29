#!/usr/bin/env python3
"""Render every figure the README references, into docs/img/.

Run after a segmentation pass (subject_mask.py --save-masks) and a perturbation pass.
Each figure is self-contained: source frames are pulled from the paths recorded in the
labels JSON, so the script works from any checkout as long as those videos exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mask_perturb import SHAPE_CAPS, SHAPES, _perturb_frame  # noqa: E402

YEL, GRN, RED = (0, 255, 255), (0, 255, 0), (0, 0, 255)


def label(img, text, colour=YEL, y=0, scale=0.5):
    w = int(11 * scale / 0.5 * len(text))
    cv2.rectangle(img, (0, y), (min(img.shape[1], w), y + 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)
    return img


def frames_at(path, picks):
    cap = cv2.VideoCapture(str(path))
    out, i, hi = {}, 0, max(picks)
    while i <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if i in picks:
            out[i] = f
        i += 1
    cap.release()
    return out


def fig_shapes(masks_dir, out_dir, stem, frame):
    """One subject through every shape primitive in the pool."""
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    masks = np.load(masks_dir / f"{stem}_masks.npz")["masks"]
    m = masks[0, frame]
    tiles = [label(np.dstack([m.astype(np.uint8) * 255] * 3), "ORIGINAL MASK", RED)]
    for kind in SHAPES:
        alpha, ratio = _perturb_frame(m, kind, 0.02, 0.02, 0.03, 1.0, 0.0, 512,
                                      SHAPE_CAPS[kind], 0.02, 7)
        tiles.append(label(np.dstack([(alpha * 255).astype(np.uint8)] * 3),
                           f"{kind}  x{ratio:.2f}", GRN))
    panel = np.hstack([cv2.resize(t, (260, 347)) for t in tiles])
    cv2.imwrite(str(out_dir / "shapes.png"), panel)
    del labels
    return "shapes.png"


def fig_pipeline(masks_dir, perturb_dir, out_dir, stem, picks, size):
    """Source -> sharp ranked mask -> perturbed fill, across time."""
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    rows = [("SOURCE", frames_at(labels["video"]["path"], picks)),
            ("STAGE 1  sharp ranked mask", frames_at(masks_dir / f"{stem}_labeled.mp4", picks)),
            ("STAGE 2  shape + random colour", frames_at(perturb_dir / f"{stem}_masked.mp4", picks))]
    strips = []
    for name, got in rows:
        if len(got) < len(picks):
            continue
        strip = np.hstack([cv2.resize(got[p], size) for p in picks])
        strips.append(label(strip, name))
    cv2.imwrite(str(out_dir / f"pipeline_{stem}.png"), np.vstack(strips))
    return f"pipeline_{stem}.png"


def fig_seeds(masks_dir, seed_dirs, out_dir, stem, frame, size):
    """Same clip, different seeds: different shapes and colours every run."""
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    src = cv2.resize(frames_at(labels["video"]["path"], [frame])[frame], size)
    tiles = [label(src, "SOURCE")]
    for seed, d in seed_dirs:
        rec = json.loads((d / f"{stem}_perturb.json").read_text())
        shapes = "+".join(u["shape"] for u in rec["units"])
        got = frames_at(d / f"{stem}_masked.mp4", [frame])
        if frame not in got:
            continue
        tiles.append(label(cv2.resize(got[frame], size), f"seed {seed}: {shapes}",
                           scale=0.42))
    cv2.imwrite(str(out_dir / f"seeds_{stem}.png"), np.hstack(tiles))
    return f"seeds_{stem}.png"


def fig_alpha(masks_dir, perturb_dir, out_dir, stem, frame):
    """Mask -> opaque core -> soft alpha, per rank, with solidity."""
    masks = np.load(masks_dir / f"{stem}_masks.npz")["masks"]
    alpha = np.load(perturb_dir / f"{stem}_alpha.npz")["alpha"]
    rec = json.loads((perturb_dir / f"{stem}_perturb.json").read_text())
    shape_of = {u["rank"] - 1: u["shape"] for u in rec["units"]}

    def solidity(mm):
        c, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        vals = [cv2.contourArea(x) / max(cv2.contourArea(cv2.convexHull(x)), 1)
                for x in c if cv2.contourArea(cv2.convexHull(x)) > 8]
        return float(np.mean(vals)) if vals else 0.0

    cols = []
    for r in range(masks.shape[0]):
        m = masks[r, frame]
        if not m.any():
            continue
        core = alpha[r, frame] >= 255
        stack = [np.dstack([m.astype(np.uint8) * 255] * 3),
                 np.dstack([core.astype(np.uint8) * 255] * 3),
                 np.dstack([alpha[r, frame]] * 3)]
        col = np.vstack([cv2.resize(x, (300, 200)) for x in stack])
        label(col, f"rank{r+1} {shape_of.get(r,'?')} "
                   f"sol {solidity(m):.2f}->{solidity(core):.2f}", RED, scale=0.45)
        cols.append(col)
    if not cols:
        return None
    panel = np.hstack(cols)
    for i, t in enumerate(["mask", "alpha==1 core", "soft alpha"]):
        label(panel, t, GRN, y=200 * i + 176, scale=0.45)
    cv2.imwrite(str(out_dir / f"alpha_{stem}.png"), panel)
    return f"alpha_{stem}.png"


def fig_fills(masks_dir, fill_dirs, out_dir, stem, frame, size):
    """Every fill mode side by side."""
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    tiles = [label(cv2.resize(frames_at(labels["video"]["path"], [frame])[frame], size),
                   "SOURCE")]
    for name, d in fill_dirs:
        got = frames_at(d / f"{stem}_masked.mp4", [frame])
        if frame not in got:
            continue
        tiles.append(label(cv2.resize(got[frame], size), name, scale=0.45))
    cv2.imwrite(str(out_dir / "fills.png"), np.hstack(tiles))
    return "fills.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--perturb-dir", required=True)
    ap.add_argument("--seed-dirs", nargs="*", default=[],
                    help="seed=dir pairs, e.g. 11=/path/seed_11")
    ap.add_argument("--fill-dirs", nargs="*", default=[],
                    help="name=dir pairs, e.g. grey=/path/fill_grey")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    md = pathlib.Path(args.masks_dir).expanduser()
    pd = pathlib.Path(args.perturb_dir).expanduser()
    od = pathlib.Path(args.out).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    seeds = [(s.split("=")[0], pathlib.Path(s.split("=", 1)[1])) for s in args.seed_dirs]
    fills = [(s.split("=")[0], pathlib.Path(s.split("=", 1)[1])) for s in args.fill_dirs]

    made = []
    made.append(fig_shapes(md, od, "video", 140))
    made.append(fig_pipeline(md, pd, od, "video", [20, 140, 260], (260, 347)))
    made.append(fig_pipeline(md, pd, od, "0kpu6VM3rZU.5", [10, 50, 90], (340, 191)))
    made.append(fig_alpha(md, pd, od, "0kpu6VM3rZU.5", 50))
    made.append(fig_alpha(md, pd, od, "video", 140))
    if seeds:
        made.append(fig_seeds(md, seeds, od, "video", 140, (250, 333)))
        made.append(fig_seeds(md, seeds, od, "0kpu6VM3rZU.5", 50, (320, 180)))
    if fills:
        made.append(fig_fills(md, fills, od, "video", 140, (250, 333)))
    for m in made:
        if m:
            print("wrote", od / m)


if __name__ == "__main__":
    main()
