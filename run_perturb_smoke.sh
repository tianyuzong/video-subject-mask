#!/usr/bin/env bash
# Verify the convex-blur invariants on real perturbed output.
# Usage: ./run_perturb_smoke.sh [masks_dir] [out_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASKS="${1:-$HERE/outputs/subject_mask}"
OUT="${2:-$HERE/outputs/perturb_smoke}"

if [[ ! -d "$MASKS" ]] || ! compgen -G "$MASKS/*_masks.npz" > /dev/null; then
  echo "usage: $0 <dir with *_masks.npz from subject_mask.py> [out_dir]" >&2
  exit 2
fi

echo "== run A (seed 4242) =="
python3 "$HERE/mask_perturb.py" --masks-dir "$MASKS" --out "$OUT/a" \
  --seed 4242 --workers 4 --save-npz --lossless --fill grey

echo "== run B (seed 4242, serial) - must match A bit for bit =="
python3 "$HERE/mask_perturb.py" --masks-dir "$MASKS" --out "$OUT/b" \
  --seed 4242 --workers 1 --save-npz --lossless --fill grey

echo "== run C (seed 9999) - must differ from A =="
python3 "$HERE/mask_perturb.py" --masks-dir "$MASKS" --out "$OUT/c" \
  --seed 9999 --workers 4 --save-npz --lossless --fill grey

echo "== run D (default random fill) - per-unit colours =="
python3 "$HERE/mask_perturb.py" --masks-dir "$MASKS" --out "$OUT/d" \
  --seed 4242 --workers 4 --save-npz --lossless

python3 - "$MASKS" "$OUT" <<'PY'
import sys, json, pathlib, itertools
import cv2, numpy as np

masks_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
stems = sorted(p.name[:-len("_masks.npz")] for p in masks_dir.glob("*_masks.npz"))

def solidity(m):
    m = m.astype(np.uint8)
    if not m.any():
        return None
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(c))
    ha = cv2.contourArea(hull)
    return float(m.sum()) / ha if ha > 0 else None

def per_component_solidity(m):
    """Each blob must be convex on its own. A global solidity would look bad simply
    because separate blobs sit apart, which is intended, not a defect."""
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in c:
        ha = cv2.contourArea(cv2.convexHull(cnt))
        if ha > 8:
            out.append(cv2.contourArea(cnt) / ha)
    return out

def n_components(m, min_frac=0.02):
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return 0
    areas = [cv2.contourArea(x) for x in c]
    big = max(areas) if areas else 0
    return sum(1 for a in areas if big <= 0 or a >= min_frac * big)

def hull_growth(m):
    """Bare per-component hull over mask area - the floor the cap cannot beat."""
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return 1.0
    img = np.zeros(m.shape, np.uint8)
    for cnt in c:
        cv2.fillConvexPoly(img, cv2.convexHull(cnt), 1)
    return img.sum() / max(m.sum(), 1)

import collections
tot = dict(frames=0, superset=0, differs=0, growth_ok=0, comps_ok=0)
sol_before, sol_after, ious, smooth_default, avg_sol = [], [], [], [], []
comp_sol, growths = [], []
by_shape = collections.defaultdict(lambda: {"n": 0, "enc": 0, "g": []})
shape_count = collections.Counter()
colour_gaps, all_colours = [], []
fill_ok = fill_tot = 0

for stem in stems:
    masks = np.load(masks_dir / f"{stem}_masks.npz")["masks"]
    alpha = np.load(out / "a" / f"{stem}_alpha.npz")["alpha"]
    rec = json.loads((out / "a" / f"{stem}_perturb.json").read_text())
    K, T, H, W = masks.shape
    assert alpha.shape == masks.shape, f"{stem}: alpha shape {alpha.shape} != {masks.shape}"
    caps = rec["settings"]["shape_caps"]
    scale = rec["settings"]["max_growth_scale"]
    unit_shape = {(u["shot_index"], u["rank"] - 1): u["shape"] for u in rec["units"]}
    shot_of = {}
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    for sh in labels["shots"]:
        for fr in range(sh["start_frame"], sh["end_frame"] + 1):
            shot_of[fr] = sh["shot_index"]
    for u in rec["units"]:
        shape_count[u["shape"]] += 1
    # colours inside one shot must be far enough apart to tell subjects apart.
    # Read them from run D: run A uses a constant grey fill so it records none.
    recd = json.loads((out / "d" / f"{stem}_perturb.json").read_text())
    by_shot = collections.defaultdict(list)
    for u in recd["units"]:
        if u.get("fill_rgb"):
            by_shot[u["shot_index"]].append(tuple(u["fill_rgb"]))
            all_colours.append(tuple(u["fill_rgb"]))
    for cols in by_shot.values():
        for i in range(len(cols)):
            for jx in range(i + 1, len(cols)):
                colour_gaps.append(float(np.linalg.norm(
                    np.array(cols[i]) - np.array(cols[jx]))))

    for r in range(K):
        for t in range(T):
            m = masks[r, t]
            if not m.any():
                continue
            core = alpha[r, t] >= 255
            tot["frames"] += 1
            # 1. superset: no subject pixel escapes the fully-opaque core
            tot["superset"] += bool(np.all(core | ~m))
            # 2. core must not equal the mask, else alpha>=1.0 thresholding leaks it
            tot["differs"] += bool(not np.array_equal(core, m))
            inter = np.logical_and(core, m).sum()
            ious.append(inter / max(np.logical_or(core, m).sum(), 1))
            # 7. growth cap. The hull is the semantic floor, so the real bound is
            #    max(cap, hull_growth) - the cap cannot push below the hull.
            g = core.sum() / m.sum()
            growths.append(g)
            kind = unit_shape.get((shot_of.get(t, 0), r), "hull")
            cap = caps[kind] * scale
            tot["growth_ok"] += bool(g <= max(cap, hull_growth(m)) + 0.05)
            bs = by_shape[kind]
            bs["n"] += 1
            bs["enc"] += bool(np.all(core | ~m))
            bs["g"].append(g)
            # 3b. every blob convex on its own
            comp_sol += per_component_solidity(core)
            # 3c. must not have collapsed into a single global hull
            tot["comps_ok"] += bool(n_components(core) >= min(n_components(m), 1))
        # 3. convexity actually improved
        idx = [t for t in range(T) if masks[r, t].any()]
        if idx:
            mid = idx[len(idx) // 2]
            sb, sa = solidity(masks[r, mid]), solidity(alpha[r, mid] >= 255)
            if sb and sa:
                sol_before.append(sb); sol_after.append(sa)
            # 6. temporal mean must stay blobby (averaging attack)
            mean_core = (alpha[r][idx].astype(np.float32).mean(axis=0) >= 128)
            s = solidity(mean_core)
            if s:
                avg_sol.append(s)
            # 5. temporal smoothness of the default (drifting) mode
            d = [np.abs(alpha[r, idx[i]].astype(int) - alpha[r, idx[i-1]].astype(int)).mean()
                 for i in range(1, min(len(idx), 40))]
            if d:
                smooth_default.append(np.mean(d))

    # 4. subject invisible in the masked video (constant fills only; a noise fill has
    #    no single expected value, and its own leak check would be a different test)
    if rec["settings"]["fill_rgb"] is None:
        continue
    cap = cv2.VideoCapture(str(out / "a" / f"{stem}_masked.mkv"))
    fill = np.array(rec["settings"]["fill_rgb"][::-1], dtype=np.uint8)  # cv2 is BGR
    for t in range(T):
        ok, fr = cap.read()
        if not ok:
            break
        union = masks[:, t].any(axis=0)
        if not union.any():
            continue
        fill_tot += 1
        fill_ok += bool(np.all(fr[union] == fill))
    cap.release()

# 4b. random-fill run: each rank's region must carry that unit's own colour
rand_ok = rand_tot = 0
for stem in stems:
    masks = np.load(masks_dir / f"{stem}_masks.npz")["masks"]
    rec = json.loads((out / "d" / f"{stem}_perturb.json").read_text())
    alpha_d = np.load(out / "d" / f"{stem}_alpha.npz")["alpha"]
    cols = {(u["shot_index"], u["rank"] - 1): u["fill_rgb"] for u in rec["units"]}
    labels = json.loads((masks_dir / f"{stem}_labels.json").read_text())
    shot_of = {}
    for sh in labels["shots"]:
        for fr in range(sh["start_frame"], sh["end_frame"] + 1):
            shot_of[fr] = sh["shot_index"]
    cap = cv2.VideoCapture(str(out / "d" / f"{stem}_masked.mkv"))
    for t in range(masks.shape[1]):
        ok, fr = cap.read()
        if not ok:
            break
        for r in range(masks.shape[0]):
            m = masks[r, t]
            rgb = cols.get((shot_of.get(t, 0), r))
            if not m.any() or rgb is None:
                continue
            # a better rank overpaints via its alpha, which reaches beyond its
            # mask - excluding only its mask would flag correct pixels as wrong
            better = np.zeros_like(m)
            for rr in range(r):
                better |= alpha_d[rr, t] > 0
            vis = m & ~better
            if not vis.any():
                continue
            rand_tot += 1
            rand_ok += bool(np.all(fr[vis] == np.array(rgb[::-1], dtype=np.uint8)))
    cap.release()

# 9/10. determinism, parallel safety, seed sensitivity
same = diff = 0
for stem in stems:
    a = np.load(out / "a" / f"{stem}_alpha.npz")["alpha"]
    b = np.load(out / "b" / f"{stem}_alpha.npz")["alpha"]
    c = np.load(out / "c" / f"{stem}_alpha.npz")["alpha"]
    same += np.array_equal(a, b)
    diff += not np.array_equal(a, c)

# 10. parameter draws must be pairwise distinct across units
sigs = []
for stem in stems:
    for u in json.loads((out / "a" / f"{stem}_perturb.json").read_text())["units"]:
        sigs.append((stem, u["shot_index"], u["rank"],
                     round(u["params"]["shape_jitter"], 9), round(u["params"]["dilate"], 9)))
param_sets = {s[3:] for s in sigs}

n = tot["frames"]
g = np.array(growths)
print(f"\nrank-frames checked:                  {n}")
print(f"1. core superset of mask:             {tot['superset']}/{n}")
print(f"2. core differs from mask:            {tot['differs']}/{n}")
print(f"   core-vs-mask IoU mean:             {np.mean(ious):.3f}  (1.0 would mean no blurring)")
print(f"3. solidity before -> after:          {np.mean(sol_before):.3f} -> {np.mean(sol_after):.3f}")
print(f"   per-blob solidity (convexity):     {np.mean(comp_sol):.3f}  min {np.min(comp_sol):.3f}")
print(f"   blobs kept separate (no merge):    {tot['comps_ok']}/{n}")
print(f"4. masked video shows fill on subject:{fill_ok}/{fill_tot}")
print(f"   random fill: rank region == its colour: {rand_ok}/{rand_tot}")
print(f"5. mean |alpha_t - alpha_t-1|:        {np.mean(smooth_default):.3f} / 255")
print(f"6. solidity of time-averaged core:    {np.mean(avg_sol):.3f}  (high = averaging attack fails)")
print(f"7. growth within max(cap, hull):      {tot['growth_ok']}/{n}")
print(f"   growth mean {g.mean():.3f}  p90 {np.percentile(g,90):.3f}  "
      f"p99 {np.percentile(g,99):.3f}  max {g.max():.3f}")
print(f"9. same seed reproduces:              {same}/{len(stems)}")
print(f"   parallel(4) == serial(1):          {same}/{len(stems)}")
print(f"   different seed differs:            {diff}/{len(stems)}")
print(f"10. distinct parameter draws:         {len(param_sets)}/{len(sigs)}")
print("11. per-shape enclosure and growth:")
for k in sorted(by_shape):
    b = by_shape[k]; ga = np.array(b["g"])
    print(f"    {k:9s} n={b['n']:4d} encloses {b['enc']}/{b['n']}  "
          f"growth mean {ga.mean():.3f} p90 {np.percentile(ga,90):.3f} "
          f"p99 {np.percentile(ga,99):.3f} max {ga.max():.3f}")
print(f"    shape draws: {dict(shape_count)}")
if colour_gaps:
    print(f"12. min colour gap within a shot:     {min(colour_gaps):.1f} (need >= 90)")
print(f"    distinct colours across units:    {len(set(all_colours))}/{len(all_colours)}")

assert tot["superset"] == n, "FAIL: subject pixels escape the opaque core"
assert tot["differs"] == n, "FAIL: core equals mask, silhouette recoverable"
assert tot["growth_ok"] == n, "FAIL: growth exceeded max(cap, hull)"
assert tot["comps_ok"] == n, "FAIL: blobs merged - degenerated into a global hull"
assert np.mean(comp_sol) >= 0.95, f"FAIL: blobs not convex enough ({np.mean(comp_sol):.3f})"
assert fill_ok == fill_tot, "FAIL: subject visible in masked video"
assert rand_tot and rand_ok == rand_tot, "FAIL: random fill colour mismatch"
assert same == len(stems), "FAIL: not reproducible / parallel differs from serial"
assert diff == len(stems), "FAIL: different seed produced identical output"
assert len(param_sets) == len(sigs), "FAIL: parameter collision across units"
assert np.mean(sol_after) > np.mean(sol_before), "FAIL: convexity did not improve"
for k, b in by_shape.items():
    assert b["enc"] == b["n"], f"FAIL: {k} does not enclose the mask "\
                               f"({b['enc']}/{b['n']})"
assert len(shape_count) >= 2, f"FAIL: shape pool barely used: {dict(shape_count)}"
if colour_gaps:
    assert min(colour_gaps) >= 90, f"FAIL: colours too close ({min(colour_gaps):.1f})"
assert len(set(all_colours)) == len(all_colours), "FAIL: duplicate colours across units"
print("\nperturb smoke test PASSED")
PY
