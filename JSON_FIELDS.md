# JSON field reference

Two JSON products, written by two different stages. Both sit next to the video they
describe, named `<stem>_labels.json` and `<stem>_perturb.json`.

```
subject_mask.py   -> <stem>_labels.json   + <stem>_masks.npz   + <stem>_labeled.mp4
mask_perturb.py   -> <stem>_perturb.json  + <stem>_alpha_rank<N>.mkv + <stem>_masked.mp4
```

Conventions that hold in **both** files:

| Convention | Value |
|---|---|
| Frame indices | **global**, 0-based into the source video (never shot-relative) |
| Frame ranges | **inclusive** on both ends (`start_frame`..`end_frame`) |
| Boxes | `[x0, y0, x1, y1]`, pixels, top-left to bottom-right, in `video.width/height` space |
| Colours | `[R, G, B]`, 0-255 (RGB order, **not** BGR) |
| `rank` | 1 = most important subject. `obj_id == rank`; npz first axis index is `rank - 1` |
| Angles | radians |

---

## `<stem>_labels.json` — segmentation stage

### `video` — source clip specs

| Field | Type | Meaning |
|---|---|---|
| `path` | str | Absolute path of the source video |
| `width`, `height` | int | Pixel dimensions. All boxes and areas live in this space |
| `fps` | float | Frame rate |
| `num_frames` | int | Frames actually decoded. The labelled output has exactly this many |
| `duration_sec` | float | Duration in seconds |

### `models` — provenance

| Field | Type | Meaning |
|---|---|---|
| `detector` | str | Grounding DINO directory used to find subjects |
| `segmenter` | str | SAM 2 directory used to produce pixels |

### `params` — everything needed to reproduce the run

| Field | Type | Meaning |
|---|---|---|
| `max_subjects` | int | Cap on ranked subjects per shot |
| `box_threshold` | float | Detector confidence floor; below this is not a detection |
| `text_threshold` | float | Token-match floor; decides which words form `text_label` |
| `nms_iou` | float | Class-agnostic NMS IoU. Two boxes above this are judged the same object |
| `min_rel_score` | float | Rank 2+ scoring below this fraction of rank 1 is dropped. Rank 1 always kept |
| `keyframe_stride` | int | Detector runs every N frames; SAM 2 covers the gaps |
| `recheck_stride` | int | Every N frames the propagated mask is compared against a fresh detection |
| `iou_reseed` | float | If that comparison falls below this IoU, the box is re-injected |
| `scene_threshold` | float | ffmpeg scene-cut sensitivity; lower cuts more often |
| `close_kernel` | int | Morphological closing kernel, fills small holes |
| `prompt` | str \| null | `null` = auto-rank from the built-in vocabulary; otherwise the phrase given |

### `palette` — rank to colour map

List of `{rank: int, name: str, rgb: [int,int,int]}`.
`name` is `primary` / `secondary` / `tertiary`.

### `notes`

Free-text caveats that travel with the data (frame indexing, per-shot rank scoping).

### `shots[]`

| Field | Type | Meaning |
|---|---|---|
| `shot_index` | int | 0-based shot number |
| `start_frame`, `end_frame` | int | **Inclusive** global frame range |
| `subjects` | list | Ranked subjects in this shot |

### `shots[].subjects[]`

| Field | Type | Meaning |
|---|---|---|
| `rank` | int | 1 = main subject |
| `rank_name` | str | `primary` / `secondary` / `tertiary` |
| `rgb` | [int,int,int] | Colour painted in `_labeled.mp4` |
| `obj_id` | int | SAM 2 tracking id. Equals `rank`; npz index is `obj_id - 1` |
| `text_label` | str | Open-vocabulary phrase from the detector. May be multi-word (`"person man"`) or a fallback noun (`"object"`) |
| `det_score` | float | **Detector confidence only** — how sure it is that something of that phrase is there |
| `subject_score` | float | **The ranking criterion**: `det_score^0.5 x area_ratio^0.6 x centrality`. A confident but small off-centre object has high `det_score` and low `subject_score` |
| `seed_frame` | int | Frame where the box was handed to SAM 2. Propagation runs both directions from here, so this is often not frame 0 |
| `seed_box_xyxy` | [float x4] | The **detector's** box at `seed_frame`. Fixed for the shot. May extend a pixel or two outside the frame; masks are clipped |
| `first_frame`, `last_frame` | int \| null | First/last frame with a non-empty mask. `null` if never present |
| `num_present_frames` | int | Count of non-empty frames. Lower than `last-first+1` means it vanished mid-shot |
| `mean_area_ratio` | float | Mean screen fraction **over present frames only** (absent frames are not in the denominator) |
| `per_frame` | list | One row per frame of the shot |

### `shots[].subjects[].per_frame[]`

| Field | Type | Meaning |
|---|---|---|
| `frame` | int | Global frame index |
| `present` | bool | Whether the mask is non-empty on this frame |
| `bbox_xyxy` | [float x4] \| null | **Tight box of the actual mask** — computed from pixels, unlike `seed_box_xyxy` which came from the detector. `null` when `present` is false |
| `area_px` | int | Mask pixel count. Equals `masks[rank-1, frame].sum()` exactly |
| `area_ratio` | float | `area_px / (width * height)`, 6 decimals |

---

## `<stem>_perturb.json` — convex-blur stage

### `video`

Same fields as above minus `duration_sec`.

### `run_seed`

| Field | Type | Meaning |
|---|---|---|
| `run_seed` | int | Root of the whole batch's randomness. Per-unit parameters come from `blake2b(run_seed \| stem \| shot_index \| rank)`, so workers never collide and passing this back via `--seed` replays the batch bit for bit |

### `settings`

| Field | Type | Meaning |
|---|---|---|
| `work_res` | int | Longest side the perturbation was computed at before upscaling |
| `max_growth` | float | Hard cap on `core_area / mask_area`. Exceeding it bisects the dilation down |
| `fill` | str | `grey` \| `black` \| `white` \| `noise` \| `ranked` |
| `fill_rgb` | [int,int,int] \| null | The constant fill colour. `null` for `noise` and `ranked`, which have no single colour |
| `lossless` | bool | Whether `_masked` is ffv1/mkv rather than h264/mp4 |

### `sampling_ranges`

The `[min, max]` each parameter was drawn from. Mirrors the `RANGES` table in
`mask_perturb.py`; recorded so a file stays interpretable after the code moves on.

### `notes`

The two invariants and the seed-derivation rule, carried with the data.

### `units[]`

One entry per `(shot_index, rank)` — the granularity at which parameters are drawn.

| Field | Type | Meaning |
|---|---|---|
| `shot_index` | int | Matches `shots[].shot_index` in `_labels.json` |
| `rank` | int | Matches `subjects[].rank` in `_labels.json` |
| `params` | object | The draw (below) |
| `mean_growth` | float | Mean achieved `core_area / mask_area` over the unit's frames |
| `max_growth_seen` | float | Worst case. Sitting at `settings.max_growth` means the cap bound and dilation was rolled back |

### `units[].params`

These are **base values**; each frame adds the drift term, so the effective value at
frame `t` is `base * (1 + drift_amp * sin(2*pi*t / drift_period_frames + phase))`.

| Field | Type | Meaning |
|---|---|---|
| `convex` | float | Blend toward the convex hull. 0 = plain dilation, 1 = dilated hull |
| `dilate` | float | Core growth as a fraction of `sqrt(mask_area)` |
| `feather` | float | Alpha ramp width, same units. Ramp lives entirely in the grown region |
| `aniso` | float | Stretch factor; 1.0 = isotropic blob |
| `aniso_angle` | float | Stretch direction, radians |
| `drift_amp` | float | Relative amplitude of the temporal wobble |
| `drift_period_frames` | float | Wobble period **in frames** (already converted from seconds) |
| `drift_phase` | [float x3] | Phase offsets for `convex`, `dilate`, `feather` respectively — in that order |

### Timing

| Field | Type | Meaning |
|---|---|---|
| `perturb_ms_per_rank_frame` | float | Mean ms per rank-frame, excluding encode. Recorded every run so batch regressions surface |
| `rank_frames_processed` | int | Number of (rank, frame) pairs with a non-empty mask |

---

## Joining the two files

`shot_index` and `rank` are the join keys.

```python
labels  = json.load(open(f"{stem}_labels.json"))
perturb = json.load(open(f"{stem}_perturb.json"))
masks   = np.load(f"{stem}_masks.npz")["masks"]    # (K, T, H, W) bool, sharp
alpha   = np.load(f"{stem}_alpha.npz")["alpha"]    # (K, T, H, W) uint8, blurred

subj = labels["shots"][0]["subjects"][0]           # rank 1
unit = next(u for u in perturb["units"]
            if u["shot_index"] == 0 and u["rank"] == subj["rank"])

r, f = subj["rank"] - 1, 55
assert masks[r, f].sum() == subj["per_frame"][f]["area_px"]   # exact
assert np.all((alpha[r, f] >= 255) | ~masks[r, f])            # core covers subject
```
