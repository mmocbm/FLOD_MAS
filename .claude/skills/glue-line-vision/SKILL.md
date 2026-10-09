---
name: glue-line-vision
description: Measured facts and tested approaches for detecting and measuring the glue line on fabric in this project, especially clear glue on white fabric. Load before changing segmentation, measurement, capture or lighting code, before choosing or training a vision model for the glue line, or when judging whether a detection result is correct.
---

# Glue-line vision: what is known

Everything here was measured on the sample frames in `enhancement_images/` unless marked
otherwise. Re-measure before relying on a number for a new product or camera position.

## The target

- The glue is the **thin glossy bead**: a narrow bright line with a thin dark line along each
  edge. It is **not** the wider pale band that runs beside it. The user confirmed this from the
  images; an earlier detector outlined the band and was wrong.
- Width at full resolution: about 15-21 px on the two-panel product (`10_05/0036-0052`), about
  12 px on the four-panel product (`WIN_*.jpeg`). Length 1300-2000 px, S-curved.
- On white fabric the bead differs from the fabric beside it by 1-2 gray levels on average.
  The usable signal is its gloss and relief, not its colour.
- The only defect the user wants graded is **width out of range** (target +/- tolerance).
- Nothing has been checked against a ruler or caliper. Pixel widths are consistent between
  frames; their accuracy in millimetres is unknown.

## The images

- Frames are 4000x3000 PNG or 4608x3456 JPEG. Dark navy fabric: `10_05/0001-0035`. White
  fabric: `10_05/0036, 0037, 0041-0052` and the two `WIN_*` files. `0038-0040` are a different
  dark moulded part.
- White frames are soft: Laplacian sharpness 3-6 against 21-27 for dark frames. They also show
  blocky compression artifacts. Exposure drifts between captures.
- Translucent white mesh lies over part of each panel and produces moire. Beads under mesh are
  the ones most often missed.

## Image shape

- **Never resize the whole frame for the glue line.** At 1024 px wide the bead is 2-5 px. This
  is why the shipped cloud workflow returns 0 panels and 0 strips on white fabric.
- Work at native resolution: the full frame for the classical detector, or 1024x1024 tiles with
  overlap for a model.

## What has been tried, and how it did

| Approach | Result on white fabric |
|---|---|
| Roboflow `custom-workflow` (panel exemplars + SAM3), frame resized to 1024x768 | 0 panels, 0 strips, with or without contrast enhancement |
| Cloud SAM3 text prompt on the resized frame | Only yellow tape. With work-area crop + CLAHE: all 4 strips on one frame, but masks 38-59 px wide where the bead is about 12 |
| Cloud SAM3 on native 1024 tiles, prompt `thin glossy line` | Finds beads at about the right width (14-19 px) but misses some, fragments others, and catches straight panel seams. Prompts `glue line` and `clear adhesive bead` returned nothing |
| `segmentation/glue_line.py` (classical, no model) | 50 of about 64 beads on 16 white frames, most 1200-1950 px long, widths repeatable; 7-30 s per frame on CPU; reports nothing on dark fabric by design |
| U-Net (ResNet34 encoder) on native 512 tiles with classical pseudo-labels | Training was started on the RTX 3050 and interrupted at 1000 of 2500 iterations. **No result yet.** Scripts are in `~/flod_experiments/` |

Counts of beads "present" were made by eye, not from labels.

## The line's shape: an omega

- The user's rule, confirmed by measurement: the glue line always has the same wave, two humps
  on one side of the chord joining its ends with a dip between them ("W, like omega"). Mirror
  images occur.
- 19 whole lines of the two-panel product agree with their mean to 0.74 % of the chord (14 px
  in 1863). In chord-frame units: hump 0.082, dip about 0.03.
- Most "S-shaped" partial detections were a hump and a half of that omega, cut short where the
  edges faded. A view of `gray - GaussianBlur(gray, 25)` scaled x7 shows every bead clearly
  along its full length; use it to judge results by eye.
- `segmentation.glue_line.shape_prior` (opt-in) fits the omega to pieces of evidence, rejects
  curves of another shape, follows the bead inside a 30 px corridor, and measures width along
  the whole line. On the 16 white frames: 50 whole lines, width measured on 62-97 % of each.
- Lessons: a partial piece fixes the line sideways but lets it slide along its own length, so
  the coarse template match must be optimised exactly (Nelder-Mead over size and start), the
  size range kept tight (0.90-1.10), and the proposing piece required to stay on the line. A
  line lying across the others is moire, not glue.
- Only `two_panel` is in `segmentation/glue_shapes.json`. The four-panel frames gave one whole
  line, too few to learn from; that product is matched with the two-panel omega.
- `tools/learn_glue_shape.py NAME frames...` learns a template for a new product or height.

## How the classical detector works

See the module docstring in `segmentation/glue_line.py`. In short: flatten lighting, run an
elongated second-derivative line filter in 12 directions keeping dark and bright responses
separately, restrict to pale fabric away from its outline, then at every dark-line pixel look
across for a second dark line running the same way with a bright line between. Midpoints form
the centreline; the distance is the width.

Lessons that cost time:
- Pairing must be done point by point. Tracing edges as curves fails because the two edges
  join at the bead's end and trace as one U-shaped curve.
- Two dark lines with nothing brighter between them are folds, not glue. The check compares the
  centre against fabric sampled outside both edges.
- Weave and moire can be chained into a strip-shaped object only by bridging long gaps, so the
  share of length actually seen (`min_seen_share`) is what rejects them.

## Models

Research summary (sources are in the session export in the repo root):
- Most segmentation models output masks at 1/4 to 1/16 resolution, too coarse for a 10-20 px
  bead measured to 1-2 px. Mask R-CNN and YOLACT++ are unsuitable; Mask2Former is overkill.
- Best fit once labels exist: a plain U-Net (nnU-Net recipe) on native-resolution tiles, used
  to **locate** the bead, with width still measured on raw pixels.
- Labels generated by the classical detector or by SAM carry their width bias. Width ground
  truth must come from a physical measurement.
- Dice/IoU do not indicate width accuracy. Evaluate width error in pixels or millimetres.
- Learned super-resolution or deblurring must not be used before measuring: it can invent or
  shift the edges being measured.
- `facebook/sam3` and `briaai/RMBG-2.0` are gated on Hugging Face; a logged-in account that has
  accepted the licence is needed. An RMBG-2.0 research-and-test task was started and did not
  finish; partial files are in `~/flod_experiments/rmbg/`.

## Capture and hardware (researched, not bench-tested here)

Ranked for this setup: UV-fluorescent glue with UV light and a lens filter (if the glue can be
changed); low-angle dark-field lighting; several directional lights fired in turn (photometric
stereo); polarisation difference; 3D laser profiler; thermal camera for hot-melt glue. Lock
focus and exposure, reduce compression, and put more pixels on the bead. `capture.controls` in
`config.json` already exposes exposure, gain and white balance; values there are placeholders.

Unknown and worth asking the user: how the glue is applied (hand or machine, hot-melt or not)
and whether the glue type can change.

## Experiment environment

- `~/flod_experiments/gpu/bin/python`: Python 3.12, CUDA PyTorch, transformers,
  segmentation-models-pytorch, OpenCV, scikit-image, inference-sdk. GPU is an RTX 3050 with
  6 GB.
- The system Python is 3.14, which `inference-sdk` does not support.
- `tools/replay_capture.py` replays image files through gate, segmentation and measurement with
  `--set section.key=value` overrides and no camera.

## Checking a result

Do not declare a detection correct from an overlay alone. Open a contrast-boosted close-up
beside the marked one and ask the user to confirm: they can see the glue and have corrected a
confident wrong call before.

## Outlining with SAM 3 (2026-10-08)

Full record, numbers and method: `GLUE_LINE_FINDINGS.md` in the repo root. In short:

- Working chain: panel ROI model -> rough line per ROI region -> SAM 3 point prompts on
  native 1024 tiles -> leak repair (`sam_refine.trim_ring`) inside the ROI. On 18 M1 frames
  all 62 lines were outlined; share of the outline that is SAM's own edge: dark 1.00, white
  0.95, pink 0.86 (worst pink line 0.56).
- No filter before SAM 3 or before the local detector improves the result on any colour.
  Contrast gain and flatten make SAM outline a 55-70 px band (bead plus the pale band beside
  it) instead of the about 30 px object it outlines on dark fabric. Enlarging tiles
  fragments the outline.
- Pale fabric is helped by asking SAM again with other tiles and points and merging the
  steady parts, and by judging each edge alone. Not by a filter.
- Width from SAM is about 30 px on every colour; the local detector's is 12-17 px. Which is
  the glue's width is still open and needs caliper data.

## Best strategy so far (version 4, 2026-10-08)

The user judged this the best result to date. Read *Strategies* in `GLUE_LINE_FINDINGS.md`
before changing any of it. Chain: panel ROI model -> rough line -> SAM 3 point prompts on raw
full-resolution tiles (leak repair, ask again up to twice) -> state of the picture place by
place (`bead_edges.measure`: shiny, matte, one-sided, weak, none) -> band edges from the
straightened strip as evidence -> one smooth even-width ribbon fitted to both
(`segmentation/ribbon.py`). Rules that cost time to learn: never filter SAM's input; never
draw a classical edge finder's edges directly; report a width deviation only when both
sources agree; the user's eye decides, not an overlay that looks plausible. Soft white
frames from `10_05` are still not solved.

## Accuracy is now measurable (2026-10-08)

Do not judge a change to edges or width by eye or against another method first: run
`tools/accuracy_bench.py` (painted glue lines of known edges on real fabric) and check repeat
shots. Measured: steepest-slope edges are unbiased and about 1 px RMS; the trained trees
(`segmentation/bead_ml.py`) cut that by a third and remove the bias from a core along one
edge and from blur; glue fills the weave (texture inside the band is 0.2-0.4 of the
fabric's) and that helps as a feature, not as an edge. Width defects: smallest found reliably
is about 8 px over 60 px; on real lines the rule alone gives too many single-shot reports.
Averaging repeat shots does not help. Full tables: `GLUE_LINE_FINDINGS.md`, *Accuracy*.

## Ground facts from the user (2026-10-08)

Normal glue strip width is 4 mm; the camera is not calibrated; **the supplied strips have no
defects**. So: the ~30 px band is the 4 mm strip (about 7.6 px/mm, uncalibrated); every
width deviation reported on the supplied images is a false alarm (use them to measure the
false-alarm rate, never as proof of detection); band edges alone flag 40-87 % of good lines,
only the rule needing SAM's agreement is usable (1 of 257 lines at threshold 7); whether it
catches a real defect is untested. Details: `GLUE_LINE_FINDINGS.md`, *Ground facts from the
user*.

## SegFormer (2026-10-09)

A SegFormer (B0 or B2) fine-tuned on the version 5 ribbons, run on native 512 px tiles,
marks every glue line as one piece with no prompts or rough line (78 of 78 held-out lines,
nothing stray), in about 1 s per frame, and is the first method to handle the soft white
frames from `10_05` / `real_captures` (widths 28-32 px, 1 of 17 lines doubtful; SAM: 22-36
px, 9 of 17). Its output is at quarter resolution: use the mask as evidence for the ribbon,
not as the edge. CLAHE before it does not help and hurts on soft frames. Its licence is
non-commercial (NVIDIA). Models and scripts: `~/flod_experiments/segformer/`. Not wired into
the tool. Tables: `GLUE_LINE_FINDINGS.md`, *SegFormer and CLAHE*.
