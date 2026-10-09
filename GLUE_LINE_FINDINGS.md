# Glue-line detection: findings and methodology

What has been measured about finding and outlining the glue line on dark, pink and white
fabric, how each experiment was run, and what is still unverified. State as of 2026-10-08,
branch `tooling/pipeline-lab`, nothing committed. Start with *Strategies*; the sections after
it hold the measurements behind each statement, in the order the work was done.

How to use the tool is in `AUTOMATIC_INSPECTION.md` (section *Pipeline Lab*). This file
records why it works the way it does.

## Strategies

### The strategy in use (version 4, judged best so far by the user, 2026-10-08)

Each stage does one job, and no stage is trusted for what it is bad at.

| Stage | Job | Why this and not something else |
|---|---|---|
| 1. Panel ROI model | Say where glue can be: one region per panel strip | The glue is always inside it; everything outside (mesh, cardboard) is the source of false lines |
| 2. Rough line (local detector) | Say roughly where each line runs, on the original pixels | Needs only to be within some 20 px; cheap, no upload |
| 3. SAM 3, point prompts, full-resolution tiles | Say which object the line is | SAM picks the right object far more reliably than any classical rule; never on a reduced frame, never on a filtered one |
| 4. Leak repair + asking again | Keep only SAM's steady edges; ask up to twice more with other tiles | On pale fabric SAM spills in places; it spills elsewhere each time, so the answers are merged |
| 5. Look place by place | Name what the picture is like along the line: shiny, matte, one-sided, weak, none | The same line is shiny in one place and not in the next; which evidence to trust depends on it |
| 6. Band edges on the straightened strip | Read the dark band's two edges after averaging along the line | The only filter that makes the 5-level band measurable on white; used as evidence, not as the answer |
| 7. Smooth ribbon | Fit one smooth curve of even width to both kinds of evidence, weighted by stage 5 | The real line is smooth and even; drawn sample by sample, either source is jagged or notched |
| 8. Ends and deviations | End where the band ends; report stretches where both sources disagree with the ribbon | Ends from SAM alone were 47 px off on white; a real width change must not be smoothed away |

In the tool: ROI "Panel model", no pre-processing steps, "SAM 3 - points placed
automatically", tick "Smooth ribbon from the true shape". Code: `tools/pipeline_steps.py`
(`segment_sam_from_lines`, `finish_as_ribbon`), `segmentation/sam_refine.py`,
`segmentation/bead_edges.py`, `segmentation/ribbon.py`.

### Candidate next strategy (version 6: trained segmenter, researched, not in the tool)

| Stage | Job | Against version 5 |
|---|---|---|
| 1. Panel ROI model | Where glue can be | same |
| 2. SegFormer-B2 on native 512 px tiles over the ROI | Mark the glue band, every line at once, no prompts | replaces SAM 2.1 and its point prompts; about 1 s instead of 3-4.5 s |
| 3. Rough line | Give each line's course for reading the mask | same for now; the mask itself could give it (not done) |
| 4. Leak repair, states, band edges + trees, ribbon, ends, defect rule | As in version 5, with the mask as the outline evidence | same |
| Fallback | SAM 2.1 where the trained model gives no mask or has not seen the material | new |

Why it is a candidate: it is the first method that reads the soft white frames (1 of 17
lines doubtful against 9 of 17), and it is faster. Why it is not adopted: its licence is
non-commercial, it must be trained per product, it has been tried on six new-piece frames
only, and the user has not judged it. Details in *SegFormer and CLAHE*.

### Versions and the user's verdict

| Version | Outline comes from | User's verdict |
|---|---|---|
| 1 | SAM 3, each answer accepted or thrown away whole | Dark good; pale mostly thrown away |
| 2 | SAM 3 + leak repair + asking again | Good; black perfect; white still jagged, ends too short or too long |
| 3 | v2 + band edges drawn as the outline | Worse than v2: steps, notches, wrong ends |
| 4 | v2 + band edges as evidence + smooth ribbon | Best so far |
| 5 | v4 on the local GPU + trained-trees edges + noise-scaled defect rule | not yet judged by the user |
| 6 (candidate, not built into the tool) | Trained SegFormer-B2 as the outline source in place of SAM, same trees and ribbon | not yet judged by the user |

### Strategies that work

- **Full resolution always.** A reduced frame leaves the bead 2-5 px wide.
- **Confine everything to the ROI**, line by line, one region at a time.
- **Prompt SAM with points along a known line**, not with text and not with a whole frame.
- **Ask again instead of filtering.** Other tiles and point spacings; merge the steady parts.
- **Judge each edge alone.** A spill moves one edge and leaves the other.
- **Average along the line's own direction**, never across it, to bring out the band.
- **Treat edges as evidence with weights**, and let what the picture is like at each place
  set the weights.
- **Use the known shape**: smooth curve, even width. Fit to it; report what departs from it.
- **Take the reference from black fabric**, where the line is seen perfectly: how regular
  lines are, how even the width is, where the ends lie.
- **Check with measures the method cannot satisfy by construction**: agreement in shape
  between lines, end positions, the dark control. Then the user's eye.

### Strategies that do not work (measured)

- **Any filter as SAM 3's input**: CLAHE, flatten, bead enhance, contrast gain, denoise,
  unsharp mask, negative-kernel sharpen, negative, quantise, shine compress, dimming outside
  the ROI, smoothing along the line, and combinations; also choosing the filter per tile.
  Equal at best. Anything that raises contrast makes SAM outline a wider band (55-70 px).
- **Enlarging** tiles 2x or 3x: the outline fragments.
- **Tighter "not this" points** beside the line: no reliable gain.
- **A whole-image filter before the local detector**: 34 of 40 panels with or without.
- **Drawing a classical edge finder's edges as the outline**: notches where the shiny core
  runs along one edge.
- **Reporting a width deviation from one source alone**: 17 false alarms on dark fabric.
- **Using the black-fabric wave as a template for other panel cuts**: white and pink panels
  are a different, longer cut.

### Accuracy (see *Accuracy*)

- **Measure error before changing anything**: paint glue lines of known edges onto real
  fabric from the same frame; check precision on repeat shots of the same piece.
- **Correct the edge with trained trees** (gradient boosting on the profile around the
  edge in lightness, core-removed lightness and weave energy): a third less noise, no bias
  from a core along one edge or from blur.
- **Use the weave**: glue fills it. As a feature it helps; as an edge on its own it cannot
  be placed.
- **Judge width defects against a running median, scaled by the line's own noise**, and
  only where both SAM's and the band's edges show them.
- Does not work: sub-pixel peak fitting, phase congruency, coarser scales, averaging shots,
  shorter averaging along the line.

### Speed (see *Local GPU flow*)

- **Run SAM on the PC's own GPU, in half precision, one encode per tile.** 0.3 s per tile
  against 0.7-0.9 s per cloud call, and nothing is uploaded.
- **Ask each line once** when the ribbon is on; the band edges cover what SAM missed.
- **Do not compute what the next stage does not read**: the rough line spent 90 % of its
  time on whole-frame filters the region-guided search never used.
- **ROI model on the GPU** through its PyTorch checkpoint.
- Does not work: MobileSAM and FastSAM on pale fabric; batching tiles; full precision.

### Trained segmenter (see *SegFormer and CLAHE*)

- **A SegFormer trained on the version 5 ribbons** finds every line as one piece without a
  rough line or prompts, in about 1 s, and is the first method that copes with the soft
  white frames. Its mask is evidence for the ribbon like SAM's outline. Not in the tool yet;
  its licence is non-commercial.
- CLAHE before it: no gain, worse on soft frames.

### Still open

- **Whether a real width defect is caught at all** (see *Ground facts from the user*): no
  defective sample exists, and the usable rule may be blind where SAM's edge is repaired.
- **Camera calibration** for millimetres: widths read 7 % differently by position.

- Soft white frames from other sets (`10_05`, some `real_captures`): SAM outlines the wider
  band there (36-50 px) and the ribbon follows it. Marked CHECK BY EYE, not solved.
- Width in millimetres, and whether the 30 px band is the glue's full width: no caliper data.
- Pink is the least certain colour; white has only 6 frames in the M1 set.
- Whether the smallest real width defect survives the ribbon's smoothing: only a 200 px
  bulge has been tested (synthetic).

## The object

- The glue is a thin glossy bead: a bright core with a dark line along each side. It runs
  along the curved edge of each panel strip, always inside the ROI the panel model returns.
- Every line has the same wave (two humps and a dip, "omega"); mirror images occur.
- A frame holds two to four lines, one per panel strip.
- Lightness profile across the line, averaged along it (levels relative to the fabric):

  | Fabric | Dark sides | Bright core | Fabric lightness |
  |---|---|---|---|
  | Dark | -52 | +38 | low |
  | White | -4 to -5 | +4 | high |
  | Pink | -4 to -6 (one band, weak core) | about 0 | high |

  On pale fabric the signal is a tenth of what it is on dark fabric.
- No colour channel separates the glue on pale fabric better than lightness. Measured
  separation (mean step / spread along the line) on pink: lightness 2.6-3.4, saturation
  0.2-0.6, Lab a 0.3, Lab b 0.5-0.8. On dark fabric saturation is the strongest (2.3).
- Beside the bead on pale fabric lies a paler smooth band, 30-40 px wide, between the bead
  and the mesh. It is not the bead. Methods that outline 55-70 px have taken bead and band
  together.
- Brown cardboard is not glue. A colour rule for cardboard must also require brightness
  (V > 100), because dark mesh lying over cardboard is warm and saturated too.

## Images used

| Set | Path | Content |
|---|---|---|
| M1 captures | `enhancement_images/M1_Masks/captures` | 74 frames 4608x3456: dark, pink, white, two to four panels |
| M1 masks | `enhancement_images/M1_Masks/full_mask` | hand masks of the panel strips; rotated 180 degrees relative to the captures |
| 10_05 | `enhancement_images/10_05` | 4000x3000: `0001-0035` navy, `0036, 0037, 0041-0052` white, `0038-0040` another part |
| Real captures | `enhancement_images/real_captures` | 6 webcam frames |
| Loose files | `enhancement_images/*.jpeg` | 4 frames, two of them small |

White `10_05` frames are soft (Laplacian sharpness 1-6 against 20-25 for navy and 90-600 for
M1 captures) and show compression blocks. Frame `0036` (sharpness 1.4) gives no line with
any method tried.

## Method in use

1. **ROI.** `model_for_segment_bra_roi places/.../strip_unet_resnet34.onnx` (U-Net,
   ResNet-34, input 1152 px wide, mirrored second pass) marks the panel strips; cardboard is
   removed from the mask. Needs `onnxruntime`; `cv2.dnn` cannot load it.
2. **Rough line.** `segmentation/glue_line.py` in region-guided mode looks along the curved
   side of each ROI region, on the original picture at full resolution. It gives the line's
   position, not a trustworthy outline.
3. **SAM 3.** Points every 150 px along the rough line are sent with 1024x1024 tiles of the
   original pixels (`segmentation/sam_refine.py`, Roboflow serverless API). If the answer is
   far wider than a bead it is asked again with "not this" points 45 px to each side.
4. **Leak repair** (`sam_refine.trim_ring`). (Since the update above, step 6 below decides
   the final outline; this step gives SAM's own outline, shown thin beside it.)

   Details: The mask is cut to the ROI and read as a ribbon:
   at samples 4 px apart along the line, how far the mask reaches to each side.
   - The line's steady course is taken from stretches of normal width (running median over
     31 samples).
   - Each edge is then judged alone: within 5 px of its course it is kept, otherwise it is
     replaced by the course. A spill moves one edge and leaves the other.
   - A line outlined on less than 95 % of its edges is put to SAM again, up to twice, with
     tile stride / point spacing 560/110 and 1000/200 px. The answers spill in different
     places; per sample the steady ones are combined.
   - A line SAM took too wide over most of its length is re-read against the width SAM gave
     the frame's evenly outlined lines.
5. **Reported per line:** length, width (median, 10th-90th percentile), fabric colour, and
   **"SAM outlined N %"**: the share of the two edges that is SAM's own. The rest is the
   carried-over course, where a locally narrow or wide spot would not show.

6. **Bead edges** (`segmentation/bead_edges.py`): the band's two edges read from the original
   picture along the line. Version 3 drew them as the outline (tick box, now off by default).
7. **Smooth ribbon** (`segmentation/ribbon.py`, tick box): version 4, see *Strategies* above
   and *Smooth ribbon* below.

All of this lives in `tools/pipeline_steps.py` (`segment_sam_from_lines`) and
`tools/pipeline_lab.py` (`run_job`). The inspection app is unchanged.

## Results

### Validation: 18 M1 frames, every fourth capture, not used for tuning

No pre-processing steps, ROI from the panel model, SAM 3 with automatic points.

| Fabric | Frames | Lines | Share SAM outlined, mean / worst | Lines at 80 % or more | Width px, median (range) | Seconds per frame |
|---|---|---|---|---|---|---|
| Dark | 8 | 29 | 1.00 / 0.96 | 29 | 31.6 (29-32) | 11-15 |
| Pink | 8 | 27 | 0.86 / 0.56 | 17 | 28.4 (26-32) | 15-53 |
| White | 2 | 6 | 0.95 / 0.76 | 5 | 32.8 (26-36) | 15-20 |

Before the leak repair the same SAM answers were accepted or thrown away whole: pink 1 of 4
and white 3 of 4 on the first frames tried, the rest falling back to the rough line.

### Leak repair, step by step

Ten hardest pink and white lines, same measure throughout (share of edges that is SAM's):

| Answers merged | Mean share |
|---|---|
| 1 | 0.79 |
| 2 | 0.87 |
| 3 | 0.88 |
| 4 | 0.90 |

Two extra questions are used; a fourth answer adds little. Examples: 0.86 -> 1.00,
0.67 -> 0.88, 0.64 -> 0.82. The lines that stay low (0.52 -> 0.56, 0.55 -> 0.73) are ones
where SAM takes bead and pale band together in every answer.

### Filters before SAM 3

Two frames per colour, seven lines each, measured before the per-edge change (so compare
within the table only):

| Filter | Pink: share / width px | White: share / width px |
|---|---|---|
| None | 0.68 / 28 | 0.96 / 32 |
| CLAHE (clip 4, grid 16) | 0.68 / 29 | 0.90 / 32 |
| Bead enhance (top-hat / black-hat) | 0.68 / 28 | 0.85 / 31 |
| Flatten lighting | 0.55 / 56 | 0.73 / 32 |
| Denoise + local contrast gain x6 | 0.46 / 60 | 0.61 / 32 |
| Local contrast gain x6, smoothing 3 | 0.88 / 67 | 0.83 / 58 |

- Contrast gain gives an even outline, but of a band 55-70 px wide, not the about 30 px
  object outlined on dark fabric. It is not the same segment, so it is not used.
- An earlier sweep of 14 recipes (also denoise, sharpen, quantise, shine compress, dim
  outside ROI and combinations) under the older accept-or-refuse rule: none accepted more
  lines than the plain image; quantising and denoise + sharpen accepted fewer.
- Enlarging tiles 2x and 3x: outlines fragment; 0 of 10 hard lines accepted.
- Tighter "not this" points (22 and 30 px instead of 45): no reliable gain.

### Sharpening and negative kernel before SAM 3, white fabric

Four white frames, 12 lines, SAM 3 with repeat asking. "Edge distance" is the mean distance
of SAM's two edges from the bead's own edges (next section), in pixels.

| Filter | Share SAM outlined (mean / worst) | Edge distance, mean | Worst line |
|---|---|---|---|
| None | 0.96 / 0.76 | 2.8 | 5.9 |
| Denoise + negative-kernel sharpen (strength 1, reach 1) | 0.92 / 0.70 | 3.0 | 6.1 |
| Denoise + negative-kernel sharpen (2, reach 3) | 0.88 / 0.61 | 3.5 | 8.4 |
| Denoise + negative-kernel sharpen (1, reach 6) | 0.92 / 0.53 | 3.2 | 5.9 |
| Denoise + unsharp mask (2, radius 3) | 0.95 / 0.74 | 3.0 | 5.1 |
| Negative (inverted picture) | 0.92 / 0.61 | 3.6 | 12.4 |
| Negative + denoise + sharpen | 0.91 / 0.68 | 3.0 | 6.7 |
| Denoise + sharpen + negative | 0.92 / 0.69 | 3.1 | 6.8 |
| CLAHE + denoise + sharpen | 0.90 / 0.69 | 2.8 | 6.4 |

No filter brings SAM's outline closer to the bead than the plain picture. The steps are in
the tool (*Sharpen (negative kernel)*, *Negative*) for trying by hand.

### The bead's own edges (`segmentation/bead_edges.py`)

What the picture shows once it is straightened along the line and averaged along it: the
bead is a band darker than the fabric on both sides, with crisp edges along its whole
length, on every colour. Its shiny core and the thin black lines beside the core lie inside
the band. The band is 30-50 levels darker than dark fabric and 4-7 levels darker than pink
or white fabric; averaging 42 px along the line is what makes the 5-level step measurable.

Method: straighten a strip 70 px to each side of the line; average along the line; remove
the thin black lines (closing, 7 px across) and the shiny core (opening, 21 px across; a
core lying at the band's edge is counted as bead); follow the dark stripe along the line (it
wanders up to some 15 px off the rough line); at each sample take the steepest darkening on
one side and the steepest brightening on the other as the two edges; lengthen the line by
120 px at each end and cut it where the stripe stops being dark. No model, no upload, about
0.1 s per frame.

All 74 M1 captures, rough line from the local detector, no SAM:

| Fabric | Frames | Lines | Edges found | Width px, median (5th-95th pct of lines) | Spread within a line, median | Edges on steady course |
|---|---|---|---|---|---|---|
| Dark | 34 | 126 | 126 | 32.0 (30.4-32.8) | 4 | 0.96 |
| Pink | 34 | 113 | 113 | 30.4 (24.6-34.0) | 11 | 0.98 |
| White | 6 | 18 | 18 | 34.0 (31.8-35.2) | 4 | 0.99 |

- On dark fabric, where SAM 3 is reliable, the two methods agree: width 32.0 against 31.6,
  centre lines 2.2 px apart (median over 29 lines). That is the check that the edge finder
  measures the same object.
- On white frame `230306_169300` SAM's line 1 was 36 px wide and off the bead; the edges are
  35 px wide and on it, and the outline now ends at the bead's tips.
- Pink is the weak spot: on some lines the core runs along one edge of the band and the
  fabric beyond is as bright as the core's side, so the edge alternates between the core's
  far side (about 20 px) and a faint outer boundary (about 28 px). Width within such a line
  varies by 11 px although the bead probably does not.
- Eye checks: frame `230306` (3 lines, crops at both ends and the middle), strips of one
  pink line. Everything else by numbers.

### Smooth ribbon (`segmentation/ribbon.py`, `bead_edges.measure`)

**Reference from black fabric** (`tools/learn_bead_reference.py`, stored in
`segmentation/bead_reference.json` as `m1_panel`; 124 lines from 34 dark frames):

| Property | Value on black fabric |
|---|---|
| Wave of the centre line | lines agree with their mean to 0.42 % of the chord (about 8 px in 1800) |
| Width | 31.3 px, 1.0 px between lines; even along the line to 1 % except at the tips |
| Ends | 71 px and 86 px short of where the line would leave its ROI region (spread 5 and 8 px) |
| Edge wobble of SAM's outline | 0.68 px RMS |

White and pink panels in the M1 set are a different, longer cut (lines about 2130 and 2490
px against 1917), so the black wave is not their wave: their residual against it (1.3 % and
2.8 % of the chord) measures the product, not an error. What does carry over is how
*regular* lines are: agreement among lines of one product, even width, end gaps.

**What the picture is like, place by place** (`bead_edges.measure`, per 2 px sample, settled
over 22 px, drawn as dots beside each line):

| State | Read from the straightened strip | Share on white / pink / dark (4 sample frames) | Edge evidence trusted |
|---|---|---|---|
| shiny | core at least 8 levels over the band | about 60-70 % / 55-75 % / 85-95 % | SAM 1.0, band edges 1.0 |
| matte | band, no core | 25-30 % / 20-45 % / 5-15 % | SAM 1.0, band edges 1.0 |
| one-sided | core within 6 px of a band edge | 0-11 % / 1-27 % / 1-3 % | SAM 1.0, band edges 0.15 |
| weak | band under 60 % of the line's usual darkness | 1-6 % / 0-4 % / 0-1 % | SAM 0.6, band edges 0.4 |
| none | band under 30 % | ends and gaps | SAM 0.3, band edges 0 |

`one-sided` is where the edge finder made its notches; there it is nearly ignored.

**Fit.** Evidence that disagrees with its neighbours loses weight over three rounds; the
centre line is smoothed as a curve in the picture (local cubic over about 400 px, which
keeps the omega's humps); the width is one value varying only over about 150 px. The band
tracker is kept within 22 px of SAM's outline (left free, it wandered onto a neighbouring
stripe on one pink line). Ends: where the band falls below 30 % of the line's darkness; the
value 0.3 was chosen because it gives the most constant end gaps on all three colours
(0.2, 0.3, 0.4 tried). A stretch where SAM's edges and the band's edges are both steadily
5 px or more wider or narrower than the ribbon for 60 px is returned as a width deviation
and drawn red, not smoothed away.

**Runs** (cached SAM 3 answers, so only the finishing differs). A = v2 (SAM 3 + leak
repair); B = ribbon from SAM's edges only; C = ribbon from SAM's and the band's edges.

| | White, 18 lines | | | Pink, 22 lines | | | Dark, 126 lines | | |
|---|---|---|---|---|---|---|---|---|---|
| | A | B | C | A | B | C | A | B | C |
| Edge wobble, px RMS | 1.04 | 0.12 | 0.00 | 1.10 | 0.12 | 0.01 | 0.68 | 0.03 | 0.00 |
| Width spread within a line, px | 12.1 | 8.0 | 3.6 | 14.4 | 11.7 | 5.2 | 4.1 | 2.8 | 2.5 |
| Lines of one colour agree in shape to (% of chord) | 0.70 | | 0.28 | 0.99 | | 0.88 | 0.47 | | 0.42 |
| End gap spread, near / far end, px | 47 / 7 | | 8 / 11 | 34 / 13 | | 9 / 17 | 17 / 10 | | 5 / 8 |
| End gap, near / far, px | 79 / 122 | | 77 / 96 | 115 / 143 | | 76 / 99 | 91 / 109 | | 71 / 86 |
| Ribbon backed by evidence | | 0.53 | 0.98 | | 0.53 | 0.92 | | 0.66 | 0.99 |
| Evidence kept lies within (median, px) | | 0.94 | 0.68 | | 0.97 | 0.76 | | 0.57 | 0.48 |
| Width, median px | 32.6 | 32.6 | 33.8 | 28.0 | 29.0 | 30.0 | 31.0 | 32.0 | 32.0 |

- C on white is as regular as black fabric: lines agree in shape to 0.28 % (black 0.42 %),
  and the end gaps are the same on all three colours (about 75 and 95 px), which v2's are
  not. Zero wobble is by construction and proves nothing by itself; the shape agreement and
  the end gaps are the independent checks.
- Dark control: C's centre line is within 0.5 px (median; worst line 0.7) of v2's.
- Pink mixes two panel cuts, which is why its shape agreement stays near 0.9 %.
- Width deviations reported: dark 0, white 0, pink 3 (all at 91-95 % of the line, 6-8 px
  narrower: the taper before a tip, not checked by eye). An earlier version that let either
  source alone raise a deviation reported 17 on dark, all false.

**Filters as SAM input** (white, 18 lines; "distance" = mean distance of SAM's own edges
from the ribbon C's edges):

| SAM input | Share SAM outlined (mean / worst) | Distance, px (mean / worst line) |
|---|---|---|
| Plain picture | 0.93 / 0.55 | 2.4 / 6.6 |
| Smooth along the line, 31 px, everywhere | 0.91 / 0.79 | 9.3 / 18.7 |
| Smooth along the line, 51 px | 0.93 / 0.80 | 10.8 / 18.6 |
| Contrast gain x2 + smooth along | 0.85 / 0.61 | 12.0 / 19.0 |
| Smooth along except where shiny (per tile) | 0.89 / 0.62 | 2.8 / 10.8 |
| Smooth along only where weak or one-sided (per tile) | 0.93 / 0.55 | 2.4 / 6.6 (rule never applied: no tile is mostly weak) |
| Shine compress where shiny, smooth along elsewhere | 0.90 / 0.76 | 9.5 / 17.0 |

A random search over the filter's settings (length 15-61, direction scale 3-10, contrast
gain 1-3, gain radius 20-80, shine knee 0-40; scored on 3 white frames, 9 lines) was stopped
after 5 of the planned 40 trials at about 5 minutes each: distances 2.8, 6.7, 9.4, 14.5 and
17.8 px. None beat the plain picture (2.4 px on all 18 lines); the best was the shortest
filter, which changes the picture least. The winner was not re-checked on the held-out
frames.

The oriented filter makes the band look clean and crisp to the eye and makes SAM outline a
different, wider object evenly. Same finding as for every other filter: SAM 3 is best on the
camera's own pixels. The filter stays in the tool as a step ("Smooth along the line").

**Other image sets, run once with A + C, not tuned on:**

| Frame | Result |
|---|---|
| `real_captures/WIN_..21_40_20` (sharp, pink) | 4 lines, width 30-32 px, backed 97-98 %, misfit 0.6 px |
| `10_05/0041, 0045, 0049` (soft white, other product) | 11 lines; widths 36-42 px, backed 71-92 %, misfit 1.9-3.5 px; one line refused (ribbon 632 px long, misfit 41 px), one with no band found |
| `real_captures/WIN_..21_30_28, ..21_43_32` (soft white) | 7 lines; widths 29-50 px, backed 56-92 %, misfit 2.1-4.8 px |

On the soft white frames the widths (36-50 px) are far above the 15-21 px measured for that
product earlier, so SAM is outlining the wider band there and the ribbon follows it. These
frames are **not solved**. The tool now marks a ribbon "CHECK BY EYE" when its misfit is
over 2 px or it is backed under 80 %, and keeps the earlier outline when the ribbon misfits
by more than 6 px or covers under 70 % of the line.

### Filters before the local detector (other agent's session, `session-ses_ee8c.md`)

32 frames (12 M1, 7 white, 3 navy, 10 others), local detector only, scored by counts and not
checked by eye. Scripts: `tools/pipeline_sweep.py`, `tools/roi_sweep.py`; output in
`~/flod_experiments/out/cristal_sweep` and `roi_sweep`.

| Run | M1 panels with a line | White frames with 3 or more lines |
|---|---|---|
| No filter | 34 of 40 | 6 of 7 |
| Flatten + bead enhance | 36 of 40 (with my ROI model) | 6 of 7 |
| Other recipes | 34 of 40 | 6 of 7 |

Flatten + bead enhance also adds fragments (up to 10-13 lines on a frame with two panels).

### Earlier findings that still hold

- **Never resize the whole frame.** At 1024 px wide the bead is 2-5 px; the shipped cloud
  workflow returns nothing on white fabric for that reason.
- SAM 3 with a text prompt finds some beads on native tiles (`thin glossy line`) but misses
  and fragments others and catches seams; point prompts along a known line are far better.
- SAM 2 with point prompts also works on dark fabric and is slower. The tool offers SAM 3
  only.
- The bead can lie about 24 px outside the edge of the first ROI model
  (`segmentation/models/glue_roi.onnx`) but is inside the user's model's ROI.
- Model research: a U-Net on native-resolution tiles is the best fit once labels exist;
  Mask R-CNN and YOLACT++ are too coarse. Learned super-resolution must not precede
  measuring. A bead U-Net training run and an RMBG-2.0 test were started and never finished.
- Hardware options, ranked, not bench-tested: UV-fluorescent glue with UV light and a filter;
  low-angle dark-field light; several lights fired in turn; polarisation; 3D profiler;
  thermal camera. For shine: crossed polarisers, a minimum over several lights, HDR, or a
  diffuse dome.

## Time per frame (version 4 operation flow)

Measured 2026-10-08 on the development PC (Intel i5-13420H, 12 threads, 14 GB RAM; the ROI
model ran on CPU because only the CPU build of onnxruntime is installed; office internet
connection to the Roboflow API). Three frames per colour, 4608x3456, through the tool's own
`run_job` with every stage timed inside it. Recipe: ROI "Panel model", no pre-processing
steps, SAM 3 with automatic points, smooth ribbon on.

### Flow and time, seconds per frame (mean of 3 frames)

| # | Stage | Where it runs | Dark | Pink | White |
|---|---|---|---|---|---|
| 1 | Read the image file | local | 0.08 | 0.12 | 0.09 |
| 2 | ROI: panel model, two passes | local, CPU | 2.07 | 2.05 | 2.38 |
| 3 | ROI: cardboard mask | local | 0.11 | 0.11 | 0.11 |
| 4 | Rough line (local detector) | local, CPU | 10.6 | 8.7 | 8.1 |
| 5 | SAM 3: tiles sent, answers read | Roboflow API | 8.8 | 33.0 | 11.8 |
| 6 | Leak repair | local | 0.10 | 0.41 | 0.15 |
| 7 | Place-by-place states + band edges | local | 0.15 | 0.19 | 0.13 |
| 8 | Ribbon fit | local | 0.06 | 0.07 | 0.05 |
| 9 | Width and length of each line | local | 0.07 | 0.09 | 0.06 |
| 10 | Drawing the overlay and the rest | local | 0.20 | 0.25 | 0.20 |
| | **Whole frame** | | **22.1** | **44.9** | **23.0** |
| | Range over the 3 frames | | 20.0-23.5 | 42.5-48.5 | 18.3-25.4 |

Also: the first run of a session took 29.4 s on a dark frame against about 22 s afterwards
(loading the ROI model, opening the API session). Encoding the overlay as JPEG is not in the
table.

### Where the time goes

| Share of the frame's time | Dark | Pink | White |
|---|---|---|---|
| SAM 3 (stage 5) | 40 % | 74 % | 51 % |
| Rough line (stage 4) | 48 % | 20 % | 35 % |
| ROI (stages 2-3) | 10 % | 5 % | 11 % |
| Everything version 4 added (stages 7-8) | 1 % | 0.6 % | 0.8 % |
| Leak repair, measuring, drawing | 2 % | 1.7 % | 1.8 % |

Two stages are 86-94 % of the time. What version 4 added over version 2 costs 0.2-0.3 s.

### SAM 3 in detail

| | Dark | Pink | White |
|---|---|---|---|
| Lines per frame | 4 | 3-4 | 3-4 |
| API calls per frame | 12 | 31-40 | 9-18 |
| Calls per line | 3 | 8-11 | 3-6 |
| Seconds per call, mean | 0.72 | 0.93 | 0.79 |
| Slowest call | 1.3 | 2.0 | 1.5 |
| Lines asked once / twice / three times (12, 11 and 10 lines) | 12 / 0 / 0 | 3 / 1 / 7 | 8 / 0 / 2 |

- One call is one 1024x1024 tile (about 375 kB as JPEG) with its prompt points. A line
  takes 2-4 tiles; a tile whose first answer is far too wide is asked a second time with
  "not this" points.
- Pink is slow because most of its lines are outlined on less than 95 % of their edges at
  the first asking and are asked twice more, which triples the calls.
- **Sending calls at the same time does not help.** The same 8 calls took 9.6 s one after
  another, 9.2 s four at a time and 9.4 s eight at a time, through the shared client.
  Whether the limit is in the client or at the service was not determined.

### Rough line in detail (one pink frame, 8.5 s)

| Part | Seconds | Works on |
|---|---|---|
| Line filter in 12 directions (`line_strength`) | 5.1 | whole frame |
| Lighting flattening (`flatten`) | 1.3 | whole frame |
| Audit image | 1.15 | whole frame |
| Looking along each ROI region (`_panel_lines`) | 0.86 | ROI regions only |

Only the last part is what the flow needs when the ROI model is used; the first three are
whole-frame work done before it.

### Where time could be saved (not done, not measured unless stated)

| Idea | Expected effect | Status |
|---|---|---|
| Do the rough line's whole-frame filters inside the ROI regions only, or skip those the region-guided path does not use | up to about 7 s per frame on every colour | not tried; needs a check that the lines found do not change |
| On pale fabric ask SAM once, and let the ribbon lean on the band edges for the rest | about 20 s per pink frame | not tried; needs a run comparing ribbons from one answer and from three |
| ROI model on the GPU | 2.1 s to about 0.35 s (the model's README, RTX 4060) | needs the GPU build of onnxruntime; not measured here |
| Several SAM calls at once | none | measured: no gain |
| Smaller or fewer tiles | unknown | not tried; fewer pixels per tile makes the bead narrower for SAM |

### Limits of these figures

- Three frames per colour, one session, one machine. The pink and white means would move
  with other frames (the white range is 18-25 s).
- API time depends on the connection and on the service's load at the time.
- The inspection PC is a lower-powered Windows machine: stages 2-4 and 6-10 will take
  longer there; stage 5 depends on its connection. Nothing was timed on it.
- This is the Pipeline Lab's flow. The inspection app's own cycle (`PERFORMANCE.md`) was not
  re-timed.

## Local GPU flow (offline)

Measured 2026-10-08 on the same PC: RTX 3050 Laptop GPU, 6 GB; PyTorch 2.11 with CUDA 12.8;
`transformers` 5.18; `ultralytics` 8.4. Aim: the version 4 result with nothing uploaded and
as little delay as possible. Quality is judged against the cloud version 4 ribbons of all 74
M1 frames (257 lines), which the user judged best so far; it is a comparison with that
result, not with ground truth.

### Result

| Flow | Seconds per frame: dark / pink / white | Uploads | GPU memory | Against cloud version 4 |
|---|---|---|---|---|
| Cloud version 4 (before) | 22.1 / 44.9 / 23.0 | every tile | - | reference |
| **Local SAM 2.1 large, ask once, ribbon** | **5.7 / 6.5 / 4.8** (9 frames); 4.3 / 4.7 / 3.4 (all 74) | none | 1.0 GB | centre line 0.1 / 0.5 / 0.4 px away; width 0.4 / 1.3 / 1.7 px narrower; 257 of 257 lines "ok" |
| Local SAM 2.1 tiny, ask once, ribbon | 3.1 / 3.3 / 2.6 | none | 0.6 GB | centre 0.2 / 0.6 / 0.5 px; width 0.3 / 1.2 / 1.1 px narrower; 4 pink lines "check" |
| No SAM: rough line + band edges + ribbon | 1.2 / 1.2 / 1.0 | none | 0.5 GB | centre 0.2 / 0.7 / 0.5 px; width 0.2-0.6 px wider; 19 pink lines "check", 2 lines more than 3 px off |

- Whole-frame time fell from 22-45 s to about 5 s with the same kind of result, and to
  about 1 s without SAM.
- The first run of a session takes 11-16 s (models loaded onto the GPU).
- The two timings of the recommended flow differ (9 frames timed alone; 74 frames timed
  while their overlays were being produced); the GPU's speed varies between runs. Take
  4-6 s as the figure.
- One pink line of the 257 lies more than 3 px (5.4 px) from its cloud version 4 ribbon, with
  either local SAM model; which of the two is right there was not checked.
- The local SAM 2.1 flow marks no line "check by eye"; cloud version 4 marked 16 on pink.
  Whether those 16 are now right, or only pass the checks, has not been judged by eye.
- Offline proof: one frame run with the network namespace removed (`unshare -n`),
  `HF_HUB_OFFLINE=1`, no API key in the environment and the cloud client replaced by one
  that raises. It completed: 3 lines, widths 33.6, 32.4 and 30.0 px.

### Where the time went, and what changed it

| Stage | Cloud flow, s (dark / pink / white) | Local flow, s | What changed |
|---|---|---|---|
| ROI model | 2.1 / 2.1 / 2.4 | 0.25 / 0.24 / 0.22 | PyTorch checkpoint on the GPU instead of ONNX on the CPU; masks agree (IoU 0.99998) |
| Rough line | 10.6 / 8.7 / 8.1 | 0.94 / 0.93 / 0.71 | Whole-frame filters skipped: the region-guided search never used them; lines identical (44 of 44) |
| SAM | 8.8 / 33.0 / 11.8 | 3.85 / 4.57 / 3.21 | Local model in half precision; each line asked once |
| Everything else | 0.7 / 1.1 / 0.7 | 0.7 / 0.8 / 0.6 | unchanged |

### What a local SAM costs (one 1024x1024 tile)

| Model | Encode, ms | Ask, ms | Peak GPU memory, MB | Note |
|---|---|---|---|---|
| SAM 2.1 large, half precision | 298 | about 10 | 743 | chosen |
| SAM 2.1 large, full precision | 1728 | | 3043 | 5.8 times slower |
| SAM 2.1 tiny, half precision | 66 | about 10 | 285 | |
| SAM 2.1 tiny, full precision | 183 | | 1679 | |
| MobileSAM (Ultralytics) | about 105 for the whole call | | 330 | |
| FastSAM-s / FastSAM-x (Ultralytics) | 46 / 241 for the whole call | | under 900 | |

- A tile is encoded once and may be asked again for about 10 ms, against 0.7-0.9 s per call
  in the cloud. The "not this" retry is therefore free.
- Half precision is the one setting that matters. Encoding 4 tiles as one batch took the
  same time per tile (292 against 298 ms). Preparing a tile on the CPU takes 10 ms.
- `torch.compile`, TensorRT and ONNX export were not tried: the encoder is now about 3 s of
  a 5 s frame, and the tiny model already brings that to 1 s.

### Which model (9 frames, asked up to three times, against cloud version 4)

| Model | Share SAM outlined: dark / pink / white | SAM's edges from the v4 ribbon, px | Final ribbon's centre from v4, px | Lines refused / "check" |
|---|---|---|---|---|
| SAM 2.1 large | 1.00 / 0.95 / 0.98 | 0.9 / 2.9 / 3.1 | 0.1 / 0.5 / 0.3 | 0 / 0 |
| SAM 2.1 base+ | 1.00 / 0.86 / 0.96 | 0.8 / 5.0 / 4.1 | 0.1 / 0.4 / 0.4 | 0 / 1 |
| SAM 2.1 small | 1.00 / 0.92 / 0.97 | 1.1 / 4.2 / 4.0 | 0.2 / 0.5 / 0.5 | 0 / 1 |
| SAM 2.1 tiny | 0.99 / 0.96 / 0.96 | 1.2 / 3.3 / 3.2 | 0.2 / 0.7 / 0.5 | 0 / 0 |
| MobileSAM (asked once) | 0.76 / 0.46 / 0.70 | 2.0 / 16.1 / 18.2 | 0.3 / 2.4 / 5.1 | 0 / 8 |
| FastSAM-x (asked once) | 0.26 / none / none | 4.5 / - / - | - | 6 of 11 dark lines refused; no answer on pink or white |
| FastSAM-s (asked once) | 0.06 / none / none | 7.0 / - / - | - | 9 of 11 dark lines refused; no answer on pink or white |

- **SAM 2.1 large is the closest to cloud SAM 3** and is taken as the default.
- **MobileSAM** works on dark fabric and takes far too much on pale fabric: on white its
  outline is 18 px from the bead and the final ribbon comes out 9 px too wide.
- **FastSAM** segments everything first and then picks by the points; it has no mask for
  the bead on pale fabric at all and for only half the lines on dark. Not usable here.
- MobileSAM and FastSAM come through the `ultralytics` package, which is AGPL-3.0.
- **SAM 3 locally was not tested**: the weights are gated and this PC has no Hugging Face
  login yet. The backend takes `sam3` as a name already.

### Asking once

With the ribbon, the second and third askings change nothing that reaches the result: SAM
2.1 large asked once gives the same final ribbon as asked three times (centre 0.4 against
0.5 px from v4 on pink, width the same) and pink's SAM stage falls from 9.6 to 4.5 s.

### Not done

- A small trained glue-line model in place of SAM (planned as E6): the flow without SAM
  already runs in about 1 s and the local SAM flow in about 5 s, so it was left out. It
  remains the way to a model that needs neither.
- SAM 3 on this GPU (needs the login); `torch.compile` / TensorRT.
- Nothing here was timed on the Windows inspection PC.

### How to use it

- Pipeline Lab: "SAM runs on: This PC's GPU" (the default when a CUDA GPU is found), local
  model `sam2.1-large`, "Ask SAM once per line" and "Smooth ribbon" ticked.
- `tools/benchmark_flow.py frames... --sam local|cloud|none --model NAME` re-measures the
  stage table on any machine.
- Needs `torch` with CUDA and `transformers` (SAM 2.1); `ultralytics` only for MobileSAM and
  FastSAM. The inspection app needs none of them.

## Accuracy (truth bench, advanced edge methods, traditional ML)

Until this section every result was judged against another method or by eye. Here error is
measured against a known truth, and candidate methods are kept only if they lower it.
2026-10-08, same PC. Code: `segmentation/synthetic_bead.py`, `tools/accuracy_bench.py`,
`segmentation/edge_methods.py`, `segmentation/bead_ml.py`, `tools/train_bead_ml.py`.

### What a real glue line's cross-section is (measured, 37 frames, 46 000 samples)

| | Dark | Pink | White |
|---|---|---|---|
| Band darker than fabric, grey levels | 43-50 | 7 | 7 |
| Blur of the band's edge, px (sigma) | 2.2 | 2.1-2.2 | 2.0-2.1 |
| Shiny core over the band, grey levels | 122 | 16 | 29 |
| Core's place when shiny | centre (spread 9 % of width) | centre (7 %) | centre (8 %) |
| **Weave texture inside the band, share of the fabric's** (matte stretches) | **0.21** | **0.24** | **0.39** |
| Width, px | 31-32 | 30-31 | 33-34 |

New fact: glue fills the weave. Where the band is only 7 levels darker than white fabric,
its texture is still 2.5 to 5 times weaker than the fabric's.

### The truth bench

- **Painted glue lines on real fabric.** A strip of plain fabric is cut from the same frame
  beside the real line (found for 69 of 69 lines tried), and a bead is painted into it with
  the measured depth, edge blur, core, dark side lines and texture loss, along a chosen
  course with a chosen width. Its edges are therefore known exactly. Conditions: standard;
  rough line 12 px off; core along one edge; half contrast; blurred by 2.5 and 4 px (towards
  the softness of the `10_05` frames); no shine; and width defects of 3, 5 and 8 px over 30
  and 60 px.
- **Repeat shots.** 14 bursts of 3-5 shots of the same layout (51 lines), still to 0.1-0.6
  px. The same line must give the same result in each shot.
- Synthetic truth is a model of the glue line. Passing it is necessary, not sufficient.

### How accurate the flow was (steepest-slope edges, as in version 4)

Standard condition, 12 fabric strips per colour; edge error is positive outside the truth.

| | Dark | Pink | White |
|---|---|---|---|
| Edge bias, px | +0.18 | +0.04 | +0.09 |
| Edge error per sample, px RMS | 1.15 | 1.10 | 2.82 |
| Edge error after the ribbon, px RMS | 1.25 | 1.19 | 1.02 |
| Width bias, px | +0.36 | +0.08 | +0.19 |
| Edge bias when blurred by 4 px | +0.73 | +0.51 | +0.88 |
| Core along one edge: edge bias / RMS | -0.30 / 1.94 | **-1.33 / 3.68** | -0.59 / 4.16 |
| Width defects found (3-8 px, 30-60 px long) | **2 of 192** over all colours | | |

- Edges are unbiased to a few tenths of a pixel and about 1 px RMS: the outline is where
  the band is.
- Blur widens the reading: +1.0 to +1.8 px of width at 4 px blur. Soft frames read wide.
- A core along one edge pulls that edge in by 1.3 px on pink: the notch problem, now with a
  number.
- **The width-defect rule found almost nothing** (2 of 192). The one defect the inspection
  is for was not being caught. The ribbon's part of the error is its smoothing: it reports
  an even width by design, so a painted defect counts as error.

### Candidate edge methods

Edge error per sample, px RMS (bias in brackets where it matters). First block: all frames,
8 strips per colour. Second block: frames held out from the trees' training (no white frame
fell in the held-out part).

| Method | Dark | Pink | White | Verdict |
|---|---|---|---|---|
| Steepest slope, whole pixels (baseline) | 1.10 | 1.06 | 1.48 | reference |
| Same, sub-pixel by parabola | 1.12 | 1.05 | 3.00 | no gain: noise, not pixel size, is the limit |
| Steepest slope at 3 px scale | 0.98 | 0.92 | 1.29 | small gain; worse when blurred |
| Phase congruency (log-Gabor, 3 scales) | 1.01 | 1.10 | 1.53 | equal to baseline |
| Weave-texture channel alone | 9.2 | 10.7 | 94 | **dropped**: the texture edge cannot be placed to a pixel |
| Lightness + texture fused | 0.74 | 1.15 | 2.05 | **dropped**: helps dark only, fails when blurred |
| Parametric profile fit (two blurred edges) | 0.72 | 0.92 | 1.20 | better; biased wide when blurred (+1.1 to +1.5 at 4 px) |

| Held-out frames | Baseline | Profile fit | **Trees (trained)** |
|---|---|---|---|
| Standard, dark / pink | 0.98 / 1.02 | 0.69 / 0.83 | **0.63 / 0.63** |
| Core along one edge, dark | 1.87 (bias -0.36) | 1.65 (-0.47) | **0.79 (0.00)** |
| Core along one edge, pink | 3.71 (bias -1.40) | 3.46 (-1.32) | **1.13 (-0.15)** |
| Half contrast, dark | 1.20 | 0.96 | **0.83** |
| Blurred 4 px, dark | 1.00 (bias +0.28) | 1.15 (+0.65) | **0.54 (-0.03)** |
| Half contrast or blurred 4 px, pink | 8.1 / 7.7 | 8.7 / 8.6 | 7.3 / 6.3 |

- **Trees** = gradient-boosted trees (LightGBM, 600 trees, 92 features) that correct the
  steepest-slope edge from the profile around it in three channels: lightness, lightness
  with the core removed, weave energy. Trained on painted lines from 52 frames with width,
  contrast, blur, shine and core position varied; tested on 22 later-and-earlier frames
  never seen: 2.21 -> 1.41 px RMS over those hard conditions. This is where the texture cue
  pays: as a feature for the trees, not as an edge of its own.
- The trees remove the one-edge bias and the blur bias, and cut edge noise by about a third.
- Pink at half contrast or heavy blur fails for every method (6-9 px): there the band
  tracker loses the band before any edge method runs. Not solved.

### Width defects

Bench: 96 painted lines, 192 defects, held-out frames plus white.

| Rule | Found | False reports per line |
|---|---|---|
| Old: 5 px for 60 px against the smoothed ribbon | 2 of 192 | 0.1 |
| **New: noise-scaled; slope edges** (measured against the smoothed ribbon, before the running median was introduced) | 73 of 192 | 1.1 |
| **New, against a running-median width; trees edges** | 85 of 192 | 0.7 |

Found by size (new rule, trees): 8 px over 60 px 20 of 21; 5 px over 60 px 26 of 36; 8 px
over 30 px 23 of 36; 5 px over 30 px 4 of 36; 3 px over 60 px 11 of 42; 3 px over 30 px 1 of
21. **The smallest defect found reliably is about 8 px over 60 px.** Short defects are
smeared by the 42 px averaging along the line that makes pale fabric measurable at all:
averaging over 22 px instead finds 27-31 of 36 of the 8 px / 30 px defects but doubles the
false reports (2.1-2.7 per line). Not adopted.

The new rule compares the line's own width, lightly smoothed, with its running median over
600 px, and reports where the difference exceeds 3.5 times the line's own width noise (and
2.5 px) for 16 px. The reference is a running median because, against the smoothed ribbon, a
wide spot made the stretches beside it read narrow.

**On real lines the rule is not yet trustworthy by itself.** On the repeat shots, from the
band edges alone: 3.2-4.5 reported places per line, of which 41-42 % recur in all or all but
one shot and about a third appear in one shot only. In the full flow, where SAM's edges and
the band's edges must both show the deviation, 257 real lines gave 0 on dark, 9 on pink and
1 on white. Requiring a deviation to recur across 2-3 shots would remove the single-shot
ones; this is the one real use found for taking several shots.

### Real repeat shots (precision)

| Edge method | Width of the same line, SD between shots, px (dark / pink) | Width noise along a line, px |
|---|---|---|
| Steepest slope | 0.10 / 0.13 | 0.73 / 1.26 |
| Profile fit | 0.08 / 0.20 | 0.83 / 1.51 |
| Trees | **0.07 / 0.14** | **0.64 / 1.11** |

- The final width repeats to about a tenth of a pixel between shots.
- **Averaging the shots gains almost nothing** (width noise 0.73 -> 0.72 dark, 1.26 -> 1.13
  pink). What limits the reading is the fixed pattern of the weave, not camera noise, so
  bursts are not needed for accuracy.

### Version 5

Local SAM 2.1 large (asked once) + trees edges + ribbon + new defect rule. Tool: "Edges
placed by: trees".

| | Dark | Pink | White |
|---|---|---|---|
| Lines "ok" of all (74 M1 frames) | 126 of 126 | 113 of 113 | 18 of 18 |
| Width, median (range), px | 32.0 (29.6-33.2) | 29.2 (24.8-34.8) | 32.2 (26.4-34.4) |
| Width deviations reported | 0 | 9 | 1 |
| Seconds per frame | 5.7 | 6.4 | 4.7 |

Hard frames from the other sets (soft white `10_05`, soft `real_captures`), 17 lines: widths
now 22-36 px (36-50 px with cloud version 4), misfit 0.7-1.8 px (1.9-4.8), no line refused;
9 of the 17 still marked "check by eye". Better by the numbers; not judged by eye. The
sharp pink real capture: 4 lines, 29.6-30.8 px, all "ok".

### Second round: tracker, defect scan, trees retrained

**Following the band (the weakest part on faint lines).** Three other trackers against the
present one (darkest stripe): darkness plus missing weave; a falling and a rising edge kept a
bead's width apart, followed together (coupled dynamic programming); that pair plus weave.
Strips where the band was lost (edge error over 3 px RMS), of 30 dark, 30 pink, 18 white:

| Condition | Darkness (present) | + weave | Edge pair | Pair + weave |
|---|---|---|---|---|
| Standard: dark / pink / white | 0 / 1 / 4 | 0 / 0 / 4 | 0 / 2 / 5 | 0 / 1 / 5 |
| Half contrast | 0 / 4 / 9 | 0 / 6 / 11 | 0 / 4 / 9 | 0 / 4 / 9 |
| Blurred 2.5 px | 0 / 1 / 4 | 6 / 1 / 4 | 0 / 2 / 5 | 0 / 2 / 5 |
| Blurred 4 px | 0 / 1 / 4 | 0 / 1 / 4 | 0 / 2 / 4 | 0 / 2 / 4 |
| Rough line 12 px off | 0 / 1 / 4 | 0 / 0 / 4 | 0 / 1 / 5 | 0 / 1 / 5 |

**None is better.** A first run on 10 strips had shown the weave tracker far ahead on pink;
that came from one strip with a fold in it. What loses the band is a real fold or mesh edge
inside the strip, and all four are drawn to it. The trackers stay selectable
(`bead_edges.read_strip(track=...)`); the default is unchanged. In the full flow the tracker
is held within 22 px of SAM's outline, which is what protects it.

**Width defects as a multi-scale test.** The width signal is averaged over 16, 30 and 60 px
windows and each window length is judged against the spread of its own averages (5 spreads,
at least 1.5 px). On the bench with trees edges:

| Threshold, spreads | Found of 192 | False per line |
|---|---|---|
| 4 | 116 | 0.9 |
| 5 (adopted) | 89 | 0.55 |
| 6 | 64 | 0.4 |
| Earlier single-scale rule | 85 | 0.7 |

A small gain, mostly on slight long defects (3 px over 60 px: 19 of 42 against 11). False
reports are not tied to where the shiny core starts or stops (1.1 against 1.0 per 1000
samples). **On real repeat shots it is no more trustworthy than before**: from band edges
alone 4.8-5.1 reported places per line, 45-47 % recurring across shots; at 7 spreads 3.1
per line on dark (39 % recurring) and 0.5 on pink (15 %). Real fabric disturbs the width
about five times more often than the painted bench does, so the bench flatters this rule.
A single shot's band edges cannot tell a real 5 px width change from the fabric's own
disturbance; agreement with SAM, or recurrence over shots, is needed.

**Trees retrained with every colour held out.** The hold-out is now cut per colour (10 dark,
10 pink, 2 white frames never trained on). Held-out error 2.49 -> 1.55 px RMS. Edge error
per sample, px RMS (bias in brackets), slope against trees:

| Held-out frames | Dark | Pink | White |
|---|---|---|---|
| Standard | 0.84 -> 0.51 | 0.91 -> 0.68 | 4.09 -> 2.69 |
| Core along one edge | 1.68 (-0.43) -> 0.71 (-0.11) | 3.54 (-1.27) -> 1.15 (-0.18) | 5.06 (-0.62) -> 3.16 (-0.12) |
| Half contrast | 1.15 -> 0.72 | 2.64 -> 1.45 | 7.64 (-0.88) -> 5.43 (-0.12) |
| Blurred 4 px | 0.97 (+0.24) -> 0.62 (-0.07) | 0.90 -> 0.78 | 5.71 -> 4.20 |

The white figures are large because 4 of the 18 white fabric strips lose the band (see
above); the bias is removed there as on the other colours. This model is the installed one.

**Version 5, final run** (retrained trees, multi-scale defect test, all 74 M1 frames): 257
of 257 lines "ok"; widths 32.0 (29.2-32.8) dark, 29.2 (24.8-34.4) pink, 32.0 (26.4-34.0)
white; width deviations where SAM and band edges agree: 1 dark, 3 pink, 2 white; 5.4-8.2 s
per frame in this run (overlays were being written at the same time).

### Ground facts from the user, and what they change (2026-10-08)

Stated by the user: the normal glue strip is **4 mm wide** most of the time; the **camera is
not calibrated**, so millimetre values may be off; **the supplied strips have no defects**.

**Scale (uncalibrated estimate).** Version 5 ribbon widths of all 257 lines:

| | Lines | Width px, median (5-95 %) | Pixels per mm if that is 4 mm |
|---|---|---|---|
| Dark | 126 | 31.2 (29.3-32.5) | 7.8 |
| Pink | 113 | 28.5 (26.5-33.1) | 7.1 |
| White | 18 | 33.2 (30.3-34.7) | 8.3 |
| All | 257 | 30.4 | **7.6 px per mm, 0.13 mm per px** |

- So the band that SAM and the edge finder outline (about 30 px) **is** the 4 mm glue strip.
  The earlier doubt between that and the 12-17 px between the dark side lines is settled.
- In these terms: edge error 0.5-0.7 px RMS is about 0.07-0.09 mm; width repeats between
  shots to about 0.01-0.02 mm; the smallest defect found reliably (8 px over 60 px) is about
  1 mm over 8 mm.
- Lines of nominally the same width read 26-35 px (SD 2.0 px = 0.27 mm). Width falls with
  distance from the frame centre (correlation -0.63), as an uncalibrated lens or a tilted
  camera would cause; fabric colour and panel cut are mixed into that, so it is a pointer,
  not a proof. **Millimetre widths, and any fixed tolerance in mm, need the camera
  calibrated** (`CalibrateAPP/`, `plane_scale.py`); until then about 7 % of the reading is
  position, not glue.

**False alarms, now with a truth.** Since no supplied line has a defect, every width
deviation reported on them is false:

| Threshold (spreads) | SAM and band edges must agree: lines falsely flagged of 257 | Band edges alone |
|---|---|---|
| 4 | 30 (11.7 %) | 223 (86.8 %) |
| 5 | 8 (3.1 %) | 190 (73.9 %) |
| 6 | 3 (1.2 %) | 156 (60.7 %) |
| **7 (now the default)** | **1 (0.4 %)** | 132 (51.4 %) |
| 10 | 0 | 102 (39.7 %) |

**Corrections to what is written above:**

- Band edges alone cannot be used for defects: at any threshold they flag 40-87 % of good
  lines. Only the rule requiring SAM's agreement is usable.
- The reports that recurred across repeat shots (41-47 %) are **not** real width changes:
  they are steady artefacts of the picture. Recurrence over shots therefore does not confirm
  a defect, and taking several shots has no use found here. The earlier suggestion is
  withdrawn.
- The 12 deviations reported on one pink line position in the first version 4 run, and the
  "repeatable 5 px width changes", were false alarms.
- The painted bench understates real disturbance badly (0.5-0.9 false per line predicted,
  about 2 per line observed from band edges). Its edge-error figures stand; its false-alarm
  figures do not.

**Not known, and now the main gap:** how many *real* defects the agreeing rule finds. SAM's
edges are kept only where they follow the line's steady course within 5 px (the leak
repair), so at a real wide or narrow spot SAM's evidence may be missing and the rule silent.
That cannot be tested on the supplied images (no defects) or on the strip bench (no SAM).
It needs either images of strips with known defects, or defects painted into whole frames
so that SAM sees them.

### Tried or planned and not kept, or not done

- Not kept (second round): weave tracker, coupled edge-pair tracker, pair + weave tracker.
- Not kept: sub-pixel slope, coarser slope scale, phase congruency, texture as an edge,
  lightness-texture fusion, shorter averaging, averaging repeat shots, requiring two edge
  methods to agree on a deviation (no better: 28-42 % recurring).
- Planned and **not done**: Steger's line detector; graph cut; a Kalman smoother (the noise-scaled rule and running-median reference were done
  instead); classical deblurring; specular-free image; a learned colour-texture projection;
  a per-pixel classifier; a learned fusion of edge finders; a learned line-quality
  classifier; the millimetre check through the plane scale; rotation / half-pixel shift
  invariance checks.
- Following the band past a fold or mesh edge remains unsolved without SAM; three trackers
  tried, none better.

### What this does not establish

- Real width in millimetres: still no caliper measurement to anchor any of it.
- That the painted bead is what glue looks like in every case; white was not in the trees'
  held-out frames in the first round (two white frames are held out since the second).
- Whether the repeatable 5 px width changes on real lines are real glue variation or a
  steady artefact of the picture.

## SegFormer and CLAHE

Researched 2026-10-09 at the user's request ("clash" was read as CLAHE, the contrast
filter). Question: can a trained SegFormer mark the glue band, in place of SAM, and does
CLAHE before it help? Scripts, models and logs are kept in `~/flod_experiments/segformer/`
(`runs.py`, `prep.py`, `train.py`, `evalseg.py`, `page.py`); nothing of it is in the tool yet.

### What SegFormer is

- A semantic-segmentation network: a hierarchical transformer encoder (MiT, sizes B0 to B5)
  and a small all-MLP decoder. It labels every pixel; it takes no prompt.
- Its output is at **a quarter of the input resolution** and is enlarged back. So it is run
  on native-resolution tiles (512 px), never on a reduced frame, and its mask is used as
  evidence for the ribbon, not as the final edge.
- **Licence: NVIDIA's SegFormer code and pretrained weights are for non-commercial use
  (research or evaluation) only.** Using it in the inspection product needs NVIDIA's
  agreement, or the same training repeated on an openly licensed network (for example the
  U-Net with a ResNet-34 encoder already used for the ROI model; not tried).

### How it was trained

- Labels: the version 5 ribbons of the M1 frames (the samples have no defects, and v5 is the
  best outline there is; there is still no hand-marked truth).
- 52 frames to train on (24 dark, 24 pink, 4 white), 22 held out (10, 10, 2), cut per colour
  by capture time so repeat shots stay on one side. 2 102 tiles of 512 px at native
  resolution: ten along each line plus plain ROI; glue is 5.5 % of the pixels.
- Fine-tuned from `nvidia/mit-b0` and `nvidia/mit-b2`; loss = cross-entropy + Dice; AdamW;
  flips, transposition and brightness jitter; half precision.

| Model | Parameters | Iterations, batch | Training time | GPU memory, training |
|---|---|---|---|---|
| SegFormer-B0 | 3.7 M | 3000, 6 | 7.5-16 min | 1.8 GB |
| SegFormer-B2 | 27.3 M | 4000, 3 | 19 min | 3.3 GB (batch 6 did not fit in 6 GB) |

### Procedure, step by step

Everything runs on this PC; nothing is uploaded. Python is `~/flod_experiments/gpu/bin/python`
and `D` is `~/flod_experiments/segformer`.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1. Labels | `runs.py D/runs` | Runs version 5 (ROI model, rough line, SAM 2.1 large asked once, trees, ribbon) on all 74 M1 frames | one file per frame: image path, colour, ROI, rough lines, ribbons |
| 2. Split and tiles | `prep.py D` | Splits frames per colour by capture time (`tools/train_bead_ml.session_split`, every third block held out); cuts training tiles | `D/held.txt`, `D/data/train_*.npz` |
| 3. Training | `train.py D b2 0 4000 3` (size, CLAHE 0/1, iterations, batch) | Fine-tunes the pretrained encoder with a new one-class head | `D/segformer_b2_plain/` |
| 4. Held-out test | `evalseg.py D segformer_b2_plain` | Mask on each held-out frame, then the version 5 finishing; compares with the version 5 ribbons | table, `D/eval_*.json` |
| 5. New-piece test | `evalseg.py D segformer_b2_plain hard` | The same on six frames from `10_05` and `real_captures` | one row per frame |
| 6. Pictures | `page.py D OUT` | Version 5 and SegFormer side by side at full zoom | `~/flod_experiments/out/segformer_vs_sam/` |

**Tiles for training.** 512x512 px at native resolution. Ten per glue line, centred on points
spread evenly along the ribbon and moved at random by up to 170 px each way, plus up to six
per frame placed at random inside the ROI (plain fabric, so the model learns what is not
glue). Label: 1 inside any version 5 ribbon of the frame, 0 elsewhere. For the CLAHE variant
the same tiles are cut from the frame after CLAHE on the lightness channel (clip limit 3,
16x16 grid over the whole frame).

**Training settings.** `SegformerForSemanticSegmentation` from `transformers`, started from
`nvidia/mit-b0` or `nvidia/mit-b2`, one output class. Input: RGB scaled to 0-1, normalised
with the ImageNet mean and deviation. The output (a quarter of the tile's size) is enlarged
to the tile bilinearly before the loss. Loss: binary cross-entropy plus Dice. Optimiser:
AdamW, learning rate 6e-5 for the encoder and 6e-4 for the head, falling linearly to zero,
weight decay 0.01. Half precision. Each batch is flipped left-right, up-down and transposed
with probability one half each, and its brightness and contrast are changed by up to 10 %.
B0: 3000 iterations of 6 tiles. B2: 4000 iterations of 3 tiles (6 did not fit in 6 GB).

**Running it on a frame.**

1. ROI from the panel model.
2. Tiles of 512 px with a step of 384 px over the ROI's bounding area (64 px margin); tiles
   that touch no ROI are skipped (30-47 tiles per frame). Eight tiles per batch, half
   precision.
3. Probability = sigmoid of the enlarged output; where tiles overlap, the mean. Mask =
   probability over 0.5, inside the ROI.
4. For each rough line: the mask is read as a ribbon along it (`sam_refine.trim_ring`), which
   gives the same kind of evidence SAM's outline gives.
5. `finish_as_ribbon` with the trees: states, band edges, smooth ribbon, ends, defect rule.

**Measures.**

| Measure | Meaning |
|---|---|
| Lines with a mask | lines for which the mask could be read as a ribbon at all |
| Separate mask pieces | connected pieces longer than 400 px; equal to the number of lines when each line is one piece and nothing else is marked |
| Stray mask | mask further than 30 px from any version 5 ribbon, as a share of the glue area |
| IoU | overlap of the mask with the filled version 5 ribbons |
| Mask's edges from the ribbon's | mean distance of the mask's two edges from the version 5 ribbon's edges, along the line |
| Final ribbon's centre / width | the ribbon fitted to the mask's evidence against the version 5 ribbon |
| Check by eye, false deviations | the tool's own flags; every deviation is false, the samples having no defects |
| Covers the line | share of the line's two edges taken from the mask itself, not repaired |

### Held-out M1 frames (78 lines): its mask in place of SAM's outline

The mask goes through the same leak repair, trees and ribbon as SAM's outline does.

| | B0 | B0 + CLAHE | B2 |
|---|---|---|---|
| Lines with a mask | 78 of 78 | 78 of 78 | 78 of 78 |
| Separate mask pieces over 400 px (should equal the lines) | 78 | 78 | 78 |
| Mask outside any glue line (share of glue area) | 0.000-0.001 | 0.000 | 0.000 |
| Overlap with the v5 ribbon, IoU: dark / pink / white | 0.962 / 0.944 / 0.941 | 0.963 / 0.946 / 0.942 | 0.972 / 0.956 / 0.948 |
| Mask's edges from the v5 ribbon's, px | 0.61 / 0.80 / 0.92 | 0.61 / 0.78 / 0.91 | **0.51 / 0.66 / 0.81** |
| Final ribbon's centre from v5's, px | 0.10 / 0.27 / 0.32 | 0.10 / 0.28 / 0.29 | 0.09 / 0.27 / 0.29 |
| Final width against v5's, px | +0.2 / +0.8 / +0.8 | +0.2 / +0.7 / +0.9 | +0.3 / +0.8 / +0.9 |
| Lines "check by eye" | 0 | 0 | 0 |
| False width deviations (all are false) | 1 | 2 | 1 |
| Seconds per frame for the mask (30-47 tiles) | 0.3-0.9 | 0.3-0.6 | 0.8-1.3 |
| GPU memory, running | 0.66 GB | 0.66 GB | 1.9 GB |

- It finds every line as one piece and marks nothing else, without being told where the
  lines are. SAM needs points placed along a rough line first.
- Its raw mask lies 0.5-0.9 px from the final ribbon; SAM 2.1's raw outline lay 0.9-3.6 px
  from it. But SegFormer was *trained on those ribbons*, on frames of the same pieces taken
  at another time, so this says it has learned the task, not that it is more accurate.
- The mask takes about 1 s per frame against 3-4.5 s for SAM 2.1 large.

### Frames from the other sets, never trained on (the real test)

Soft white `10_05` and soft `real_captures` (17 lines), and one sharp pink capture (4 lines).

| | Version 5 (SAM 2.1) | SegFormer-B0 | B0 + CLAHE | **SegFormer-B2** |
|---|---|---|---|---|
| Soft white: width range, px | 22-36 | 26.4-32.0 | 26.4-31.2 | **28.0-32.0** |
| Soft white: outline source covers the line | 0-84 % (one line refused) | 60-95 % | 0-87 % | **69-100 %** |
| Soft white: ribbon misfit, px | 0.7-1.8 | 0.8-1.5 | 0.8-1.5 | 0.6-1.3 |
| Soft white: lines "check by eye" | 9 of 17 | 1 of 17 | 1 of 17 | 1 of 17 |
| Sharp pink: covers the line / width | 97-100 % / 29.6-30.8 | 97-100 % / 30.4-31.6 | 97-100 % / 30.4-31.6 | 97-100 % / 30.4-32.0 |

- **This is the first method that handles the soft white frames**: widths come out at
  28-32 px, which is the 4 mm strip at the M1 scale, where SAM gave 22-36 px and marked half
  the lines doubtful. It was trained on sharp M1 frames only, with only 4 white ones.
- B2 is steadier than B0 on these frames; on the M1 frames the two are equal.
- One line (`0041`, line 2, the softest frame) is still marked "check": at one end the
  ribbon turns off the bead, with SAM and with SegFormer alike. SegFormer's raw mask is
  closer to the bead there than the ribbon fitted to it. Seen by eye; it is the only
  SegFormer picture examined by eye.

### CLAHE

- Trained and run with CLAHE on lightness (clip 3, 16x16 grid over the whole frame): **no
  gain on the M1 frames** (every figure equal to within 0.03 px) and **worse on the soft
  frames** (the mask covers 0-87 % of a line against 60-95 %; one line gets no mask at all).
- This agrees with every earlier filter result for SAM and for the local detector: on these
  pictures contrast filters do not help a segmenter, and on soft pictures they hurt.

### What it means

| | SAM 2.1 large (version 5) | SegFormer-B2 |
|---|---|---|
| Needs a rough line first | yes | no (its mask gives the lines) |
| Needs training on this product | no | yes: labels, about 20 minutes on this GPU |
| A new fabric colour or product | works untrained (measured on three colours) | unknown until tried; expect retraining |
| Soft / blurred frames | weak | good |
| Time for the outline | 3-4.5 s per frame | about 1 s |
| Licence | Apache 2.0 | non-commercial |

Recommended as the next version to build and judge: SegFormer-B2 as the outline source with
SAM 2.1 as the fallback for anything the trained model has not seen, both feeding the same
trees and ribbon. Not done: wiring it into the Pipeline Lab; a flow without the rough line
(lines taken straight from the mask); a larger and more varied training set; the same
training on an openly licensed network; any test on fabric or products outside these sets.

### Limits

- Agreement with version 5 is not accuracy: there is no independent truth for whole frames.
- The held-out M1 frames show the same pieces as the training frames at another time; only
  the six frames from the other sets are new pieces, and they are few.
- White: 4 training frames, 2 held out.

## Methodology

- **Full resolution always.** Every method runs on the original pixels; SAM gets 1024 px
  tiles, never a reduced frame.
- **One change at a time, same inputs.** For filter sweeps the ROI and rough lines were
  computed once per frame and cached, so only the filter differed between runs.
- **Tune and validate on different frames, as far as possible.** The repair was developed
  on 8 frames; the table above is from 18 frames taken as every fourth capture. Three of the
  18 were also tuning frames (`230044_057574` dark, `230210_313721` pink, `230606_026427`
  white), so the validation numbers are slightly optimistic.
- **Measures.**
  - *Lines found:* lines returned against ROI regions. Touching strips count as one region,
    so more lines than regions is normal.
  - *Share SAM outlined:* edge samples kept from SAM's mask over all edge samples. It says
    how much of the outline is evidence and how much is carried over.
  - *Width:* twice the distance transform on the outline's skeleton, median and 10th-90th
    percentile. Consistency between lines and frames is checked; accuracy is not.
- **Eye checks.** Overlays were viewed for 2 pink and 1 white frame after the first repair
  and 1 pink frame after the second; the rest were judged by numbers. Counts of beads
  "present" in the older `10_05` work were made by eye, not from labels.
- **Training a model on a method's own output** (SegFormer on the version 5 ribbons):
  the split is by capture time within each colour, so repeat shots of one layout are never
  on both sides; agreement with the labels on held-out frames is reported as agreement, not
  accuracy; and the model is judged on frames of other pieces from other image sets, with
  measures that do not use the labels (width against the known 4 mm strip, the share of
  lines the tool itself marks doubtful, false defect reports).
- **A result is not correct because an overlay looks plausible.** An earlier detector
  outlined the wrong feature and looked fine; the user's confirmation is the test.
- **No regression.** New behaviour is opt-in or in the separate tool. The full suite (336
  tests) passes; default detector output was checked byte-identical to the committed one
  when the opt-in paths were added.
- **Environment.** `~/flod_experiments/gpu/bin/python` (Python 3.12, RTX 3050, onnxruntime,
  inference-sdk). The project `.venv` is Python 3.14, where inference-sdk cannot be
  installed. SAM calls upload image tiles to Roboflow and need `ROBOFLOW_API_KEY`.
- **Outputs.** Overlays and per-frame numbers: `~/flod_experiments/out/lab_validation`
  (first repair) and `lab_validation2` (current).

## Not verified

- The bead's-edge widths against a caliper, and whether the dark band is the glue's full
  width. On pink, which of the two lower edges is the bead's.
- The edge finder on anything but M1 captures; its sizes are in pixels of a 4608-wide frame.

- **Width in millimetres**, and which width is the glue: SAM's about 30 px ribbon, or the
  12-17 px between the two dark side lines that the local detector measures. Needs caliper
  measurements of real lines.
- Pink lines where SAM outlined little: the width there is the line's own average.
- Line ends: the outline stops short of the bead's tips on some panels.
- `real_captures`, `10_05` and the loose files with the current method. Only M1 captures
  were run.
- White fabric beyond 2 validation frames.
- The Windows inspection PC, and the tool's clicked-point and drawn-box modes by hand.
- Whether the hardest pink lines can be outlined at all without a change of lighting.

## Next steps worth taking

1. Caliper widths on a few lines per colour, to settle the width definition and the scale.
2. Run the current method on `real_captures` and the white `10_05` frames.
3. For pink: a second light at a low angle, or use the outlines already made as labels for
   a small bead model on native tiles.
