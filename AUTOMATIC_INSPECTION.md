# Automatic inspection

Run `run_automatic.bat`, or `.venv\Scripts\python.exe main_1366.py`.
The existing camera setup, input format, resolution, rotation, lens calibration
and measurement-surface files are used. Camera 1 is the single active camera;
its device index remains configurable. There is no manual inspection trigger,
ESP button listener, mask picker or inspection ROI editor.

## Operator cycle

1. A hand enters, then leaves. Consecutive hand samples debounce presence.
2. After the configured absence interval, that exact frame is classified.
3. `full_fabric` and `half_fabric` above the confidence threshold proceed.
4. The original snapshot is corrected with the existing lens calibration.
5. Segmentation receives the **undistorted** image and returns full-frame polygons.
6. Each of up to four fabrics is measured separately using the existing strip
   skeleton, centreline, ten-segment widths, metric conversion and tolerance rules.

Full-screen fabric tabs cycle continuously and can also be selected manually.
Every result replaces the previous tab set with the latest frame's detected
fabrics: one fabric gives one tab, two give two tabs, up to four. When adding
fabric one by one, the next accepted trigger inspects all fabric currently in
view, including pieces left from the previous capture; tabs are not accumulated
from historical captures and four pieces are never required to begin.
Zoom, pan, fit and settings remain available. Tabs do not stop hand monitoring.
`result_view.show_live_preview` controls a small live camera panel beside the
large measured image. It defaults to `true`. Set it to `false` and restart to
give the result the full width; background capture and hand monitoring continue
in either mode. The side preview is not saved into result images.
Only a confident `no_fabric` verdict after another hand cycle clears the results.
A low-confidence verdict or inference failure does not claim the table is empty.
Accepted replacement captures replace the tabs once their results are ready.
An accepted capture with no strip detections shows a warning and removes old tabs.

A hand returning invalidates any in-flight result for display. That capture is
still saved. There is only one segmentation request at a time. A new accepted
trigger while it is busy retains the latest immutable capture and processes it
next automatically, without another hand cycle. At most one capture waits;
a newer hand cycle discards an obsolete waiting capture, and a no-fabric verdict
cancels it. Empty-table checking remains active. Opening settings pauses and
resets the hand cycle and cancels any waiting capture.

## Configuration

Edit `config.json`, or press **S** during startup to use its settings form.
Restart after changing JSON. Width, tolerance and tab timing can also be changed
in the existing inspection-profile UI.

| Setting | Default | Purpose |
|---|---:|---|
| `auto_trigger.hand_absence_seconds` | 1.0 | Time hands must remain absent after debounce |
| `auto_trigger.debounce_frames` | 3 | Fresh consistent hand samples before a presence change |
| `auto_trigger.min_hand_present_seconds` | 0.0 | Minimum presence duration to arm the cycle |
| `auto_trigger.fabric_min_confidence` | 0.5 | Acceptance and empty-table confidence threshold |
| `auto_trigger.max_hands` | 2 | Check both hands |
| `inspection.strip_width_mm` | 4.0 | Common target width |
| `inspection.strip_width_tolerance_mm` | 1.0 | Common ± tolerance |
| `inspection.result_display_seconds` | 5.0 | Seconds per tab; tabs loop until fabric removal |
| `result_view.show_live_preview` | true | Show a small live camera panel beside the result tabs |
| `sam_detection.strip_segments` | 10 | Existing measurement segment count |
| `segmentation.provider` | `workflow` | Provider, or `package.module:Factory` for a local implementation |
| `segmentation.input_width`, `input_height` | 1024, 768 | Workflow input dimensions |
| `segmentation.timeout_seconds` | 180.0 | Per-attempt workflow timeout |
| `segmentation.max_retries` | 3 | Reference workflow retry count |
| `capture_storage.save_rejected_triggers` | false | Save sources before the fabric gate, including rejects/errors/superseded captures |
| `capture_storage.max_sets` | 1000 | Maximum number of capture directories |
| `capture_storage.jpeg_quality` | 82 | Derived-image JPEG quality |
| `capture_storage.preview_max_edge` | 1920 | Maximum saved JPEG dimension |

Some legacy JSON sections remain for shared calibration/configuration compatibility
but are hidden from the automatic startup form. `serial`, `crop_setup`,
`color_mask`, `region_homography`, and old SAM prompt/crop-upload controls do not
drive automatic inspection. Millimetres use the saved full measurement plane;
missing calibration is reported as unmeasured, never as a passing inspection.

## Pale and white fabric

A clear or white glue strip on white fabric has almost no brightness or colour of
its own; in the camera image it differs from the fabric beside it by one or two
gray levels and is visible only as a thin dark line along each edge. The options
below exist for that case. **All of them are off in the shipped `config.json`, and
with them off the inspection behaves exactly as it did before they were added.**
Switch them on one at a time and compare with `tools/replay_capture.py`.

| Setting | Default | Purpose |
|---|---:|---|
| `segmentation.roi` | `[0, 0, 1, 1]` | Part of the frame sent to the model, as `[x1, y1, x2, y2]` fractions. Cropping away the table gives the strip more model pixels. Results are mapped back to the full frame |
| `segmentation.enhance.enabled` | false | Local contrast enhancement (CLAHE on lightness) of the image sent to the model only. `clip_limit` sets its strength, `tile_grid` its locality |
| `segmentation.save_input` | false | Save the exact image sent to the model as `segmentation_input.jpg` in the capture set |
| `segmentation.report_missing_strips` | false | Show a matched fabric that has no detected strip as a `NO STRIP` tab instead of leaving it out |
| `segmentation.fragment_policy` | `error` | `error` stops the capture when one fabric returns several strip pieces. `largest` keeps the largest piece, marks that fabric `FRAGMENTED` and still grades the others |
| `segmentation.min_confidence` | 0.0 | A strip below this confidence is shown as `LOW CONFIDENCE` instead of PASS/FAIL |
| `sam_detection.refine_edges` | false | Move each strip edge from the model's outline onto the dark edge line in the full-resolution image before measuring |
| `sam_detection.refine_search_px` | 10.0 | How far, in full-resolution pixels, an edge may be moved |
| `capture.controls.enabled` | false | Fix exposure, gain and white balance instead of leaving them automatic. See `CONFIGURATION.md` |

`NO STRIP`, `FRAGMENTED` and `LOW CONFIDENCE` fabrics are never graded: each raises
a warning, so the dashboard shows WARNING rather than PASS. A `FRAGMENTED` or
`LOW CONFIDENCE` tab still shows its measured figures for the operator.

### Edge refinement

The model outlines the strip on a 1024 × 768 image, where a 4 mm strip is about
six pixels wide, so one model pixel of error on each edge is larger than a ±1 mm
tolerance. With `refine_edges` on, the outline only says roughly where the strip
is; the width is then read from the camera's own pixels. On the sample white-fabric
images, seven deliberately wrong outlines whose average widths spanned 20 pixels
came back within 2.5 pixels of each other.

It works only where the strip edges show as dark lines. A strip with a plain
brightness step, or nothing visible, keeps the model's widths, and
`edge_refinement.refined_fraction` in `result.json` records how much of each strip
was actually relocated. Per-segment values are noisier than the average. Check the
result against a ruler or caliper measurement of the same strips before relying on
it for grading.

### Replaying saved images

`tools/replay_capture.py` runs the fabric gate, the model and the measurement on
image files, with no camera, without touching saved data and without editing
`config.json`:

```text
.venv\Scripts\python.exe tools\replay_capture.py enhancement_images\*.jpeg
.venv\Scripts\python.exe tools\replay_capture.py enhancement_images\*.jpeg --out replay_enhanced ^
    --set segmentation.enhance.enabled=true --set segmentation.roi=[0.1,0.05,0.95,0.85] ^
    --set segmentation.report_missing_strips=true --set segmentation.fragment_policy=largest ^
    --set sam_detection.refine_edges=true
```

Each image gets a folder with `gate.json`, `segmentation_input.jpg`, `overview.jpg`,
`fabric_NN.jpg` and `result.json`. Add `--calibration` and `--extrinsics` with the
camera's saved files for lens correction and millimetres; without them widths stay
in pixels.

### Local glue-line detector

`segmentation/glue_line.py` finds glue strips on pale fabric without a model and
without a network connection. Select it with

```json
"segmentation": { "provider": "segmentation.glue_line:GlueLineSegmenter" }
```

The shipped provider stays `workflow`; nothing changes until this is set.

The glue is the thin shiny bead: a narrow bright line with a thin dark line along
each edge. The detector enhances the image (lighting gradient removed, then a long
directional line filter kept separately for dark and bright lines), looks only
inside pale fabric away from its outline, and at every point of every dark line
looks straight across for a second dark line running the same way within the strip
width range, with a bright line between the two. Those points are joined into
centrelines and short faded stretches are bridged. A strip is kept only where the
fabric along its centre is brighter than the fabric just outside its edges, which
is the bead itself. Folds, mesh edges, seams, the wider band
beside the bead and weave or moire texture do not have that signature.

The strip outline is the pair of dark edges, so its width is the edge-to-edge
distance in camera pixels, and the existing measurement and width grading run on
it unchanged.

Settings are in `segmentation.glue_line`; every distance is in pixels of the
full camera frame, so they need revisiting if the camera height or the product
changes.

| Setting | Default | Purpose |
|---|---:|---|
| `min_width_px`, `max_width_px` | 8, 34 | Narrowest and widest bead looked for |
| `min_length_px` | 900 | A shorter stretch is not reported |
| `line_sensitivity` | 2.0 | Line threshold in multiples of the fabric's noise; lower finds fainter beads and more clutter |
| `min_seen_share` | 0.55 | Share of a strip's length that must be seen rather than bridged |
| `min_gloss_share` | 0.35 | Share of a strip along which the bright bead must be seen |
| `fabric_margin_px` | 30 | Band along the fabric outline that is ignored |
| `max_saturation`, `min_fabric_gray` | 45, 90 | What counts as pale fabric |
| `join_gap_px` | 250 | Stretches of one strip closer than this, end to end, are joined |
| `track_gaps` | true | Measure a bridged stretch where the edges are still faint enough to find, instead of guessing it |
| `working_scale` | 1.0 | The bead is only 10-17 pixels wide, so detection runs at full resolution |

Confidence is the share of the strip along which the bead was seen. With
`segmentation.save_input` on, the saved `segmentation_input.jpg` shows the dark
lines in red and the bright bead in green.

Measured on the sample images (56 frames, 7-30 seconds each on a CPU): 52 beads
reported, most of them 1200-1950 pixels long. Widths repeat from frame to frame
(about 15-19 pixels on the two-panel product, about 12 on the four-panel one).
Beads are still missed or cut short where the frame is soft or mesh lies over the
bead, and frame 0036, the softest, gives none. It reports nothing on dark fabric,
which the workflow provider already handles. Widths have not been checked against
a ruler. Turning `track_gaps` on changes 13 of the 56 frames, almost all of them
by less than 0.01 in confidence; the strip it most repairs was the second bead of
the WhatsApp frame, whose confidence goes from 0.562 to 0.652 and whose width from
12.6 to 15.8 pixels, in line with the other beads in the same frame.

`line_sensitivity` (default 2.0) and `min_seen_share` (default 0.55) trade
completeness against clutter: weave and moire can be chained into something
strip-shaped only by bridging long gaps, and `min_seen_share` is what rejects
those. `join_gap_px` (default 250) is the longest faded stretch that is bridged.
Where a bead fades but its two edges are still there, `track_gaps` looks for them
again along the stretch at a relaxed threshold and measures the width there, so
that stretch counts as seen rather than as a guess; if neither edge survives, the
straight bridge is used and confidence falls exactly as before.

### What software cannot fix

- **The fabric classifier** was not trained on white fabric. If it reports
  `no_fabric` for a white piece the capture is skipped and the tabs are cleared.
  Set `capture_storage.save_rejected_triggers` to true during trials so those
  frames are kept, then retrain the classifier with them.
- **Lighting.** Even overhead light hides clear glue. A light at a low angle across
  the table turns the strip edges into strong lines and helps more than any setting
  here.

## Where to modify future versions

- `auto_trigger/checkpoint.py`: hand-cycle timing and debounce, derived from the reference.
- `auto_trigger/models.py`: hand inference and accepted fabric classes.
- `auto_trigger/fabric_check.py`: reference H5 loading and 224×224 input preparation.
- `auto_trigger/controller.py`: events, snapshots, background processing and result invalidation.
- `segmentation/provider.py`: workflow-specific preprocessing and polygon mapping.
- `segmentation/workflow/`: locally copied transport, retry and response parsing.
- `inspection/measurement.py`: per-fabric use of the existing measurement method.
- `inspection/result_view.py`: tab presentation; `inspection/dashboard.py`: UI integration.
- `inspection/storage.py`: source/derived image formats and capture-set retention.
- `strip_analysis.py` (`refine_widths`): full-resolution edge refinement.
- `tools/replay_capture.py`: offline replay of saved images through every stage.

### Local model contract

Implement a class with `__init__(settings)` and `segment(frame_bgr)`.
Return `segmentation.SegmentationResult(instances, (width, height))`.
Each `segmentation.Instance` supplies an N×2 polygon, confidence, label and
optional fabric box `(x1, y1, x2, y2)`, all in the **input frame's pixels**.
One fabric has one continuous strip. A mask-producing model can use
`segmentation.mask_to_polygons` after resizing its mask back to input size.
All preprocessing, resizing, model loading, inference and postprocessing belong
inside the provider. Never return resized-model coordinates to the caller.

Set `segmentation.provider` to `segmentation.local_model:LocalSegmenter` when
that implementation exists. No local segmentation weights are supplied here;
the default provider uses the reference Roboflow `custom-workflow`, with the
existing root `.env` or `ROBOFLOW_API_KEY` environment variable.

## Saved data

Data follows the existing application storage root: Windows Documents / `data files`.
The default relative directory is `Dataset_capture/Automatic`.

```text
Automatic/
  YYYYMMDD_HHMMSS_microseconds_uniqueid/
    .inspection-set
    capture.json       # timestamp, frame sequence, gate scores and target settings
    original.png       # full-resolution camera snapshot, existing rotation applied
    undistorted.png    # full-resolution lens-corrected version of that same snapshot
    overview.jpg       # compact full-frame result
    fabric_01.jpg      # annotated individual fabric (up to fabric_04.jpg)
    result.json        # frame polygons, per-segment measurements, warnings/errors
    segmentation_input.jpg  # only with segmentation.save_input: the image sent to the model
```

By default rejected gates produce no files. With `save_rejected_triggers=true`,
sources are saved before classification, so errors also remain inspectable.
If camera calibration is missing, only the original can be saved; there is no
fabricated undistorted image. Rejected/error/superseded sets count toward retention.
Derived images are JPEG previews; original polygon coordinates and numerical
measurements are preserved in JSON independently of preview scaling.

Retention reserves capacity before the next set, deleting oldest **completed**
sets together, including all their files. It only deletes recognized directories
with this application's marker and refuses links/unexpected subdirectories.
In-flight sets are protected. If a very small limit leaves no free slot because
another set is in flight, saving reports an error rather than exceeding the limit.
After a restart interrupted sets are marked interrupted and become eligible for
normal oldest-first retention. Do not run two application instances against the
same capture directory. Closing during a capture can leave such an interrupted set.

## Environment and verification

Python 3.12.10, NumPy 2.2.6, MediaPipe 1.0.1, TensorFlow/tf-keras 2.21.0,
OpenCV 4.12.0.88, inference-sdk 1.7.1. Both OpenCV distributions are pinned to
the same build because the SDK and MediaPipe request different distribution names.
The project `.venv` contains the installed dependencies; no reference environment
or reference source files are modified. Virtual environments are not copied.

Run `python -m unittest discover -s tests -p "test_*.py"` and
`python tests/verify_automatic_ui.py` using this project's interpreter.
Tests cover source retention, camera acquisition, calibration, polygon mapping,
four-strip measurement, classification gates and late-result suppression.
The UI smoke test uses synthetic fabrics and never opens a camera or calls SAM.

Physical millimetre accuracy and the full/half/empty classifier must still be
checked on the installed camera and representative 1–4-fabric arrangements.
The copied classifier classified a synthetic black image as `half_fabric`; its
predictions are not proof of correct empty-table detection on a new background.
Disconnected strip fragments or more than four strip predictions are reported
instead of silently choosing one fragment or discarding extra fabrics.
