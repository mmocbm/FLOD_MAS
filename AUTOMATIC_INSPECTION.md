# Automatic local ONNX inspection

Run `run_automatic.bat`, or `.venv\Scripts\python.exe main_1366.py`.
The existing single-camera acquisition, rotation, lens calibration and saved
measurement-plane setup remain in use. Automatic inspection has no Roboflow
request, API key, manual mask selection or fixed inspection crops.

## Operator cycle

1. A hand appears and leaves. Three fresh agreeing samples debounce presence;
   hands must then stay absent for 1 second. A hand-free scene does not retrigger.
2. Copy that exact camera frame and classify fabric on the original snapshot.
   The two-class `keras_model_2c.h5` uses `labels_2c.txt`: 0 = `full_fabric`,
   1 = `no_fabric`. Accept only when the full_fabric score is strictly greater
   than no_fabric. Ties/rejected frames do not clear results or accepted captures.
3. Undistort the accepted snapshot using the existing camera calibration.
   Missing camera calibration prevents inspection; there is no raw-frame fallback.
4. Run the local source-region model at aspect-preserving width 1152, with reflected
   padding to multiples of 32 and ImageNet RGB normalization. One inference pass
   produces a full-frame mask; threshold 0.5, minimum region area 0.2%, hole filling.
5. Extract curved boundaries, remove straight backing edges/caps and endpoint
   hooks (4% endpoint zone, 65 degree hook angle). Keep wave lines with minimum wave
   ratio 0.008, span 10% of the larger frame dimension, and boundary support 90%.
6. Sort accepted lines by mean x. With no results or outstanding inspections,
   inspect ALL lines. Otherwise inspect the configured `subsequent_line`
   (`rightmost` by default, or `leftmost`). Decide this when capturing, not later
   when the worker dequeues it. No four-line minimum or maximum is imposed.
7. Offset 100 undistorted-frame pixels into the source-mask interior. Apply CLAHE
   to source lightness (clip 2, 16x16 tiles), then unfold this curved band horizontally.
   Unfolded height follows the offset; width follows the band's middle arc length.
8. Run the local two-class SegFormer at its trained 1024x64 input and resize its
   mask back to the actual unfolded dimensions. Clean noise/spikes while preserving
   thickness changes and long missing intervals. Cleanup window 9 and short-gap
   limit 12 are scaled from model-width pixels; smoothing tolerance is 0.15.
9. Refold the cleaned mask into the full undistorted frame. Clip to the selected
   band. Label segment widths in millimetres (pixels if uncalibrated), draw passing
   segments green, and highlight out-of-tolerance segments red.
10. Measure each adhesive component with area >=100 pixels using the reference PCA
    slice centreline, perpendicular recentering, 5-pixel sampling and subpixel
    boundary intersections. Total length sums separately measured components,
    without bridging gaps. Width statistics and ten equal pixel arc-length
    segments describe the LONGEST component, matching the reference workflow.

This is the in-memory method from `anotate_main_line/live_inspection.py`; CLI,
manual-annotation, other camera backends and OBS settings were not imported.
The reference geometry and both model assets are copied into `inspection/local/`.
Nothing at runtime depends on that external Desktop folder.

## Calibrated measurements and grading

The reference workflow supplies pixels. This application additionally maps the
refolded centreline and left/right width intersections into millimetres using the
saved full-frame measurement plane (`plane_scale.load_frame_scale`). It does not
multiply all distances by a global millimetres-per-pixel estimate. Component
lengths use the transformed sampled centreline points; small chord discretization
can make them differ slightly from a scalar conversion of reference pixel length.

Before width grading, exclude `inspection.end_exclusion_percent` from each end
of each continuous adhesive component using its full centreline arc length.
The default 5% retains the middle 90%, divided into ten measurement segments.
Excluded samples do not affect pixel/mm averages, PASS/FAIL or overlay highlights.
The original image/mask and full detected length are preserved. Gaps remain gaps.
Set this in Settings → Inspection profile → "Exclude from EACH end (%)".
Save Profile persists the value; allowed range is 0 <= value < 50. Each queued
capture keeps the percentage that was active when captured.

PASS requires an available average metric width for every segment of every component,
with each average within the inclusive target +/- tolerance (default 4 +/- 1 mm).
Any measured out-of-range segment produces FAIL when all segments are available.
A segment with no measurement of its own, and missing plane calibration or other
incomplete segment measurements, give WARNING instead: the unmeasured segment is
drawn in the warning colour and labelled WARNING at its own position, and no PASS
or FAIL is shown for that fabric. Pixel results remain available. Every displayed
component is graded; this does not introduce a continuity/gap acceptance rule.

## Results and lifecycle

Each inspected line appends its own fabric tab, and the newest result opens
immediately. Tabs fit the inspected line, retain independent zoom/pan, and show
only a short status. Scroll the tab bar to revisit older lines. Mouse wheel or
+/- zooms; left-drag pans; RESET fits the image again. The small live camera
preview remains active beside the captured overlay.

Accepted captures enter a FIFO queue. The inspection worker processes every
capture in order, sharing one source-model inference across all selected lines.
Each completed line is saved and published immediately; a failed line reports an
error without preventing other lines or subsequent captures from processing.
Returning hands never invalidate accepted captures or hide their results.

The top PAUSE/PLAY toggle pauses new automatic captures only. Queued/running
inspections finish, results keep arriving, live preview and bed-marker monitoring
continue, and result zoom/pan remain available. Resuming requires a fresh hand
cycle. Settings/calibration navigation still suspends the session and cancels
pending work; this is separate from PAUSE.

While results or inspections exist, the trigger thread also detects bed markers
using `DICT_4X4_50`. Two physical ID-0 markers must remain detected for 0.5 seconds
(`auto_trigger.marker_confirm_seconds`). Duplicate IDs are counted separately.
This latches a reset request: no further captures are accepted until reset.
Hands returning or markers becoming covered do not cancel the request.
If work remains, the display says "Waiting for inspection results…". After the
queue drains, the final results remain visible for 2 seconds, then all tabs clear
and full live view returns. An empty fabric classification does not reset tabs.

Each capture stores line artifacts in `line_1/`, `line_2/`, etc. with overlay,
mask, preview and measurement JSON; the capture-level result records all lines.

## Configuration

Press S during startup, or edit `config.json` and restart. The Local Inspection
settings tab replaces the old segmentation/SAM controls.

| Setting | Default | Meaning |
|---|---|---|
| `local_inspection.source_model` | `inspection/local/models/strip_unet_resnet34.onnx` | Local source-region weights |
| `local_inspection.glue_model` | `inspection/local/models/segformer_b0.onnx` | Local unfolded adhesive weights |
| `local_inspection.source_width` | 1152 | Aspect-preserving source AI width |
| `local_inspection.offset_pixels` | 100.0 | Inward band width in full undistorted-frame pixels |
| `local_inspection.subsequent_line` | rightmost | Side inspected when a session already has results/work; leftmost also supported |
| `auto_trigger.marker_confirm_seconds` | 0.5 | Continuous visibility of two physical ID-0 bed markers before reset |
| `auto_trigger.hand_absence_seconds` | 1.0 | Quiet period after debounced hand absence |
| `auto_trigger.debounce_frames` | 3 | Consistent fresh samples per presence change |
| `inspection.end_exclusion_percent` | 5.0 | Percentage excluded from EACH end before width grading |
| `inspection.strip_width_mm` | 4.0 | Target width for the new measurements |
| `inspection.strip_width_tolerance_mm` | 1.0 | Allowed +/- width tolerance |
| `result_view.show_live_preview` | true | Small live camera panel beside results |
| `capture_storage.save_rejected_triggers` | false | Preserve sources for rejected/error triggers too |
| `capture_storage.max_sets` | 200 | Whole capture sets retained; the oldest is deleted once a newer one is started |

Model paths are relative to the application source folder (absolute paths are also
accepted). Calibration/capture paths use the existing Documents data root.
Legacy `segmentation`, `sam_detection`, crop, serial and tab-timing configuration
is retained for old helpers/compatibility but does not drive this inspection.
The reference filter, preprocessing, cleanup and measurement constants are in
`inspection/local/`; changing them changes the reference recipe.

## Saved data

Each processed accepted snapshot uses a timestamped directory under Windows
Documents / `data files/Dataset_capture/Automatic`:

```text
.inspection-set
capture.json       # trigger frame sequence, timestamp, fabric verdict, profile
original.png       # full-resolution snapshot with configured camera rotation
undistorted.png    # corrected version of the SAME snapshot
result.json        # capture status and all selected-line measurements/errors
line_1/            # one directory per inspected line (left-to-right index)
    mask.png       # lossless full-resolution refolded adhesive mask
    overlay.png    # annotated widths and tolerance highlights
    overview.jpg   # compact preview
    result.json    # line status and global-frame measurements
line_2/            # present when additional lines were inspected
```

An error still saves the source pair and error metadata. No intermediate source
masks, unfolding maps or temporary images are written during normal inspection.
Retained sets include completed/error captures. Retention preserves in-flight
sets and deletes only completed recognized sets using the existing safe policy.
This app automatically saves captures/results; the reference app's manual-only
save policy has deliberately not replaced that behavior.

## Dependencies and verification

The two copied models use CPU ONNX Runtime 1.30.0, listed in `requirements.txt`.
The working runtime was copied from the reference into this project's `.venv`;
models load once in the background when the automatic controller starts. Hand
and H5 fabric models still use existing MediaPipe/TensorFlow dependencies.

Run with the project interpreter:

```text
python -m unittest discover -s tests -p "test_*.py"
python tests/verify_automatic_ui.py
python tests/verify_startup_settings.py
python tests/verify_local_models.py --reference PATH_TO_REFERENCE --image IMAGE
```

The last check uses actual copied ONNX models and compares full-size masks,
overlays, line selection, components, pixel lengths and width segments against
the reference on exactly the same image. The reference argument is needed only
for this developer comparison, never for application startup or inspection.

Tests cover all/leftmost/rightmost selection, full-frame refolding, component gaps, metric
endpoint conversion, missing calibration/segments, lossless storage, trigger
FIFO queues, pause, duplicate-ID markers, delayed reset, and the result/settings UI. Physical hand/fabric
triggering and millimetre accuracy must still be verified with representative
fabrics on the installed camera and calibrated measurement surface.


Capture preparation runs on a separate fabric-check worker. Hand/bed monitoring
continues while fabric classification, undistortion, saving, or inspection is busy.
Hand cycles retain their own frame and profile snapshots and enter the preparation
queue in order; accepted captures then enter the inspection FIFO. Pause stops new
triggers while already captured work drains. The persistent queue bar shows captures
checking (including waiting for the fabric worker), waiting for inspection, and
currently inspecting, including while results are displayed.
