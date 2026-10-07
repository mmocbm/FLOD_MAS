# Automatic local ONNX inspection

Run `run_automatic.bat`, or `.venv\Scripts\python.exe main_1366.py`.
The existing single-camera acquisition, rotation, lens calibration and saved
measurement-plane setup remain in use. Automatic inspection has no Roboflow
request, API key, manual mask selection or fixed inspection crops.

## Operator cycle

1. A hand appears and leaves. Three fresh agreeing samples debounce presence;
   hands must then stay absent for 1 second. A hand-free scene does not retrigger.
2. Copy that exact camera frame and classify fabric on the original snapshot.
   `full_fabric` or `half_fabric` at confidence >=0.5 proceeds. Confident `no_fabric`
   clears the result and pending capture. An uncertain verdict retains the result.
3. Undistort the accepted snapshot using the existing camera calibration.
   Missing camera calibration prevents inspection; there is no raw-frame fallback.
4. Run the local source-region model at aspect-preserving width 1152, with reflected
   padding to multiples of 32 and ImageNet RGB normalization. One inference pass
   produces a full-frame mask; threshold 0.5, minimum region area 0.2%, hole filling.
5. Extract curved boundaries, remove straight backing edges/caps and endpoint
   hooks (4% endpoint zone, 65 degree hook angle). Keep wave lines with minimum wave
   ratio 0.008, span 10% of the larger frame dimension, and boundary support 90%.
6. Sort accepted lines by mean x and inspect only the RIGHTMOST line. Boundary
   count is not fabric count. No four-line minimum or maximum is imposed.
7. Offset 100 undistorted-frame pixels into the source-mask interior. Apply CLAHE
   to source lightness (clip 2, 16x16 tiles), then unfold this curved band horizontally.
   Unfolded height follows the offset; width follows the band's middle arc length.
8. Run the local two-class SegFormer at its trained 1024x64 input and resize its
   mask back to the actual unfolded dimensions. Clean noise/spikes while preserving
   thickness changes and long missing intervals. Cleanup window 9 and short-gap
   limit 12 are scaled from model-width pixels; smoothing tolerance is 0.15.
9. Refold the cleaned mask into the full undistorted frame. Clip to the selected
   band. Draw adhesive boundaries red and measured centrelines yellow.
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

PASS requires an available average metric width for EVERY one of the ten segments,
with each average within the inclusive target +/- tolerance (default 4 +/- 1 mm).
Any measured out-of-range segment produces FAIL when all segments are available.
Missing plane calibration or incomplete segment measurements gives UNMEASURED
and a dashboard WARNING, while pixel results remain available. This grades widths
on the longest component; it does not introduce a continuity/gap acceptance rule.

## Results and lifecycle

The result viewer shows one selected line, with OVERLAY and MASK views, zoom,
pan, fit/reset, component/line counts, timing, total length, longest-component
mean width, and a scrollable ten-segment average/minimum/maximum width table.
Both pixel and calibrated millimetre values are shown. The optional live side
preview continues showing current frames; the large result uses the captured
undistorted frame. Overlay/mask selection is manual; there is no fabric tab timer.

One inspection runs at a time, with at most one replaceable accepted capture
waiting. A hand returning invalidates unfinished work and cancels an obsolete
pending capture. Finished obsolete captures are still saved but never displayed.
Obsolete errors cannot clear a newer result. A completed result stays visible
until replacement, confident no-fabric, a CURRENT inspection error, or reset.
Opening settings pauses/resets the trigger and cancels pending work. Camera
capture and hand monitoring continue during background ONNX inspection; the
fabric gate runs synchronously in the hand-monitor thread.

## Configuration

Press S during startup, or edit `config.json` and restart. The Local Inspection
settings tab replaces the old segmentation/SAM controls.

| Setting | Default | Meaning |
|---|---|---|
| `local_inspection.source_model` | `inspection/local/models/strip_unet_resnet34.onnx` | Local source-region weights |
| `local_inspection.glue_model` | `inspection/local/models/segformer_b0.onnx` | Local unfolded adhesive weights |
| `local_inspection.source_width` | 1152 | Aspect-preserving source AI width |
| `local_inspection.offset_pixels` | 100.0 | Inward band width in full undistorted-frame pixels |
| `auto_trigger.hand_absence_seconds` | 1.0 | Quiet period after debounced hand absence |
| `auto_trigger.debounce_frames` | 3 | Consistent fresh samples per presence change |
| `auto_trigger.fabric_min_confidence` | 0.5 | Fabric acceptance and empty-table confidence |
| `inspection.strip_width_mm` | 4.0 | Target width for the new measurements |
| `inspection.strip_width_tolerance_mm` | 1.0 | Allowed +/- width tolerance |
| `result_view.show_live_preview` | true | Small live camera panel beside results |
| `capture_storage.save_rejected_triggers` | false | Preserve sources for rejected/error triggers too |
| `capture_storage.max_sets` | 1000 | Whole capture sets retained |

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
mask.png           # lossless full-resolution refolded adhesive mask
overlay.png        # lossless full-resolution boundary/centreline overlay
overview.jpg       # compact preview
result.json        # processing status, settings, global-frame samples,
                   # selected boundary, component lengths, px/mm widths and grades
```

An error still saves the source pair and error metadata. No intermediate source
masks, unfolding maps or temporary images are written during normal inspection.
Retained sets include superseded/error captures. Retention preserves in-flight
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

Tests cover rightmost selection, full-frame refolding, component gaps, metric
endpoint conversion, missing calibration/segments, lossless storage, trigger
queues, stale results/errors, and the result/settings UI. Physical hand/fabric
triggering and millimetre accuracy must still be verified with representative
fabrics on the installed camera and calibrated measurement surface.
