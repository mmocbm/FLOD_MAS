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
