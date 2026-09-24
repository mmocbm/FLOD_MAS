# Application settings

Edit `config.json` in the main directory, then restart the application.

## Cameras

The first camera is the dashboard's left view; the second is the right view.
For each camera, set `index`, `width`, `height`, and `fps` to a mode supported by
your camera. Defaults are 2560 × 1440 at a requested 30 FPS, matching the existing
calibration image sizes. `fps` is a driver request, not a guaranteed throughput.
`fourcc` can be left blank to use the driver's format, or set to a supported
four-character format such as `MJPG`. Calibration and surface files are relative
to this directory unless an absolute path is supplied.

`rotation` sets each camera's clockwise software rotation. Allowed values are `0`,
`90`, `180`, and `270`. Rotation is applied in the shared camera stream, so setup,
preview, crops, calibration, and inspection all use the same orientation. A 90° or
270° rotation swaps the output width and height. After changing rotation, recalibrate
that camera and recreate its saved crop regions; calibration from another orientation is
not geometrically valid.

All camera acquisition goes through the same configured capture path. The app
checks actual frame sizes. If the requested size is unavailable, it tests the
`fallback_resolutions` list and selects the largest actual size that works (by pixel
count). `verification_frames` controls how many frames are read after each mode
change. The app silently continues with the best verified size. Setup shares the
selected mode and does not reopen the camera.
This is the best verified mode from the configured candidates, not an exhaustive
enumeration of every proprietary camera mode. Add other supported sizes to the
list if needed. Searching occurs only when the initial requested mode fails and
can increase startup time; the interface stays responsive during the search.

Set `capture.use_dshow` to `true` to open both cameras through Windows DirectShow.
Set it to `false` to call OpenCV's normal `VideoCapture(index)` and let OpenCV choose
the default camera backend. Restart the application after changing this setting.

The first successful result for each camera is saved in `resolution_cache_file`.
Later launches try that verified size immediately, avoiding another full search.
Changing the requested size, frame rate, format, `use_dshow`, or fallback list makes
the saved result invalid and automatically performs a fresh search.

Camera threads continuously retain the newest full-resolution image. Camera Setup
borrows these streams, so switching pages does not reopen devices. The dashboard
starts the devices in the background on first launch. Full-size photos are used
for saving, calibration, crops and measurement. Lens correction is computed when
a processing operation requests it, not on every live preview frame.

## Preview

`max_width` and `max_height` limit copies used for dashboard/setup previews;
`interval_ms` controls preview refresh (66 ms is approximately 15 updates/second).
Lower these for lighter display work. They do not change the camera capture size
or the original image used for calculations. Crop editing still maps its display
coordinates back to the full-resolution captured image.

## Manual crop setup

`crop_setup` configures the two independent model-input regions used for each
camera. `aspect_ratio` is fixed at `[4, 1]`, and `output_size` is `[2208, 552]`, so
deskewing and resizing use the same ratio and therefore apply one uniform scale.
Crop definitions are stored in `Files/crop_regions.json` by default.

In **Crop Setup**, select a camera and Crop 1 or Crop 2, capture an undistorted
frame, and drag the red 4:1 region. The region can be moved by dragging inside it
or resized from a corner. Select **Set Line** and click two points along an edge
that should become horizontal, then save the crop. Repeat for all four regions.

Inspection saves one timestamp-matched original image, undistorted image, and two
deskewed crops under `Dataset_capture/CameraN/{Original,Undistorted,Crop1,Crop2}`.
The two crops are prepared independently at 2208 × 552 and displayed vertically.
The crops selected in `sam_detection.send_crops` are then sent for object
detection, described in the next section.

## SAM object detection

`sam_detection` sends selected crops to the Roboflow serverless workflow
`obhashcolab-1/sam3-with-prompts`, which runs SAM3 with a text prompt and returns
polygons. It runs inside the inspection freeze, after the crops are saved and
before the view is restored, so the operator sees a progress line while it works.

- `enabled`: master switch. When false, inspection stops after saving crops and no
  request is made.
- `send_crops`: per camera, which of the two crops to upload — for example
  `{"1": [true, false], "2": [true, false]}` sends only Crop 1 from each camera.
  Crops are listed in `Crop1`, `Crop2` order.
- `prompt`: one text prompt applied to every uploaded crop. Commas separate
  multiple classes; a class name may contain spaces.
- `jpeg_quality`: 1–100. Crops are re-encoded as JPEG at this quality before
  upload, which shrinks the upload without changing its pixel dimensions, so
  detection still sees full 2208 × 552 detail. Lower it to send less data, raise
  it if fine detail is being lost.
- `timeout_seconds`: how long to wait for one crop before abandoning the request.
- `save_overlay` / `save_polygons`: whether to write the annotated image and the
  JSON result beside the saved crops.
- `analyze_strip`: whether to measure the adhesive strip. When true, the largest
  returned polygon is treated as the strip and a centreline, ten segment
  boundaries and a width label per segment are drawn on the overlay.
- `strip_segments`: how many equal-length segments to cut the strip into. Ten by
  default. More segments give finer resolution along the strip and shorter spans
  to average over, so the per-segment figures get noisier.

### Strip measurement

SAM3 returns one dense ring tracing both edges of the strip, which says nothing
about where the strip runs or how thick it is. The ring is rasterised,
skeletonised to its medial axis and ordered into a single path, which is smoothed
and cut into `strip_segments` equal-length pieces. Width is measured across the
strip perpendicular to that centreline, not along an image axis, so it stays
correct where the strip curves.

Every figure is in **source-image pixels**, measured on the 2208 × 552 crop.
Converting to millimetres needs a pixels-per-millimetre factor for the crop, which
is not yet applied.

Two limits are worth knowing. The centreline is trimmed by half a strip width at
each end, because a skeleton sprouts short forks at a flat strip end; the reported
length is therefore slightly shorter than the physical strip. And because segment
boundaries land on whole centreline samples, segment lengths can differ from each
other by one sample's worth of arc length.

Results land in `Dataset_capture/CameraN/Detected/` as `crop_<timestamp>_<n>.png`
and `.json`. The JSON holds the prompt, workflow, upload size, image size and every
polygon with its class, confidence, area and points, so results can be compared
across runs. When a strip was measured, a `strip` block is added alongside: which
polygon it came from, the total length and the average, minimum and maximum width,
and then one entry per segment with its own length, widths, sample count and
midpoint. It is `null` when nothing measurable was found.

Detection is fail-soft: a missing key, network error, timeout or unusable response
never fails the inspection. The raw crop is displayed and saved as usual, the JSON
record is still written with its `error` field set, and the dashboard shows a
warning instead of a PASS or FAIL. This keeps a dead API distinguishable from a
frame in which nothing was found.

The API key is never stored in `config.json`, because that file is committed. Put
it in a `.env` file in the project root (already listed in `.gitignore`):

```
ROBOFLOW_API_KEY=your_key_here
```

An environment variable of the same name takes precedence over the file. The key
is sent as an authorization header, never in the URL.

## Board and camera setup

`board` describes the physical printed board: number of squares, square and marker
sizes in millimetres, and OpenCV dictionary name. These must match the actual board.
`calibration` controls the required photo count, minimum valid photos/corners and
quality checks. Numerical quality thresholds are for maintainers; operators see
simple instructions in the app. Standalone board tools read the same parameters.
After selecting a camera and starting its live feed, **Open Calibration Checks**
opens the combined lens and real-world measurement checker. The lens check captures
one fresh ChArUco-board image and reports the board-pose reprojection RMS. A
result at or below `calibration.verification_max_rms_px` (default `1.0` pixels)
means the saved lens calibration is suitable for that test view; a higher result
recommends recalibration.

After the measurement surface has been saved, **Check Measurement Accuracy** uses
the saved fixed-plane pose instead of estimating a new pose from the test image. It
compares detected ChArUco geometry with known short and long board distances and
reports measured distance, signed/absolute millimetre error, percentage error,
mean absolute error, RMSE and maximum error. It intentionally does not assign a
PASS/FAIL grade. In two-board mode both uniquely numbered boards must be visible;
distances are checked inside each board because their separation is not fixed.

`two_board.enabled` selects the initial Camera Setup method. The operator can also
switch between **One Board** and **Two Boards** directly in Camera Setup before the
first accepted photo. One-board mode remains the original workflow. Two-board mode
uses two boards with separate marker-number ranges so the app can tell them apart.
Do not use two copies of the same printed board.

Generate the matching printable images with:

```powershell
.\.venv\Scripts\python.exe .\CalibrateAPP\generate_two_boards.py
```

The files are written under `Files/calibration_boards`. Print both at exactly
385 × 245 mm for the current 11 × 7, 35 mm-square configuration, with no
fit-to-page scaling. `two_board.dictionary` must contain enough unique markers for
both boards. `second_board_start_id` starts Board 2's non-overlapping marker range.
`mask_padding_px` controls the automatic masked retry around a board already found.

The coverage grid rows and columns divide the preview into areas used to recommend
the next board position. Increasing the grid dimensions produces more detailed
guidance but also requires more photos to cover every area.

The live setup preview does not run board detection. Detection starts only after the
operator presses **Capture Photo**. The full-resolution capture is saved as a lossless
PNG, loaded back from that path, and then checked. Valid files remain in the session
folder; rejected files are deleted. Points found in accepted captures remain overlaid
on later live frames as placement guidance. The full camera calculation runs only
after all required photos have been accepted.

In two-board mode, each accepted photo set supplies two independent board poses to
the camera calculation. If both boards are clear in one photo, that photo completes
the set. If only one is clear, the app keeps it and asks for the other board in a
second photo. The app masks the first detected board area and retries the other
detector automatically before requesting the second photo.

## Measurement surface mode

`measurement_surface.enabled` selects which plane calibration is prepared for the
later measurement stage:

- `true`: Camera Setup requires the existing flat-board surface photo and saves its
  pose for later millimetre conversion.
- `false`: Camera Setup finishes after the camera photos; the configured ArUco marker
  remains available for a future per-inspection plane calculation.

The current inspection implementation prepares and saves the two deskewed crops
only. It does not yet run the model, calculate millimetres, or make PASS/FAIL
decisions; those stages will be connected after crop preprocessing is validated.

`measurement_surface.board_thickness` controls the optional correction from the
top of the ChArUco board to the bed beneath it. It is disabled by default. When the
operator enables **Correct board thickness to bed plane** before saving the surface,
the entered thickness is used to move the stored product-measurement plane away
from the camera. The unshifted board-top pose is also saved so a later accuracy
check can correctly measure a verification board resting on the same bed.

The default marker is `DICT_4X4_50`, ID `0`, with a 25 mm side.
`aruco_marker_length_mm` must equal the measured outer black-square side of the
printed marker or every result will have a scale error. The marker and product must
be flat on the same plane, and the full marker must be visible during inspection.
`minimum_marker_side_px` rejects markers that are too small for reliable use.

If you change the capture mode, lens focus, physical board, camera position or
measurement surface, run Camera Setup again and verify real measurements. Matrix
scaling supports resized images, but cannot compensate for a camera driver changing
its crop or field of view. Recreate saved crop regions if the view changes.

## Other settings

- `serial`: Arduino connection enabled, port and baud rate.
- `inspection`: selectable sizes, starting size, tolerances and segment count.
- `color_mask`: legacy values retained for the standalone colour-mask helper; the
  dashboard's new crop workflow does not use them.

Keep JSON syntax valid (double quotes, no trailing commas). Invalid key settings
are rejected on startup. No extra packages are required for configuration.
