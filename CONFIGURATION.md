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
camera. Crop definitions are stored in `Files/crop_regions.json` by default.

`aspect_ratio` selects the shape of a crop and `output_size` the resolution it is
resized to. They default to `[4, 1]` and `[2208, 552]`, and **must describe the same
ratio**: deskewing and resizing are one transform, so a mismatch would not fail
loudly, it would stretch every crop by a small amount. The app rejects a
configuration where the two disagree and names both values.

`margin_percent` grows the marked region on all four sides, so the fabric never sits
flush against the crop edge. It is a percentage of the marked size: `8` adds 8% of
the marked width to each side horizontally and 8% of the marked height vertically.
The default is `8`. The margin is applied when the region is marked, so it is baked
into the saved crop — see the note on adjusting it below.

### Marking a region

In **Crop Setup**, select a camera and Crop 1 or Crop 2, then press **Capture** to
freeze and undistort one frame.

Press **Mark 4 Points** and click the four corners in order around the region:

1. the first corner of the region
2. the next corner along that end edge — clicks 1 and 2 are the boundary
3. the corner across the region — clicks 2 and 3 run along the long side
4. the last corner, level with point 1

Clicks 2 and 3 set the deskew: that line is what becomes horizontal. The crop is then
the marked region, grown by `margin_percent` and fitted to `aspect_ratio`. It can be
moved by dragging inside it, or resized from a corner, and **Save Crop** stores it.
Repeat for all four regions.

Every step only ever grows the box, never shrinks it, so whatever was marked plus its
margin is always inside the crop. If the margin would push the crop past the edge of
the image the app says so and refuses to save rather than silently clipping the
region or quietly moving it; mark further from the edge, or lower `margin_percent`.
A mark the app cannot make sense of — corners out of order, points too close, the
short side clicked where the long side belongs — is reported with the reason and the
clicks are left on screen so the mistake is visible. Pressing **Mark 4 Points** again,
or clicking once more, starts over.

The crop region must stay inside the image, and the framing is measured, so a sloppy
mark is widened rather than losing fabric. The marked corners are stored with the
crop, which is what allows the two settings below to be changed without re-marking.

### Changing the ratio or the margin afterwards

`aspect_ratio` is not stored per crop: only a normalized width is, and the height is
re-derived from the configured ratio when a crop is loaded. **Changing `aspect_ratio`
therefore reinterprets every saved crop**, and the app says so when Crop Setup is next
opened. Re-mark all four regions after changing it.

`margin_percent` is stored, but the marked corners are kept alongside it, so changing
it re-derives the crop from the original mark — the larger or smaller margin appears
the next time crop setup is opened, and **Save Crop** persists it. A region that was
adjusted by hand after marking has no corners left to re-derive from and keeps the
box it was given. Changing either setting does not alter anything until a crop is
saved, so an inspection can never be changed by an unsaved edit.

Inspection saves one timestamp-matched original image, undistorted image, and two
deskewed crops under `Dataset_capture/CameraN/{Original,Undistorted,Crop1,Crop2}`.
The two crops are prepared independently at the configured output size and displayed
vertically. The crops selected in `sam_detection.send_crops` are then sent for object
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
- `measure_in_mm`: whether to report the strip in millimetres. When true, the
  overlay labels widths in mm to two decimals and the JSON gains `_mm` figures
  beside the pixel ones. When false, or when the calibration needed for it is
  missing, everything stays in pixels. See below.

### Strip measurement

SAM3 returns one dense ring tracing both edges of the strip, which says nothing
about where the strip runs or how thick it is. The ring is rasterised,
skeletonised to its medial axis and ordered into a single path, which is smoothed
and cut into `strip_segments` equal-length pieces. Width is measured across the
strip perpendicular to that centreline, not along an image axis, so it stays
correct where the strip curves.

#### Millimetres

With `measure_in_mm` on, the crop is mapped onto the **calibrated measurement
plane** — the plane the calibration board sat on — using that camera's intrinsics
and extrinsics. Each crop pixel picks out a ray, and the ray crosses that plane at
one point, so a pixel becomes a real position on the plane. The crop warp and the
plane intersection are both projective, so they compose into a single homography,
built by probing the tested intersection at the crop's four corners.

Widths and lengths are then recomputed from mapped points rather than scaled by a
single factor, because a pixel is **not** a fixed number of millimetres across a
projected crop. The `mm_per_pixel` figure is a summary at the crop centre, not a
constant.

The overlay then shows `3: 3.10mm`. The JSON keeps the pixel figures alongside the
millimetre ones: they are what the scale was applied to, so they make a
millimetre value auditable and let it be recomputed if a calibration changes.

This measures the plane, so it is only valid for a strip lying **on** that plane.
A strip standing proud of it, a moved camera, or a re-aimed camera all invalidate
the answer. Crops whose region the board never covered are extrapolated from the
plane and should be treated with more caution than ones the board sat in.

Millimetres are only produced when that camera's calibration and extrinsics files
both exist and are readable. Otherwise the run falls back to pixels and records
the reason as a warning, so a missing calibration degrades the units rather than
losing the inspection.

Three limits are worth knowing. The centreline is trimmed by half a strip width at
each end, because a skeleton sprouts short forks at a flat strip end; the reported
length is therefore slightly shorter than the physical strip. Because segment
boundaries land on whole centreline samples, segment lengths can differ from each
other by one sample's worth of arc length. And widths read about 0.3 px long per
edge, from rasterising the polygon to a whole-pixel mask — that is a fraction of a
percent on a wide strip but several percent on one only a few pixels across.

Results land in `Dataset_capture/CameraN/Detected/` as `crop_<timestamp>_<n>.png`
and `.json`. The JSON holds the prompt, workflow, upload size, image size and every
polygon with its class, confidence, area and points, so results can be compared
across runs. When a strip was measured, a `strip` block is added alongside: which
polygon it came from, whether it is metric, the total length and the average,
minimum and maximum width, and then one entry per segment with its own length,
widths, sample count and midpoint. Each of those carries both a pixel figure and a
`_mm` one; the `_mm` fields are `null` unless a millimetre scale was applied. The
whole block is `null` when nothing measurable was found.

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

## Region homography mode (optional)

Everything above converts crop pixels to millimetres one way: one global lens
calibration plus one global measurement-plane pose, composed into a single
homography per crop region. That assumes the bed is one flat plane and that the
lens model fits every part of the image equally well.

`region_homography.enabled` adds a second, independent way to get millimetres for
each crop region. A small ChArUco board is laid flat *inside* a region, and a
homography is fitted directly from the corners the camera actually sees there. The
intrinsics stay global; each region gets its own homography, and measurement uses
that region's.

**This does not replace the existing workflow.** With `enabled: false` — the
default — nothing changes anywhere in the application. Both modes can be used, and
compared, without either being removed.

### Configuration

```json
"region_homography": {
  "enabled": false,
  "store_file": "Files/region_homographies.json",
  "board_profiles_file": "Files/board_definitions.json",
  "default_profile": "Region board 17x7",
  "minimum_corners": 8,
  "maximum_rms_mm": 0.5,
  "maximum_tilt_degrees": 2.0,
  "maximum_gap_mm": 2.0,
  "maximum_scale_error_percent": 2.0,
  "refine_intrinsics": false,
  "minimum_refinement_views": 5
}
```

`minimum_corners` is the fewest ChArUco corners a capture may yield and still be
fitted; `maximum_rms_mm` grades the fit itself. The next three are the save gates,
described below. `refine_intrinsics` is off by default and there is a good reason
for that — see the limits.

### Calibrating a region

Open **REGION CALIBRATION** from the settings camera card. The page shows the
**deskewed crop of one region**, not the full camera frame, because the crop is
exactly the raster the homography is fitted to; showing anything else would mean
the operator places the board against something other than what is measured.

The page is gated on the crop regions: if either region for that camera has not
been marked, every action is disabled and the banner names the missing region.
Mark them in manual crop setup first.

Pick a region, print its board, lay it flat inside the region and press **CAPTURE
& FIT**. Regions are calibrated one at a time — region 01 with its board, then
region 02 with its board. Only one board is needed.

The board is described by named profiles in `board_profiles_file`, and every field
is editable in the page, so a new physical board can be added without editing code.
The default profile is DICT_5X5_100, 17 × 7, 15 mm checker, 11 mm marker. Sizes are
the printed sizes: the checker figure must be the actual square pitch, because a
board printed at 10 mm squares but described as 15 mm behaves differently, and
worse, silently (see below).

The page shows the board's footprint against the region's measured extent and turns
red when the board cannot fit, with a **SHRINK TO FIT** button that computes the
largest whole square size that does.

### The three gates

A fit is graded before it may be saved, and each gate catches a failure that leaves
no trace in the residuals:

- **Residual RMS** against `maximum_rms_mm`. The obvious one, and the weakest.
- **Flatness.** The board's own plane is compared with the saved measurement plane
  and must be within `maximum_tilt_degrees` and `maximum_gap_mm`. A board propped
  up at an angle or resting on a shim fits *perfectly* — the homography describes
  the board's plane instead of the measurement plane — and every later measurement
  is then wrong by an amount nothing in the stored data reveals. This is the gate
  that matters most.
- **Scale agreement.** The fit's millimetres-per-pixel is compared against the
  global plane's, within `maximum_scale_error_percent`. This is the *only* check
  that can catch a board declared at the wrong size: a mis-declared board still
  lines up exactly with the coordinates the wrong profile assigns it, so the
  homography absorbs the factor and fits to a ten-thousandth of a millimetre while
  reporting every length wrong by that factor.

A failing gate refuses to save. The operator can insist past the flatness and scale
gates with a confirmation, which is deliberate: a blocked measurement is worse than
a flagged one, as long as the flag is recorded.

### Saved data and invalidation

`store_file` holds one entry per (camera, region), each carrying the homography,
its residual metrics, the board that produced it, the intrinsic evaluation, and a
`crop_signature`. The signature is a hash of the crop definition and the frame size
it was fitted at.

**Re-marking a crop region invalidates that region's homography**, and the
signature is how that is detected. A stored homography describes pixels in the crop
raster it was fitted on; move the region's corners and the same pixels are now a
different part of the bed, so the stored millimetre scale is quietly wrong. The
region is reported as stale rather than used, and must be re-calibrated.

The homographies live in their own file, not inside the crop definitions, because
`crop_processing` rebuilds each crop definition from scratch when a region is
saved: anything stored inside one would be silently dropped the next time that
region was re-marked.

### Limits worth knowing

- **Validity is local.** A region homography is exact only for objects lying on the
  plane the board was on, within the region it was fitted on. Extrapolating outside
  that is unchecked — the same caveat the global plane already carries.
- **One view has no redundancy.** There is no outlier rejection in the sense of a
  second opinion; a mis-detected corner biases the homography directly. Mitigated
  by a ChArUco board yielding dozens of corners from a single view, and by gates
  above.
- **A single flat view cannot refine the lens.** `refine_intrinsics` needs several
  *tilted* views and refuses below `minimum_refinement_views`, reporting why. This
  is measured, not assumed: against a deliberate focal-length error, the local fit
  is blind to it *exactly* (identical residuals whether the focal length is right
  or five times wrong), and the lens evaluation does see it but far too weakly to
  refine against — a 15% error reads as about half a pixel, which is inside
  detection noise. Leave refinement off unless you have captured tilted views
  deliberately. When it does run it is stored beside the global calibration and
  never overwrites it.

`tests/verify_region_calibration.py` measures all of the above against a synthetic
rig with exactly known truth, and prints the numbers rather than asserting them.

### Checking a region

The calibration checks page gains a **Region homography** section alongside the
existing lens and measurement-accuracy checks, which are unchanged. It captures a
fresh image, maps the board's detected corners through the *saved* region
homography, and reports measured against known distances in millimetres — graded
against `maximum_rms_mm`. It also runs the same board through the global plane for
comparison, so the two modes can be seen side by side.

## Other settings

- `serial`: Arduino connection enabled, port and baud rate.
- `inspection`: selectable sizes, starting size, tolerances and segment count.
- `color_mask`: legacy values retained for the standalone colour-mask helper; the
  dashboard's new crop workflow does not use them.

Keep JSON syntax valid (double quotes, no trailing commas). Invalid key settings
are rejected on startup. No extra packages are required for configuration.
