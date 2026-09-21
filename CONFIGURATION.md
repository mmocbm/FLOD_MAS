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
the default camera backend. An individual camera can override this with its own
`use_dshow` value. Camera 1 currently has `use_dshow: false` because DirectShow
could not open that index; Camera 0 follows the global DirectShow setting. If a
DirectShow-selected camera cannot open, the app retries that index with the default
backend. Restart the application after changing these settings. Camera indexes
can differ between backends, so verify that the left and right previews show the
intended devices.

The first successful result for each camera is saved in `resolution_cache_file`.
Later launches try that verified size immediately, avoiding another full search.
Changing the requested size, frame rate, format, effective backend, or fallback list makes
the saved result invalid and automatically performs a fresh search.

Camera threads continuously retain the newest full-resolution image. Camera Setup
borrows these streams, so switching pages does not reopen devices. The dashboard
starts the devices in the background on first launch. Full-size photos are used
for saving, calibration, masks and measurement. Lens correction is computed when
a processing operation requests it, not on every live preview frame.

## Preview

`max_width` and `max_height` limit copies used for dashboard/setup previews;
`interval_ms` controls preview refresh (66 ms is approximately 15 updates/second).
Lower these for lighter display work. They do not change the camera capture size
or the original image used for calculations. Mask editing still maps its display
coordinates back to the full-resolution captured image.

## Board and camera setup

`board` describes the physical printed board: number of squares, square and marker
sizes in millimetres, and OpenCV dictionary name. These must match the actual board.
`calibration` controls the required photo count, minimum valid photos/corners and
quality checks. Numerical quality thresholds are for maintainers; operators see
simple instructions in the app. Standalone board tools read the same parameters.

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

`measurement_surface.enabled` selects how image pixels are converted to millimetres:

- `true`: Camera Setup requires the existing flat-board surface photo and inspection
  uses its saved position.
- `false`: Camera Setup finishes after the camera photos. Every Inspect click must
  see the configured 4×4 ArUco marker; that frame's marker defines the measurement
  plane and scale.

The default marker is `DICT_4X4_50`, ID `0`, with a 25 mm side.
`aruco_marker_length_mm` must equal the measured outer black-square side of the
printed marker or every result will have a scale error. The marker and product must
be flat on the same plane, and the full marker must be visible during inspection.
`minimum_marker_side_px` rejects markers that are too small for reliable use.
If the marker is unavailable, inspection continues without a popup and reports only
pixel distances. The dashboard shows a warning because millimetre results and metric
PASS/FAIL decisions are not calculated for that frame.

If you change the capture mode, lens focus, physical board, camera position or
measurement surface, run Camera Setup again and verify real measurements. Matrix
scaling supports resized images, but cannot compensate for a camera driver changing
its crop or field of view. Recreate saved masks if the view changes.

## Other settings

- `serial`: Arduino connection enabled, port and baud rate.
- `inspection`: selectable sizes, starting size, tolerances and segment count.
- `color_mask`: hue, saturation and brightness tolerances for selecting material.

Keep JSON syntax valid (double quotes, no trailing commas). Invalid key settings
are rejected on startup. No extra packages are required for configuration.
