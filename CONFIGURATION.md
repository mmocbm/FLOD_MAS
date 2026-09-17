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

The first successful result for each camera is saved in `resolution_cache_file`.
Later launches try that verified size immediately, avoiding another full search.
Changing the requested size, frame rate, format, backend, or fallback list makes
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
