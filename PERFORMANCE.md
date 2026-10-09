# Low-power performance review

## Changes made

| Source of load | Change | Tradeoff |
| --- | --- | --- |
| Two 4608 × 3456 cameras requesting 30 FPS continuously | Request 10 FPS | Less fluid video; driver must support the requested rate |
| Full-frame area averaging on the dashboard's Tk thread | Linear interpolation for live previews | Preview can show aliasing; inspection pixels are unchanged |
| Dashboard ignored configured preview size limits | Enforce 880 × 540 maximum, including maximized view | Maximized preview has less display detail |
| Same camera snapshot repeatedly resized and uploaded to Tk | Skip unchanged snapshots unless canvas geometry/content changes | Applies to immutable snapshots from CameraStream; rotated reads create new arrays |
| Hidden cameras still read/rotated by the dashboard | Read only visible cameras; skip reads during frozen crop editing | Acquisition still drains both devices for freshness |
| Slow rendering followed immediately by another callback | Leave at least 100 ms idle after each dashboard update; back off further under load | Display rate falls below 10 FPS on slow machines |
| OpenCV kernels potentially recruit many CPU workers | Default to one OpenCV internal thread | Some inspections/calibrations may take longer |
| Animated setup scrolling and inspection spinner | Immediate scrolling and static busy indicator | Progress updates and automatic result sequencing remain available |

Full-resolution capture, crop extraction, calibration and measurement algorithms
are preserved. No extra image processing runs continuously for inspection.

## Evidence

306 unit/regression tests passed, including camera capture, crop processing,
calibration, inspection and new preview workload checks. Hardware and live network
inference were not exercised by these tests.

A synthetic benchmark on the development machine resized a random uint8 BGR
4608 × 3456 frame to 660 × 495, using OpenCV 4.12.0 and one thread. After five
warmups, median times over 30 samples were:

- Previous area interpolation: **34.757 ms/frame**.
- New linear interpolation: **1.028 ms/frame** (about **34× faster** for resizing).

This measures only resizing, not end-to-end application speed or total PC CPU
usage. Capture, decoding, Tk image uploads and inspection still consume resources.

## Remaining costs and practical next steps

1. **Camera capture/decoding:** each uncompressed BGR frame is about 45.6 MiB.
   If both cameras actually deliver 30 FPS, that is about 2.87 GB/s of decoded
   pixel data before further copies (not USB bandwidth). Reducing the requested
   FPS helps only when the driver honors it. Test supported device modes. Avoid
   adding sleeps to the acquisition loop: queued frames can become stale.
2. **Inspection disk writes:** `_simulate_detection` saves the original JPEG,
   full-resolution undistorted PNG and crop PNGs; overlays add more writes.
   These run in the inspection worker but still consume CPU and disk bandwidth.
   If dataset retention is optional, a future configurable retention policy can
   remove unnecessary writes. Current output files remain enabled.
3. **Lens correction and analysis:** full-resolution undistortion and strip
   skeleton/width analysis can cause temporary load during inspection. They
   already run on demand. Caching full-resolution remap tables would trade CPU
   savings for substantial RAM use, so it was not added for a low-power target.
4. **Crop setup capture:** the one-shot lens correction still runs synchronously
   in the setup capture callback. If that specific button stalls on target
   hardware, move it into a bounded worker with a disabled capture button while
   processing. Live preview scaling has been reduced first.
5. **SAM service latency:** inspection uploads crops to a remote workflow;
   network/service time is distinct from local UI lag. Keep its progress visible;
   lowering local CPU usage cannot eliminate remote response time.

## Verify on the target PC

Restart the app with the updated config. In Task Manager, compare CPU, memory and
disk use over a minute of idle live preview, maximized preview, frozen crop setup
and repeated inspections. Check button responsiveness and camera freshness as
well as average CPU. Verify measurements against a known reference.

If idle CPU remains high, establish which FPS the camera driver actually delivers.
If rendering is still the bottleneck, try a 150–200 ms preview interval and smaller
preview bounds. A supported lower camera resolution is a larger potential saving,
but calibration, crop alignment and measurement accuracy must be verified at that
resolution. Settings and their tradeoffs are described in `CONFIGURATION.md`.

## Glue-line Pipeline Lab

The Pipeline Lab (`tools/pipeline_lab.py`) is separate from the inspection cycle reviewed
above. Its time per frame, stage by stage, is in `GLUE_LINE_FINDINGS.md`, section *Time per
frame*: about 22 s on dark and white fabric and 45 s on pink on the development PC with
the cloud API, and about 5 s with SAM on the PC's own GPU (section *Local GPU flow*).
