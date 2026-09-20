# Camera calibration pipeline

## What is mathematically valid

The application always calibrates the camera and supports two measurement-plane modes:

1. **Intrinsics:** detect ChArUco corners in varied views, match each detected ID
   to its board coordinate, and solve for the camera matrix and distortion.
   The operator may use the original single-board method or two uniquely numbered
   boards. In two-board mode, every board pose is passed as its own calibration view;
   two independently placed boards are never incorrectly treated as one rigid board.
2. **Saved surface mode:** place the same board flat on the final measurement
   surface, estimate its pose, and reuse that saved plane during inspection.
3. **Per-inspection marker mode:** detect configured 4×4 ArUco marker ID 0 in each
   inspection frame and calculate an image-to-plane homography in millimetres.
   If the marker is unavailable, continue with pixel-only distances and display a
   non-blocking warning; do not calculate metric PASS/FAIL results for that frame.

This follows OpenCV's current ChArUco workflow. OpenCV recommends ChArUco corners
for calibration because they combine partial-board detection with subpixel chessboard
corner accuracy. Its current examples use `CharucoDetector.detectBoard()`,
`CharucoBoard.matchImagePoints()`, and `calibrateCamera()` across multiple viewpoints.

The pose step follows OpenCV's documented convention: `solvePnP()` returns the
rotation and translation that transform board/object coordinates into camera
coordinates. The application uses the planar IPPE initializer, refines it with
Levenberg-Marquardt, and checks the resulting reprojection error before saving.

References:

- [OpenCV 5: Calibration with ChArUco](https://docs.opencv.org/5.0/tutorials/objdetect/aruco_calibration/aruco_calibration.html)
- [OpenCV: ChArUco detection and pose](https://docs.opencv.org/5.0/tutorials/objdetect/charuco_detection/charuco_detection.html)
- [OpenCV: solvePnP pose convention](https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html)
- [OpenCV: camera matrix resolution scaling](https://docs.opencv.org/4.13.0/d4/d94/tutorial_camera_calibration.html)

## Correctness controls implemented

- Exactly one camera is calibrated per session.
- In one-board mode, twenty camera views are requested. In two-board mode, twenty
  photo sets are requested and each set contributes two independent board views.
  Each accepted view is stored immediately. The full camera calculation runs once,
  after all requested photos or photo sets are accepted.
- Two-board prints use non-overlapping marker IDs. If both are found in one image,
  both views are accepted. Otherwise the first view and image remain saved while
  the UI requests the missing board in a second image. An automatic masked retry
  is performed before falling back to that two-image flow.
- Views require the configured minimum number of ChArUco corners.
- All intrinsic images must have one consistent resolution.
- The live preview draws a 3×3 coverage map, points from previous captures, saved
  capture centers, and the recommended area for the next view.
- Board detection runs only when **Capture Photo** is pressed. The full-resolution
  frame is saved as PNG and reloaded before detection. Valid images remain; rejected
  images are deleted. The frame is held while points are drawn, then live display resumes.
- Coverage guidance improves image distribution but does not replace the operator's
  need to tilt the board and vary its distance where the physical setup allows.
- Camera matrices are checked for finite, positive focal lengths.
- Overall and per-view intrinsic reprojection errors are saved.
- In saved surface mode, the surface reference can only be captured after camera
  calibration. In marker mode, Camera Setup finishes immediately after calibration.
- The UI explicitly requires the board to be flat on the final measurement surface.
- The extrinsic reference requires at least twenty corners.
- The extrinsic pose is refined and rejected above 1.5 px RMS reprojection error.
- The exact reference and an axes-annotated image are saved for audit.
- Capture resolution is specified per camera in root `config.json`, shared by setup
  and runtime. Unsupported sizes trigger a search for the best verified camera mode
  and silently use it. Original frames remain full resolution; preview copies can
  be smaller. Intrinsics are scaled to the
  processing image size. Recalibrate after changing capture mode or field of view.

## Numerical proof

Run:

```powershell
.\.venv\Scripts\python.exe .\CalibrateAPP\verify_calibration_pipeline.py
```

The script projects known board coordinates through a known synthetic camera, then
checks that the pipeline recovers the pose, preserves projection under resolution
scaling, and maps an image pixel back to its original point on the measurement plane.

This proves the coordinate and transformation implementation under controlled input.
It does **not** prove a physical installation by itself. Physical accuracy also depends
on the printed board dimensions, image focus, lighting, board flatness, camera rigidity,
and the product occupying the same plane as the extrinsic board.

## Physical acceptance test

After calibrating each camera:

1. Do not move the camera or measurement surface.
2. Place a traceable ruler or gauge block flat on the measurement plane.
3. Measure several known distances near the center and all four image corners.
4. Record absolute errors and repeatability over at least ten captures.
5. Accept the installation only if those errors meet the production tolerance.

If the camera moves, focus changes, the capture resolution/aspect ratio changes, or
the measurement-surface height changes, repeat the relevant calibration.
