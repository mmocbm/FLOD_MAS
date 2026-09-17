# Camera calibration pipeline

## What is mathematically valid

The application uses a two-stage calibration:

1. **Intrinsics:** detect ChArUco corners in varied views, match each detected ID
   to its board coordinate, and solve for the camera matrix and distortion.
2. **Measurement-plane pose:** place the same board flat on the final measurement
   surface, match its points, estimate `rvec`/`tvec`, and use that pose to intersect
   image rays with the board's `Z=0` plane.

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
- Twenty intrinsic views are requested; at least ten valid views must survive
  re-detection.
- Views require at least ten ChArUco corners.
- All intrinsic images must have one consistent resolution.
- Camera matrices are checked for finite, positive focal lengths.
- Overall and per-view intrinsic reprojection errors are saved.
- Extrinsics can only be captured after intrinsics complete.
- The UI explicitly requires the board to be flat on the final measurement surface.
- The extrinsic reference requires at least twenty corners.
- The extrinsic pose is refined and rejected above 1.5 px RMS reprojection error.
- The exact reference and an axes-annotated image are saved for audit.
- Capture resolution is specified per camera in root `config.json`, shared by setup
  and runtime. Unsupported sizes trigger a search for the best verified camera mode
  and an OK warning before use. Original frames remain
  full resolution; preview copies can be smaller. Intrinsics are scaled to the
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
