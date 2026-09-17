"""
Extrinsic Camera Calibration
Computes the pose (rotation and translation) of the ChArUco board reference plane.
"""

import cv2
try:
    from .runtime_config import CONFIG, project_path
except ImportError:
    from runtime_config import CONFIG, project_path
import numpy as np
import json
import os

# Board parameters (must match intrinsics calibration)
SQUARE_LENGTH = CONFIG['board']['square_length_mm']
MARKER_LENGTH = CONFIG['board']['marker_length_mm']
SQUARES_X = CONFIG['board']['squares_x']
SQUARES_Y = CONFIG['board']['squares_y']

# Paths
CALIB_FILE = "camera_calibration.json"
REF_IMAGE_PATH = "calibration_images/20260206_005308_100473.jpg"  # Image with board on measurement plane
OUTPUT_FILE = "camera_extrinsics.json"

def calibrate_extrinsics():
    """Compute board pose from reference image"""
    
    print("="*70)
    print("EXTRINSIC CALIBRATION (Board Pose)")
    print("="*70)
    
    # Load intrinsic calibration
    if not os.path.exists(CALIB_FILE):
        print(f"\nERROR: Intrinsic calibration file not found: {CALIB_FILE}")
        print("Run 'calibrate_intrinsics.py' first!")
        return False
    
    with open(CALIB_FILE, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)
    
    print(f"\n✓ Loaded intrinsic calibration from: {CALIB_FILE}")
    
    # Load reference image
    if not os.path.exists(REF_IMAGE_PATH):
        print(f"\nERROR: Reference image not found: {REF_IMAGE_PATH}")
        print("Place the reference image (board on measurement plane) and update REF_IMAGE_PATH")
        return False
    
    ref_image = cv2.imread(REF_IMAGE_PATH)
    print(f"✓ Loaded reference image: {REF_IMAGE_PATH}")
    print(f"  Image size: {ref_image.shape[1]} x {ref_image.shape[0]}\n")
    
    # Initialize detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG['board']['dictionary']))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), 
        SQUARE_LENGTH / 1000, 
        MARKER_LENGTH / 1000, 
        aruco_dict
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Detect board
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray_ref)
    
    if charuco_corners is None or len(charuco_corners) < 10:
        print(f"ERROR: Board not detected properly!")
        print(f"  Corners detected: {len(charuco_corners) if charuco_corners is not None else 0}")
        print("  Need at least 10 corners visible")
        print("\nTips:")
        print("  - Ensure board is clearly visible and well-lit")
        print("  - Check that board parameters match your printed board")
        return False
    
    print(f"✓ Detected {len(charuco_corners)} ChArUco corners")
    
    # Compute board pose
    obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
    img_pts = charuco_corners.reshape(-1, 2)
    
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, 
        img_pts, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("ERROR: solvePnP failed to compute board pose")
        return False
    
    print("\n" + "="*70)
    print("✓ EXTRINSIC CALIBRATION COMPLETE")
    print("="*70)
    
    print(f"\nRotation Vector (rvec):")
    print(rvec.T)
    
    print(f"\nTranslation Vector (tvec):")
    print(tvec.T)
    print(f"  Distance from camera: {np.linalg.norm(tvec):.3f} meters")
    
    # Save extrinsics
    extrinsics_data = {
        'rvec': rvec.reshape(-1).tolist(),
        'tvec': tvec.reshape(-1).tolist()
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(extrinsics_data, f, indent=2)
    
    print(f"\n✓ Extrinsics saved to: {OUTPUT_FILE}")
    
    # Visualize detection (save annotated image)
    vis_img = ref_image.copy()
    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
    
    # Draw coordinate axes
    axis_length = 0.05  # 50mm
    axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    origin = tuple(imgpts[0].ravel())
    vis_img = cv2.line(vis_img, origin, tuple(imgpts[1].ravel()), (0,0,255), 5)  # X red
    vis_img = cv2.line(vis_img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)  # Y green
    vis_img = cv2.line(vis_img, origin, tuple(imgpts[3].ravel()), (255,0,0), 5)  # Z blue
    
    output_vis = "reference_detected.jpg"
    cv2.imwrite(output_vis, vis_img)
    print(f"✓ Visualization saved to: {output_vis}")
    
    print("\n⚠ IMPORTANT: Keep camera position FIXED from now on!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    calibrate_extrinsics()
