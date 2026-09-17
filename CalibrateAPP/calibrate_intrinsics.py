"""
Intrinsic Camera Calibration
Detects ChArUco boards in images and computes camera matrix and distortion coefficients.
"""

import cv2
try:
    from .runtime_config import CONFIG, project_path
except ImportError:
    from runtime_config import CONFIG, project_path
import numpy as np
import json
import glob
import os
from pathlib import Path

# Board parameters (in mm)
SQUARE_LENGTH = CONFIG['board']['square_length_mm']
MARKER_LENGTH = CONFIG['board']['marker_length_mm']
SQUARES_X = CONFIG['board']['squares_x']
SQUARES_Y = CONFIG['board']['squares_y']

# Paths
CALIB_IMAGES_FOLDER = "temp_calibration_images"  # Folder with calibration images
OUTPUT_FILE = "camera_calibration.json"

def calibrate_camera():
    """Run intrinsic calibration"""
    
    print("="*70)
    print("INTRINSIC CAMERA CALIBRATION")
    print("="*70)
    
    # Initialize ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG['board']['dictionary']))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), 
        SQUARE_LENGTH / 1000,  # Convert to meters
        MARKER_LENGTH / 1000, 
        aruco_dict
    )
    
    # Initialize detector
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    print(f"\nBoard Configuration:")
    print(f"  Size: {SQUARES_X} x {SQUARES_Y}")
    print(f"  Square length: {SQUARE_LENGTH} mm")
    print(f"  Marker length: {MARKER_LENGTH} mm\n")
    
    # Get calibration images
    image_files = glob.glob(f"{CALIB_IMAGES_FOLDER}/*.jpg") + \
                  glob.glob(f"{CALIB_IMAGES_FOLDER}/*.png") + \
                  glob.glob(f"{CALIB_IMAGES_FOLDER}/*.jpeg") + \
                  glob.glob(f"{CALIB_IMAGES_FOLDER}/*.JPG") + \
                  glob.glob(f"{CALIB_IMAGES_FOLDER}/*.PNG")
    
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in '{CALIB_IMAGES_FOLDER}' folder!")
        print(f"Please add calibration images to this folder.")
        return False
    
    print(f"Found {len(image_files)} images in '{CALIB_IMAGES_FOLDER}'\n")
    
    # Detect corners in all images
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    
    print(f"{'Image':<40} {'Corners':<10} {'Status'}")
    print("-"*70)
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"{os.path.basename(img_path):<40} {'N/A':<10} ✗ Load failed")
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco board
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        
        if charuco_corners is not None and len(charuco_corners) >= 10:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            print(f"{os.path.basename(img_path):<40} {len(charuco_corners):<10} ✓")
        else:
            corners_count = len(charuco_corners) if charuco_corners is not None else 0
            print(f"{os.path.basename(img_path):<40} {corners_count:<10} ✗ Insufficient")
    
    print("-"*70)
    print(f"Valid frames: {len(all_charuco_corners)}\n")
    
    if len(all_charuco_corners) < 8:
        print(f"ERROR: Need at least 8 valid frames, but only found {len(all_charuco_corners)}")
        print("Add more calibration images with clear board views.")
        return False
    
    # Run calibration
    print("Running calibration algorithm (this may take 10-30 seconds)...\n")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, 
        all_charuco_ids, 
        board, 
        image_size, 
        None, 
        None
    )
    
    print("="*70)
    print("✓ CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nRMS Reprojection Error: {ret:.4f} pixels")
    
    if ret < 1.0:
        print("  Quality: Excellent!")
    elif ret < 2.0:
        print("  Quality: Good")
    else:
        print("  Quality: Acceptable (consider recalibrating with better images)")
    
    print(f"\nCamera Matrix:")
    print(camera_matrix)
    print(f"\nFocal Length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    print(f"Principal Point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
    
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs.flatten())
    
    # Save to file
    calib_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'rms_error': float(ret),
        'image_size': image_size
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"\n✓ Calibration saved to: {OUTPUT_FILE}")
    print("="*70)
    
    return True

if __name__ == "__main__":
    # Create calibration folder if it doesn't exist
    os.makedirs(CALIB_IMAGES_FOLDER, exist_ok=True)
    
    calibrate_camera()
