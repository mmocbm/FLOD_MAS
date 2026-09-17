"""
ChArUco Measurement Accuracy Checker - Simple Version with Long Distances

This script validates calibration accuracy by measuring known distances on ChArUco board,
including very long distances (edge to edge, corner to corner).

Usage:
    1. Change IMAGE_PATH in the code below
    2. Run: python check_accuracy.py
"""

import cv2
try:
    from .runtime_config import CONFIG, project_path
except ImportError:
    from runtime_config import CONFIG, project_path
import numpy as np
import json


# ==================== CONFIGURATION - CHANGE THESE ====================

IMAGE_PATH = r"C:\Users\Obhash\Desktop\Factory_Day_14\2\captures\cam1\20260214_204811_426733.jpg"  # ← CHANGE THIS to your test image path


# ChArUco board parameters (must match calibration)
SQUARE_LENGTH = CONFIG['board']['square_length_mm']
MARKER_LENGTH = CONFIG['board']['marker_length_mm']
SQUARES_X = CONFIG['board']['squares_x']
SQUARES_Y = CONFIG['board']['squares_y']

# Calibration files
CALIB_FILE = project_path(CONFIG['cameras'][1]['calibration_file'])
EXTRINSICS_FILE = project_path(CONFIG['cameras'][1]['extrinsics_file'])

# ======================================================================


def image_point_to_plane(img_point, camera_matrix, rvec, tvec):
    """Project image point to 3D measurement plane"""
    R, _ = cv2.Rodrigues(rvec)
    plane_normal = R[:, 2]
    d = -plane_normal.dot(tvec.flatten())
    
    no_distortion = np.zeros((5, 1), dtype=np.float64)
    pts = np.array(img_point, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, camera_matrix, no_distortion, P=None)
    x, y = und[0, 0, 0], und[0, 0, 1]
    
    ray_cam = np.array([x, y, 1.0])
    denom = plane_normal.dot(ray_cam)
    
    if abs(denom) < 1e-9:
        raise ValueError("Ray parallel to plane")
    
    s = -d / denom
    point_cam = s * ray_cam
    obj_xy = R[:, :2].T.dot(point_cam - tvec.flatten())
    
    return np.array([obj_xy[0], obj_xy[1], 0.0])


def measure_distance(p1_2d, p2_2d, camera_matrix, rvec, tvec):
    """Measure 3D distance between two undistorted image points in mm"""
    p1_3d = image_point_to_plane(p1_2d, camera_matrix, rvec, tvec)
    p2_3d = image_point_to_plane(p2_2d, camera_matrix, rvec, tvec)
    distance_m = np.linalg.norm(p1_3d - p2_3d)
    return distance_m * 1000


def get_ground_truth(id1, id2, squares_x, square_length):
    """Calculate ground truth distance between two ChArUco corners in mm"""
    corners_per_row = squares_x - 1
    
    row1 = id1 // corners_per_row
    col1 = id1 % corners_per_row
    row2 = id2 // corners_per_row
    col2 = id2 % corners_per_row
    
    dx = abs(col2 - col1)
    dy = abs(row2 - row1)
    
    return np.sqrt(dx**2 + dy**2) * square_length


def main():
    print("="*80)
    print("ChArUco Measurement Accuracy Checker - WITH LONG DISTANCES")
    print("="*80)
    
    # Load calibration
    print("\nLoading calibration files...")
    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(calib['dist_coeffs'], dtype=np.float64)
    print(f"✓ Loaded intrinsics (RMS: {calib.get('rms_error', 'N/A')})")
    
    with open(EXTRINSICS_FILE, 'r') as f:
        extrin = json.load(f)
    rvec = np.array(extrin['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(extrin['tvec'], dtype=np.float64).reshape(3, 1)
    print(f"✓ Loaded extrinsics")
    
    # Initialize detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG['board']['dictionary']))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH / 1000,
        MARKER_LENGTH / 1000,
        aruco_dict
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Load and undistort image
    print(f"\nLoading image: {IMAGE_PATH}")
    image_original = cv2.imread(IMAGE_PATH)
    if image_original is None:
        print(f"✗ ERROR: Cannot load image: {IMAGE_PATH}")
        return
    
    print(f"✓ Image loaded: {image_original.shape[1]}x{image_original.shape[0]}")
    
    print("Undistorting image...")
    image_undistorted = cv2.undistort(image_original, camera_matrix, dist_coeffs)
    print("✓ Image undistorted")
    
    # Detect corners
    print("Detecting ChArUco corners...")
    gray = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    
    if charuco_corners is None or len(charuco_corners) == 0:
        print("✗ ERROR: No ChArUco corners detected!")
        return
    
    print(f"✓ Detected {len(charuco_corners)} corners")
    
    # Create measurement pairs
    ids_flat = charuco_ids.flatten()
    corners_per_row = SQUARES_X - 1
    corners_per_col = SQUARES_Y - 1
    
    pairs = []
    
    # ========== SHORT DISTANCES ==========
    
    # Horizontal pairs (1 and 2 squares)
    for id1 in ids_flat:
        col = id1 % corners_per_row
        if col < corners_per_row - 1 and (id1 + 1) in ids_flat:
            pairs.append((id1, id1 + 1, "Horizontal 1sq"))
        if col < corners_per_row - 2 and (id1 + 2) in ids_flat:
            pairs.append((id1, id1 + 2, "Horizontal 2sq"))
    
    # Vertical pairs (1 and 2 squares)
    for id1 in ids_flat:
        row = id1 // corners_per_row
        if row < corners_per_col - 1 and (id1 + corners_per_row) in ids_flat:
            pairs.append((id1, id1 + corners_per_row, "Vertical 1sq"))
        if row < corners_per_col - 2 and (id1 + 2*corners_per_row) in ids_flat:
            pairs.append((id1, id1 + 2*corners_per_row, "Vertical 2sq"))
    
    # Diagonal pairs (1 square)
    for id1 in ids_flat:
        row = id1 // corners_per_row
        col = id1 % corners_per_row
        if row < corners_per_col - 1 and col < corners_per_row - 1:
            if (id1 + corners_per_row + 1) in ids_flat:
                pairs.append((id1, id1 + corners_per_row + 1, "Diagonal 1sq ↘"))
        if row < corners_per_col - 1 and col > 0:
            if (id1 + corners_per_row - 1) in ids_flat:
                pairs.append((id1, id1 + corners_per_row - 1, "Diagonal 1sq ↙"))
    
    # ========== MEDIUM DISTANCES ==========
    
    # Horizontal - 3, 4, 5 squares
    for id1 in ids_flat:
        col = id1 % corners_per_row
        for offset in [3, 4, 5]:
            if col < corners_per_row - offset and (id1 + offset) in ids_flat:
                pairs.append((id1, id1 + offset, f"Horizontal {offset}sq"))
    
    # Vertical - 3, 4, 5 squares
    for id1 in ids_flat:
        row = id1 // corners_per_row
        for offset in [3, 4, 5]:
            if row < corners_per_col - offset and (id1 + offset*corners_per_row) in ids_flat:
                pairs.append((id1, id1 + offset*corners_per_row, f"Vertical {offset}sq"))
    
    # ========== LONG DISTANCES ==========
    
    # Full horizontal spans (entire row width)
    for row in range(corners_per_col):
        id_left = row * corners_per_row  # Leftmost corner in row
        id_right = row * corners_per_row + (corners_per_row - 1)  # Rightmost corner in row
        if id_left in ids_flat and id_right in ids_flat:
            pairs.append((id_left, id_right, "FULL WIDTH"))
    
    # Full vertical spans (entire column height)
    for col in range(corners_per_row):
        id_top = col  # Top corner in column
        id_bottom = (corners_per_col - 1) * corners_per_row + col  # Bottom corner in column
        if id_top in ids_flat and id_bottom in ids_flat:
            pairs.append((id_top, id_bottom, "FULL HEIGHT"))
    
    # ========== MAXIMUM DISTANCES - DIAGONALS ==========
    
    # Top-left to bottom-right (main diagonal)
    id_tl = 0
    id_br = (corners_per_col - 1) * corners_per_row + (corners_per_row - 1)
    if id_tl in ids_flat and id_br in ids_flat:
        pairs.append((id_tl, id_br, "DIAGONAL TL→BR (MAX)"))
    
    # Top-right to bottom-left (anti-diagonal)
    id_tr = corners_per_row - 1
    id_bl = (corners_per_col - 1) * corners_per_row
    if id_tr in ids_flat and id_bl in ids_flat:
        pairs.append((id_tr, id_bl, "DIAGONAL TR→BL (MAX)"))
    
    # Top-left to bottom-center
    id_tc = 0
    id_bc = (corners_per_col - 1) * corners_per_row + corners_per_row // 2
    if id_tc in ids_flat and id_bc in ids_flat:
        pairs.append((id_tc, id_bc, "TL→BC (LONG)"))
    
    # Top-right to bottom-center
    id_tr = corners_per_row - 1
    id_bc = (corners_per_col - 1) * corners_per_row + corners_per_row // 2
    if id_tr in ids_flat and id_bc in ids_flat:
        pairs.append((id_tr, id_bc, "TR→BC (LONG)"))
    
    # Center to all corners
    id_center = (corners_per_col // 2) * corners_per_row + (corners_per_row // 2)
    if id_center in ids_flat:
        corners_ids = [0, corners_per_row - 1, 
                       (corners_per_col - 1) * corners_per_row,
                       (corners_per_col - 1) * corners_per_row + (corners_per_row - 1)]
        for corner_id in corners_ids:
            if corner_id in ids_flat:
                pairs.append((id_center, corner_id, "CENTER→CORNER"))
    
    # Edge to opposite edge (maximum spans)
    # Left edge to right edge (middle rows)
    mid_row = corners_per_col // 2
    id_left_mid = mid_row * corners_per_row
    id_right_mid = mid_row * corners_per_row + (corners_per_row - 1)
    if id_left_mid in ids_flat and id_right_mid in ids_flat:
        pairs.append((id_left_mid, id_right_mid, "LEFT→RIGHT EDGE"))
    
    # Top edge to bottom edge (middle columns)
    mid_col = corners_per_row // 2
    id_top_mid = mid_col
    id_bottom_mid = (corners_per_col - 1) * corners_per_row + mid_col
    if id_top_mid in ids_flat and id_bottom_mid in ids_flat:
        pairs.append((id_top_mid, id_bottom_mid, "TOP→BOTTOM EDGE"))
    
    print(f"✓ Created {len(pairs)} measurement pairs (including LONG distances)")
    
    # Measure distances
    print("\n" + "="*80)
    print("MEASUREMENTS")
    print("="*80)
    print(f"{'ID1':>4} {'ID2':>4} {'Type':<20} {'Measured':>10} {'Truth':>10} {'Error':>10} {'Error%':>8}")
    print("-"*80)
    
    errors = []
    results_by_type = {}
    
    for id1, id2, desc in pairs:
        idx1 = np.where(charuco_ids.flatten() == id1)[0][0]
        idx2 = np.where(charuco_ids.flatten() == id2)[0][0]
        
        corner1 = charuco_corners[idx1][0]
        corner2 = charuco_corners[idx2][0]
        
        measured = measure_distance(corner1, corner2, camera_matrix, rvec, tvec)
        truth = get_ground_truth(id1, id2, SQUARES_X, SQUARE_LENGTH)
        error = measured - truth
        error_pct = (error / truth) * 100
        
        errors.append(abs(error))
        
        # Group by type
        type_key = desc.split()[0] if ' ' in desc else desc
        if type_key not in results_by_type:
            results_by_type[type_key] = []
        results_by_type[type_key].append(abs(error))
        
        print(f"{id1:4d} {id2:4d} {desc:<20} {measured:9.2f}mm {truth:9.2f}mm {error:+9.2f}mm {error_pct:+7.2f}%")
    
    # Overall statistics
    errors = np.array(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    max_err = np.max(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print("\n" + "="*80)
    print("OVERALL ACCURACY STATISTICS")
    print("="*80)
    print(f"Total Measurements: {len(pairs)}")
    print(f"Mean Error:         {mean_err:.3f} mm")
    print(f"Std Deviation:      {std_err:.3f} mm")
    print(f"Max Error:          {max_err:.3f} mm")
    print(f"RMSE:               {rmse:.3f} mm")
    print("="*80)
    
    # Statistics by distance type
    print("\n" + "="*80)
    print("ACCURACY BY DISTANCE TYPE")
    print("="*80)
    print(f"{'Type':<20} {'Count':>8} {'Mean Error':>12} {'Max Error':>12}")
    print("-"*80)
    
    for type_key in sorted(results_by_type.keys()):
        type_errors = np.array(results_by_type[type_key])
        print(f"{type_key:<20} {len(type_errors):8d} {np.mean(type_errors):11.3f}mm {np.max(type_errors):11.3f}mm")
    
    print("="*80)
    
    # Interpretation
    print("\nInterpretation:")
    if mean_err < 0.5:
        print("  ✓ EXCELLENT accuracy (< 0.5mm)")
    elif mean_err < 1.0:
        print("  ✓ GOOD accuracy (< 1.0mm)")
    elif mean_err < 2.0:
        print("  ⚠ ACCEPTABLE accuracy (< 2.0mm)")
    else:
        print("  ✗ POOR accuracy (> 2.0mm) - Consider recalibrating")
    
    print("\nLong Distance Performance:")
    long_types = ['FULL', 'DIAGONAL', 'LONG', 'CENTER→CORNER', 'EDGE']
    long_errors = []
    for type_key in results_by_type.keys():
        if any(lt in type_key for lt in long_types):
            long_errors.extend(results_by_type[type_key])
    
    if long_errors:
        long_errors = np.array(long_errors)
        print(f"  Long distance mean error: {np.mean(long_errors):.3f} mm")
        print(f"  Long distance max error:  {np.max(long_errors):.3f} mm")
        if np.mean(long_errors) < 1.0:
            print("  ✓ Long distances are accurate!")
        else:
            print("  ⚠ Long distances have higher errors (common due to accumulation)")
    
    # Visualize
    print("\nCreating visualization...")
    vis_img = image_undistorted.copy()
    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
    
    # Draw long distance lines (highlight the maximum spans)
    long_pairs = [(id1, id2, desc) for id1, id2, desc in pairs 
                  if any(keyword in desc for keyword in ['FULL', 'DIAGONAL', 'MAX', 'LONG', 'EDGE'])]
    
    for id1, id2, desc in long_pairs:
        idx1 = np.where(charuco_ids.flatten() == id1)[0][0]
        idx2 = np.where(charuco_ids.flatten() == id2)[0][0]
        p1 = tuple(charuco_corners[idx1][0].astype(int))
        p2 = tuple(charuco_corners[idx2][0].astype(int))
        
        # Color code by type
        if 'MAX' in desc or 'DIAGONAL' in desc:
            color = (255, 0, 255)  # Magenta for maximum diagonals
            thickness = 3
        elif 'FULL' in desc or 'EDGE' in desc:
            color = (0, 165, 255)  # Orange for full spans
            thickness = 3
        else:
            color = (0, 255, 255)  # Yellow for other long distances
            thickness = 2
        
        cv2.line(vis_img, p1, p2, color, thickness)
    
    # Add text
    cv2.putText(vis_img, f"Corners: {len(charuco_corners)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Measurements: {len(pairs)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Mean Error: {mean_err:.2f}mm", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_img, f"RMSE: {rmse:.2f}mm", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Legend
    cv2.putText(vis_img, "Magenta: Max Diagonal", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(vis_img, "Orange: Full Width/Height", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(vis_img, "Yellow: Other Long Distances", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Save result
    output = IMAGE_PATH.replace('.jpg', '_accuracy.jpg').replace('.png', '_accuracy.png')
    cv2.imwrite(output, vis_img)
    print(f"✓ Saved visualization to: {output}")
    
    # Display (optional)
    scale = 0.6
    h, w = vis_img.shape[:2]
    vis_small = cv2.resize(vis_img, (int(w*scale), int(h*scale)))
    cv2.imshow("Accuracy Check (Long Distances) - Press any key to close", vis_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
