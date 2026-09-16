import cv2
import numpy as np
import json


class CylinderWidthMeasurer:
    def __init__(
        self,
        calibration_path="camera_calibration.json",
        extrinsics_path="camera_extrinsics.json",
        n_segments=10,
        cut_ratio=1.125,
        box_alpha=0.25
    ):
        """
        Initialize cylinder width measurer with camera calibration.
        
        Args:
            calibration_path: Path to camera_calibration.json
            extrinsics_path: Path to camera_extrinsics.json
            n_segments: Number of segments for width profile
            cut_ratio: Ratio to cut from ends when measuring
            box_alpha: Transparency of overlay boxes
        """
        
        # Load camera intrinsics
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)
        
        # Load camera extrinsics
        with open(extrinsics_path, 'r') as f:
            extrin_data = json.load(f)
        
        self.rvec = np.array(extrin_data['rvec'], dtype=np.float64).reshape(3, 1)
        self.tvec = np.array(extrin_data['tvec'], dtype=np.float64).reshape(3, 1)
        
        # Compute rotation matrix and plane equation
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.plane_normal = self.R[:, 2]
        self.d = -self.plane_normal.dot(self.tvec.flatten())
        
        self.n_segments = n_segments
        self.cut_ratio = cut_ratio
        self.box_alpha = box_alpha

    # ------------------------------------------------------------
    # Convert pixel point → real world (mm)
    # ------------------------------------------------------------
    def pixel_to_world(self, pt):
        """
        Project pixel point to 3D measurement plane.
        
        Args:
            pt: (x, y) pixel coordinates
            
        Returns:
            (x_mm, y_mm) in millimeters on the measurement plane
        """
        # No distortion (assuming image is already undistorted)
        no_distortion = np.zeros((5, 1), dtype=np.float64)
        
        # Undistort to normalized coordinates
        pts = np.array(pt, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.camera_matrix, no_distortion, P=None)
        x, y = und[0, 0, 0], und[0, 0, 1]
        
        # Ray from camera origin
        ray_direction = np.array([x, y, 1.0])
        
        # Intersect ray with plane
        denominator = self.plane_normal.dot(ray_direction)
        
        if abs(denominator) < 1e-9:
            raise ValueError("Ray is parallel to measurement plane")
        
        # Distance along ray to plane intersection
        s = -self.d / denominator
        
        # 3D point in camera coordinates
        point_camera = s * ray_direction
        
        # Transform to board coordinates (x, y on plane)
        obj_xy = self.R[:, :2].T.dot(point_camera - self.tvec.flatten())
        
        # Convert to millimeters
        x_mm = obj_xy[0] * 1000
        y_mm = obj_xy[1] * 1000
        
        return np.array([x_mm, y_mm])

    # ------------------------------------------------------------
    # Distance in real world (mm)
    # ------------------------------------------------------------
    def real_distance(self, p1, p2):
        """
        Calculate real-world distance between two pixel points.
        
        Args:
            p1: (x, y) first pixel point
            p2: (x, y) second pixel point
            
        Returns:
            Distance in millimeters
        """
        w1 = self.pixel_to_world(p1)
        w2 = self.pixel_to_world(p2)
        
        return np.linalg.norm(w1 - w2)

    # ------------------------------------------------------------
    # FAST regression line
    # ------------------------------------------------------------
    def fast_regression_lines_from_mask(self, mask):
        """
        Find regression lines from mask contours.
        
        Args:
            mask: Binary mask image
            
        Returns:
            List of (p1, p2, contour) tuples
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        lines = []

        for cnt in contours:
            if cnt.shape[0] < 30:
                continue

            pts = cnt.reshape(-1, 2)

            vx, vy, x0, y0 = cv2.fitLine(
                pts, cv2.DIST_L2, 0, 0.01, 0.01
            )

            vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()

            proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
            t_min, t_max = proj.min(), proj.max()

            p1 = (int(x0 + t_min * vx), int(y0 + t_min * vy))
            p2 = (int(x0 + t_max * vx), int(y0 + t_max * vy))

            lines.append((p1, p2, cnt))

        return lines

    # ------------------------------------------------------------
    # WIDTH profile (pixel + mm)
    # ------------------------------------------------------------
    def measure_width_profile(self, mask, p1, p2):
        """
        Measure width profile along a line.
        
        Args:
            mask: Binary mask
            p1: Start point
            p2: End point
            
        Returns:
            Tuple of measurement data
        """
        p1f = np.array(p1, dtype=np.float32)
        p2f = np.array(p2, dtype=np.float32)

        d = p2f - p1f
        full_length_px = np.linalg.norm(d)

        dir = d / full_length_px
        normal = np.array([-dir[1], dir[0]])

        center_start = p1f + self.cut_ratio * full_length_px * dir
        center_end = p2f - self.cut_ratio * full_length_px * dir

        widths_px = []
        widths_mm = []

        h, w = mask.shape

        for i in range(self.n_segments):
            t1 = i / self.n_segments
            t2 = (i + 1) / self.n_segments

            seg_start = center_start + t1 * (center_end - center_start)
            seg_end = center_start + t2 * (center_end - center_start)

            samples = 10
            seg_widths = []

            for s in range(samples):
                tt = s / (samples - 1)
                center = seg_start + tt * (seg_end - seg_start)

                cx, cy = int(center[0]), int(center[1])

                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue

                d_pos = 0
                for k in range(1, 300):
                    px = int(cx + k * normal[0])
                    py = int(cy + k * normal[1])

                    if px < 0 or py < 0 or px >= w or py >= h or mask[py, px] == 0:
                        break

                    d_pos = k

                d_neg = 0
                for k in range(1, 300):
                    px = int(cx - k * normal[0])
                    py = int(cy - k * normal[1])

                    if px < 0 or py < 0 or px >= w or py >= h or mask[py, px] == 0:
                        break

                    d_neg = k

                if d_pos + d_neg > 0:
                    seg_widths.append(d_pos + d_neg)

            mean_px = float(np.mean(seg_widths)) if seg_widths else 0
            widths_px.append(mean_px)

            # Convert width to mm using calibration
            if mean_px > 0:
                pA = tuple((seg_start + normal * mean_px / 2).astype(int))
                pB = tuple((seg_start - normal * mean_px / 2).astype(int))
                
                width_mm = self.real_distance(pA, pB)
            else:
                width_mm = 0

            widths_mm.append(width_mm)

        # Calculate length in mm
        length_mm = self.real_distance(p1, p2)

        return (
            widths_px,
            widths_mm,
            full_length_px,
            length_mm,
            dir,
            normal,
            center_start,
            center_end
        )

    # ------------------------------------------------------------
    # VISUALIZATION + REPORT
    # ------------------------------------------------------------
    def visualize_and_report(
        self,
        image,
        mask,
        length=233.0,
        width=4.0,
        length_tolerance=5.0,
        width_tolerance=0.5,
        min_object_area=1500,
        enable_check=True,
        show_deviation=False  # <<< NEW PARAMETER
    ):
        """
        Visualize measurements and generate report.
        
        Args:
            image: Input image (undistorted)
            mask: Binary mask of objects
            length: Expected length in mm
            width: Expected width in mm
            length_tolerance: Tolerance for length in mm
            width_tolerance: Tolerance for width in mm
            min_object_area: Minimum object area in pixels
            enable_check: Enable pass/fail checking
            show_deviation: Show deviation from expected values (±) instead of absolute values
            
        Returns:
            Annotated output image
        """
        output = image.copy()
        overlay = image.copy()

        objects = self.fast_regression_lines_from_mask(mask)

        # Allowed ranges (only used if enable_check=True)
        min_len = length - length_tolerance
        max_len = length + length_tolerance

        min_w = width - width_tolerance
        max_w = width + width_tolerance

        obj_counter = 0

        for (p1, p2, contour) in objects:
            # ---------------- Ignore small objects ----------------
            area = cv2.contourArea(contour)

            if area < min_object_area:
                continue

            obj_counter += 1

            (
                widths_px,
                widths_mm,
                length_px,
                length_mm,
                dir,
                normal,
                cstart,
                cend
            ) = self.measure_width_profile(mask, p1, p2)

            # ---------------- Length check ----------------
            if enable_check:
                length_ok = (min_len <= length_mm <= max_len)
            else:
                length_ok = True

            # ---------------- Width check ----------------
            width_ok_flags = []

            for w in widths_mm:
                if enable_check:
                    ok = (min_w <= w <= max_w)
                else:
                    ok = True

                width_ok_flags.append(ok)

            # ---------------- Calculate deviations ----------------
            length_deviation = length_mm - length
            width_deviations = [w - width for w in widths_mm]

            # ---------------- Console report ----------------
            print(f"\nObject {obj_counter}")
            print(f"Area: {area:.0f} px")
            
            if enable_check and show_deviation:
                print(f"Length: {length_mm:.2f} mm (Deviation: {length_deviation:+.2f} mm)")
            else:
                print(f"Length: {length_mm:.2f} mm")

            for i in range(len(widths_mm)):
                if enable_check and show_deviation:
                    print(f"Seg {i+1}: {widths_mm[i]:.2f} mm (Deviation: {width_deviations[i]:+.2f} mm)")
                else:
                    print(f"Seg {i+1}: {widths_mm[i]:.2f} mm")

            # ---------------- Draw regression line ----------------
            cv2.line(output, p1, p2, (0, 0, 255), 2)

            # -------- Length text: absolute or deviation --------
            if enable_check and show_deviation:
                # Show deviation with +/- sign
                length_text = f"{length_deviation:+.1f}mm"
            else:
                # Show absolute value
                length_text = f"{length_mm:.1f}mm"

            cv2.putText(
                output,
                length_text,
                p1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            # ---------------- Draw bounding box if needed ----------------
            if enable_check and not length_ok:
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

            # ---------------- Draw segments ----------------
            for i, width_px in enumerate(widths_px):
                t1 = i / self.n_segments
                t2 = (i + 1) / self.n_segments

                seg_start = cstart + t1 * (cend - cstart)
                seg_end = cstart + t2 * (cend - cstart)

                s = tuple(seg_start.astype(int))
                e = tuple(seg_end.astype(int))

                cv2.line(output, s, e, (0, 255, 255), 2)

                half = width_px / 2

                p1a = seg_start + normal * half
                p1b = seg_start - normal * half
                p2a = seg_end + normal * half
                p2b = seg_end - normal * half

                box = np.array([p1a, p2a, p2b, p1b], np.int32)

                # -------- Color logic --------
                if enable_check and not width_ok_flags[i]:
                    # Failed segment
                    fill_color = (0, 0, 255)
                    line_color = (0, 0, 255)
                    thickness = 2
                else:
                    # Normal
                    fill_color = (255, 0, 0)
                    line_color = (0, 0, 0)
                    thickness = 1

                cv2.fillPoly(overlay, [box], fill_color)

                cv2.polylines(
                    output,
                    [box],
                    True,
                    line_color,
                    thickness
                )

                # ---------------- Width text: absolute or deviation --------
                cx = int((s[0] + e[0]) / 2)
                cy = int((s[1] + e[1]) / 2)

                if enable_check and show_deviation:
                    # Show deviation with +/- sign
                    width_text = f"{width_deviations[i]:+.1f}mm"
                else:
                    # Show absolute value
                    width_text = f"{widths_mm[i]:.1f}mm"

                cv2.putText(
                    output,
                    width_text,
                    (cx + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        output = cv2.addWeighted(
            overlay,
            self.box_alpha,
            output,
            1 - self.box_alpha,
            0
        )

        return output


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    
    # Load undistorted mask and image
    mask = cv2.imread(r"get_mesurement\undist_20260211_090340 - Copy.jpg", cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(r"get_mesurement\mask_result.png")

    # Initialize measurer with calibration files
    measurer = CylinderWidthMeasurer(
        calibration_path=r"C:\Users\Obhash\Desktop\Factory_Data_Day_4 - 10\Files\camera_calibration_0.json",
        extrinsics_path=r"C:\Users\Obhash\Desktop\Factory_Data_Day_4 - 10\Files\camera_extrinsics_0.json",
        n_segments=2,
        cut_ratio=0.02,
        box_alpha=0.25
    )

    # Example 2: Show deviations
    print("\n" + "="*60)
    print("EXAMPLE 2: Deviation Values")
    print("="*60)
    result2 = measurer.visualize_and_report(
        original,
        mask,
        length=150.0,
        width=15.0,
        length_tolerance=1.0,
        width_tolerance=0.8,
        min_object_area=1500,
        enable_check=True,
        show_deviation=True  # Show deviations (±)
    )
    cv2.imwrite(r"measure\result_deviation.png", result2)

    # Display results
    cv2.imshow("Deviation Values", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()