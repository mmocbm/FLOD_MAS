import cv2
import numpy as np
import json
import time
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


class LinearFeatureInspectorOptimized:
    """
    OPTIMIZED Inspector for straight-feature filtering,
    length inspection, and width inspection using real-world measurements (mm).

    New parameters:
        me_length : float – additional tolerance for length (mm)
        me_width  : float – additional tolerance for width (mm)

    Defect decision uses total tolerance = nominal_tolerance + machine_error.
    Displayed length/width is adjusted to the nearest nominal bound when the
    measurement falls inside the total tolerance but outside the nominal tolerance.
    """

    def __init__(
        self,
        expected_lengths,
        length_tolerance,
        expected_width,
        width_tolerance,
        calibration_path,
        extrinsics_path,
        num_segments=5,
        straight_threshold=1.05,
        max_assign_distance=20,
        debug=True,
        profile=False,
        enable_inspection=True,
        me_length=0.0,      # new parameter
        me_width=0.0,        # new parameter
    ):
        print("\n" + "="*80)
        print("INITIALIZING LINEAR FEATURE INSPECTOR (OPTIMIZED)")
        print("="*80)

        self.debug = debug
        self.profile = profile
        self.enable_inspection = enable_inspection
        self.timings = {}

        self.expected_lengths = np.asarray(expected_lengths, dtype=np.float32)
        self.length_tolerance = length_tolerance
        self.expected_width = expected_width
        self.width_tolerance = width_tolerance
        self.num_segments = num_segments
        self.max_assign_distance = max_assign_distance
        self.straight_threshold = straight_threshold

        # New machine error parameters
        self.me_length = me_length
        self.me_width = me_width

        if self.debug:
            print(f"\n[DEBUG] Configuration:")
            print(f"  Expected lengths: {self.expected_lengths} mm")
            print(f"  Length tolerance: {self.length_tolerance} mm")
            print(f"  Machine error length: {self.me_length} mm")
            print(f"  Expected width: {self.expected_width} mm")
            print(f"  Width tolerance: {self.width_tolerance} mm")
            print(f"  Machine error width: {self.me_width} mm")
            print(f"  Number of segments: {self.num_segments}")
            print(f"  Inspection enabled: {self.enable_inspection}")

        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)

        self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)

        with open(extrinsics_path, 'r') as f:
            extrin_data = json.load(f)

        self.rvec = np.array(extrin_data['rvec'], dtype=np.float64).reshape(3, 1)
        self.tvec = np.array(extrin_data['tvec'], dtype=np.float64).reshape(3, 1)

        self.R, _ = cv2.Rodrigues(self.rvec)
        self.plane_normal = self.R[:, 2]
        self.d = -self.plane_normal.dot(self.tvec.flatten())

        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.normal_color = np.array([0, 180, 0])
        self.defect_color = np.array([255, 0, 0])
        self.alpha_ok = 0.15
        self.alpha_defect = 0.55

        print("\n[SUCCESS] Initialization complete!")
        print("="*80 + "\n")

    def _start_timer(self, name):
        if self.profile:
            if name not in self.timings:
                self.timings[name] = []
            return time.perf_counter()
        return None

    def _end_timer(self, name, start_time):
        if self.profile and start_time is not None:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.timings[name].append(elapsed)

    def print_profiling_stats(self):
        if not self.profile or not self.timings:
            return

        print("\n" + "="*80)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*80)

        total_time = 0
        for name, times in sorted(self.timings.items()):
            count = len(times)
            total = sum(times)
            avg = total / count if count > 0 else 0
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0

            print(f"\n{name}:")
            print(f"  Count: {count}")
            print(f"  Total: {total:.2f} ms")
            print(f"  Average: {avg:.2f} ms")
            print(f"  Min: {min_time:.2f} ms")
            print(f"  Max: {max_time:.2f} ms")

            total_time += total

        print(f"\nTOTAL TIME: {total_time:.2f} ms ({total_time/1000:.2f} seconds)")
        print("="*80 + "\n")

    def pixel_to_world_batch(self, points):
        """
        Convert multiple pixel points to world coordinates (VECTORIZED).

        Args:
            points: Nx2 array of (x, y) pixel coordinates

        Returns:
            Nx2 array of (x_mm, y_mm) world coordinates
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        n_points = points.shape[0]
        world_coords = np.zeros((n_points, 2), dtype=np.float64)

        no_distortion = np.zeros((5, 1), dtype=np.float64)

        for i in range(n_points):
            pts = np.array(points[i], dtype=np.float64).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, self.camera_matrix, no_distortion, P=None)
            x, y = und[0, 0, 0], und[0, 0, 1]

            ray_direction = np.array([x, y, 1.0])
            denominator = self.plane_normal.dot(ray_direction)

            if abs(denominator) < 1e-9:
                continue

            s = -self.d / denominator
            point_camera = s * ray_direction

            obj_xy = self.R[:, :2].T.dot(point_camera - self.tvec.flatten())

            world_coords[i] = obj_xy * 1000

        return world_coords

    def pixel_to_world(self, pt, verbose=False):
        result = self.pixel_to_world_batch([pt])
        return result[0]

    def real_distance(self, p1, p2, verbose=False):
        w1, w2 = self.pixel_to_world_batch([p1, p2])
        return np.linalg.norm(w1 - w2)

    def _nearest_expected_length(self, measured):
        idx = np.argmin(np.abs(self.expected_lengths - measured))
        return self.expected_lengths[idx]

    def _clip_line_to_mask_fast(self, pt1, pt2, mask):
        """Fast line clipping using Bresenham algorithm"""
        h, w = mask.shape
        x1, y1 = pt1
        x2, y2 = pt2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        points = []
        x, y = x1, y1

        while True:
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y, mask[y, x] > 0))
            else:
                points.append((x, y, False))

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        first_idx = None
        last_idx = None

        for i, (x, y, inside) in enumerate(points):
            if inside:
                if first_idx is None:
                    first_idx = i
                last_idx = i

        if first_idx is None or last_idx is None:
            return pt1, pt2

        return points[first_idx][:2], points[last_idx][:2]

    def filter_straight_objects(self, mask):
        t_start = self._start_timer("Step1_StraightFiltering")

        if self.debug:
            print("\n" + "-"*80)
            print("STEP 1: FILTERING STRAIGHT OBJECTS")
            print("-"*80)

        binary = mask > 0
        skeleton = skeletonize(binary)

        num_objects, labels, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if self.debug:
            print(f"[DEBUG] Found {num_objects - 1} objects in mask")

        straight_only = np.zeros_like(mask)
        skel_coords_dict = {}

        for i in range(1, num_objects):
            obj_mask = labels == i
            skel_obj = skeleton & obj_mask
            coords = np.column_stack(np.where(skel_obj))
            if len(coords) >= 2:
                skel_coords_dict[i] = coords

        straight_count = 0
        for i, coords in skel_coords_dict.items():
            index_map = {tuple(p): idx for idx, p in enumerate(coords)}
            edges, weights = [], []

            for idx, (y, x) in enumerate(coords):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        n = (y + dy, x + dx)
                        if n in index_map:
                            edges.append((idx, index_map[n]))
                            weights.append(np.hypot(dy, dx))

            if not edges:
                continue

            graph = csr_matrix(
                (weights, ([e[0] for e in edges], [e[1] for e in edges])),
                shape=(len(coords), len(coords)),
            )

            dist = shortest_path(graph, directed=False)
            valid = dist[np.isfinite(dist)]
            if len(valid) == 0:
                continue

            max_dist = valid.max()
            p1, p2 = np.unravel_index(np.argmax(dist), dist.shape)
            euclid = np.linalg.norm(coords[p1] - coords[p2])

            if euclid == 0:
                continue

            ratio = max_dist / euclid

            if ratio <= self.straight_threshold:
                straight_only[labels == i] = 255
                straight_count += 1

        if self.debug:
            print(f"\n[RESULT] {straight_count} out of {len(skel_coords_dict)} objects passed straightness filter")
            print("-"*80 + "\n")

        self._end_timer("Step1_StraightFiltering", t_start)
        return straight_only, skel_coords_dict, labels

    def inspect_lengths(self, overlay, straight_only):
        t_start = self._start_timer("Step2_LengthInspection")

        if self.debug:
            print("\n" + "-"*80)
            if self.enable_inspection:
                print("STEP 2: LENGTH INSPECTION (with machine error)")
            else:
                print("STEP 2: LENGTH MEASUREMENT (Inspection Disabled)")
            print("-"*80)

        h, w = straight_only.shape
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            straight_only, connectivity=8
        )

        if self.debug:
            print(f"[DEBUG] Inspecting {num - 1} straight objects for length")

        inspected_count = 0
        defect_count = 0

        for i in range(1, num):
            mask = labels == i
            coords = np.column_stack(np.where(mask))
            if len(coords) < 10:
                continue

            inspected_count += 1

            points = np.flip(coords, axis=1).astype(np.float32)
            pca = PCA(n_components=2).fit(points)

            direction = pca.components_[0]
            direction /= np.linalg.norm(direction)
            centroid = pca.mean_

            proj = np.dot(points - centroid, direction)
            p1 = centroid + proj.min() * direction
            p2 = centroid + proj.max() * direction

            pt1_original = tuple(np.round(p1).astype(int))
            pt2_original = tuple(np.round(p2).astype(int))

            t_clip = self._start_timer("  Clipping")
            pt1_clipped, pt2_clipped = self._clip_line_to_mask_fast(
                pt1_original, pt2_original, mask
            )
            self._end_timer("  Clipping", t_clip)

            if (
                min(pt1_clipped[1], pt2_clipped[1]) <= 0
                or max(pt1_clipped[1], pt2_clipped[1]) >= h - 1
                or min(pt1_clipped[0], pt2_clipped[0]) <= 0
                or max(pt1_clipped[0], pt2_clipped[0]) >= w - 1
            ):
                cv2.line(overlay, pt1_clipped, pt2_clipped, (0, 165, 255), 2)
                continue

            t_calib = self._start_timer("  Calibration")
            length_mm = self.real_distance(pt1_clipped, pt2_clipped)
            self._end_timer("  Calibration", t_calib)

            # Determine expected length and tolerances
            expected = self._nearest_expected_length(length_mm)
            deviation = abs(length_mm - expected)
            nominal_tol = self.length_tolerance
            total_tol = nominal_tol + self.me_length

            within_nominal = deviation <= nominal_tol
            within_total = deviation <= total_tol

            # Defect flag based on total tolerance
            defect = not within_total
            if defect:
                defect_count += 1

            # Decide displayed value
            if within_nominal:
                display_value = length_mm
            elif within_total:
                # Inside total tolerance but outside nominal → show bound
                if length_mm > expected:
                    display_value = expected + nominal_tol
                else:
                    display_value = expected - nominal_tol
            else:
                display_value = length_mm   # defect, show actual

            label = f"{display_value:.1f}mm"

            # Color: green if not defect, red if defect
            color = (0, 0, 255) if defect else (0, 255, 0)

            cv2.line(overlay, pt1_clipped, pt2_clipped, color, 2)
            cv2.circle(overlay, pt1_clipped, 4, (255, 0, 0), -1)
            cv2.circle(overlay, pt2_clipped, 4, (0, 0, 255), -1)
            cv2.putText(
                overlay, label, (pt1_clipped[0] + 30, pt1_clipped[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

            if defect:
                x, y, bw, bh = stats[i, :4]
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)

        if self.debug:
            print(f"\n[RESULT] Inspected {inspected_count} objects")
            if self.enable_inspection:
                print(f"[RESULT] Found {defect_count} length defects (using total tolerance)")
            else:
                print(f"[RESULT] Measurement-only mode (no defect checking)")
            print("-"*80 + "\n")

        self._end_timer("Step2_LengthInspection", t_start)

    def inspect_widths(self, overlay, mask, skel_coords_dict, labels):
        t_start = self._start_timer("Step3_WidthInspection")

        if self.debug:
            print("\n" + "-"*80)
            if self.enable_inspection:
                print("STEP 3: WIDTH INSPECTION (with machine error)")
            else:
                print("STEP 3: WIDTH MEASUREMENT (Inspection Disabled)")
            print("-"*80)

        t_dt = self._start_timer("  DistanceTransform")
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        self._end_timer("  DistanceTransform", t_dt)

        total_segments = 0
        defect_segments = 0

        for obj_idx, (i, skel_coords) in enumerate(skel_coords_dict.items(), 1):
            t_obj = self._start_timer("  ProcessObject")

            object_mask = labels == i

            index_map = {tuple(p): idx for idx, p in enumerate(skel_coords)}
            edges, weights = [], []

            for idx, (y, x) in enumerate(skel_coords):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        n = (y + dy, x + dx)
                        if n in index_map:
                            edges.append((idx, index_map[n]))
                            weights.append(np.hypot(dy, dx))

            if not edges:
                continue

            graph = csr_matrix(
                (weights, ([e[0] for e in edges], [e[1] for e in edges])),
                shape=(len(skel_coords), len(skel_coords)),
            )

            dist = shortest_path(graph, directed=False)
            max_dist = np.max(dist[np.isfinite(dist)])
            s_idx, e_idx = np.unravel_index(np.argmax(dist), dist.shape)

            d_start = dist[s_idx]
            main = np.isclose(d_start + dist[:, e_idx], max_dist, atol=1e-6)

            ordered = skel_coords[main][np.argsort(d_start[main])]
            seg_ids = np.clip(
                np.linspace(0, self.num_segments, len(ordered), endpoint=False).astype(int),
                0, self.num_segments - 1,
            )

            for s in range(self.num_segments):
                seg_skel_points = ordered[seg_ids == s]

                if len(seg_skel_points) == 0:
                    continue

                total_segments += 1

                num_samples = min(4, len(seg_skel_points))
                if num_samples < len(seg_skel_points):
                    sample_indices = np.random.choice(len(seg_skel_points), num_samples, replace=False)
                    sampled_points = seg_skel_points[sample_indices]
                else:
                    sampled_points = seg_skel_points

                seg_center_idx = len(seg_skel_points) // 2
                if len(seg_skel_points) >= 3:
                    p1_idx = max(0, seg_center_idx - 1)
                    p2_idx = min(len(seg_skel_points) - 1, seg_center_idx + 1)

                    p1 = seg_skel_points[p1_idx]
                    p2 = seg_skel_points[p2_idx]

                    tangent = np.array([p2[1] - p1[1], p2[0] - p1[0]], dtype=np.float32)
                    tangent_norm = np.linalg.norm(tangent)

                    if tangent_norm > 0:
                        tangent = tangent / tangent_norm
                        perpendicular = np.array([-tangent[1], tangent[0]])
                    else:
                        perpendicular = np.array([0, 1], dtype=np.float32)
                else:
                    perpendicular = np.array([0, 1], dtype=np.float32)

                t_width_calc = self._start_timer("  WidthCalculation")

                radii = dist_transform[sampled_points[:, 0], sampled_points[:, 1]]

                points_a = sampled_points[:, [1, 0]] + perpendicular * radii[:, np.newaxis]
                points_b = sampled_points[:, [1, 0]] - perpendicular * radii[:, np.newaxis]

                h, w = mask.shape
                points_a = np.clip(points_a, [0, 0], [w-1, h-1])
                points_b = np.clip(points_b, [0, 0], [w-1, h-1])

                all_points = np.vstack([points_a, points_b])
                world_coords = self.pixel_to_world_batch(all_points)

                world_a = world_coords[:num_samples]
                world_b = world_coords[num_samples:]

                widths_mm = np.linalg.norm(world_a - world_b, axis=1)

                self._end_timer("  WidthCalculation", t_width_calc)

                mean_width_mm = np.mean(widths_mm)
                std_width_mm = np.std(widths_mm)

                # Determine defect and display value
                deviation = abs(mean_width_mm - self.expected_width)
                nominal_tol = self.width_tolerance
                total_tol = nominal_tol + self.me_width

                within_nominal = deviation <= nominal_tol
                within_total = deviation <= total_tol

                defect = not within_total
                if defect:
                    defect_segments += 1

                # Display value
                if within_nominal:
                    display_width = mean_width_mm
                elif within_total:
                    if mean_width_mm > self.expected_width:
                        display_width = self.expected_width + nominal_tol
                    else:
                        display_width = self.expected_width - nominal_tol
                else:
                    display_width = mean_width_mm

                label = f"{display_width:.1f}mm"

                color = self.defect_color if defect else self.normal_color
                alpha = self.alpha_defect if defect else self.alpha_ok

                # Assign pixels to segment
                pixels = np.column_stack(np.where(object_mask))
                tree = cKDTree(ordered)
                d, idx = tree.query(pixels)
                valid = d < self.max_assign_distance
                seg_pixels = pixels[(seg_ids[idx] == s) & valid]

                if len(seg_pixels) > 0:
                    overlay[seg_pixels[:, 0], seg_pixels[:, 1]] = (
                        (1 - alpha) * overlay[seg_pixels[:, 0], seg_pixels[:, 1]]
                        + alpha * color
                    ).astype(np.uint8)

                    cy, cx = seg_pixels.mean(axis=0).astype(int)
                    cv2.putText(
                        overlay, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                    )

            self._end_timer("  ProcessObject", t_obj)

        if self.debug:
            print(f"\n[RESULT] Inspected {total_segments} segments across {len(skel_coords_dict)} objects")
            if self.enable_inspection:
                print(f"[RESULT] Found {defect_segments} width defects (using total tolerance)")
            else:
                print(f"[RESULT] Measurement-only mode (no defect checking)")
            print("-"*80 + "\n")

        self._end_timer("Step3_WidthInspection", t_start)

    def inspect(self, original_bgr, mask_gray):
        t_total = self._start_timer("TotalInspection")

        if self.debug:
            print("\n" + "="*80)
            print("STARTING INSPECTION PIPELINE (OPTIMIZED)")
            print("="*80)
            print(f"[DEBUG] Image shape: {original_bgr.shape}")
            print(f"[DEBUG] Mask shape: {mask_gray.shape}")

        overlay = original_bgr.copy()

        straight_only, skel_dict, labels = self.filter_straight_objects(mask_gray)
        self.inspect_lengths(overlay, straight_only)
        self.inspect_widths(overlay, mask_gray, skel_dict, labels)

        self._end_timer("TotalInspection", t_total)

        if self.debug:
            print("\n" + "="*80)
            print("INSPECTION COMPLETE")
            print("="*80 + "\n")

        return overlay


if __name__ == "__main__":

    print("\n" + "="*80)
    print("LINEAR FEATURE INSPECTOR - OPTIMIZED VERSION")
    print("="*80 + "\n")

    image_path = r"C:\Users\Obhash\Desktop\imges\un_1.jpg"
    mask_path = r"C:\Users\Obhash\Desktop\imges\mask_1.png"

    original = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original is None or mask is None:
        print("ERROR: Could not load images")
        exit(1)

    print(f"✓ Images loaded")

    # Example 1: With inspection (defect detection enabled) and machine error
    print("\n" + "="*60)
    print("MODE 1: INSPECTION ENABLED (with machine error)")
    print("="*60)

    inspector_inspect = LinearFeatureInspectorOptimized(
        expected_lengths=[150.0, 175.0, 200.0, 120.0, 283,227],  # in mm
        length_tolerance=0.1,           # nominal tolerance (mm)
        expected_width=3.7,              # mm
        width_tolerance=0.2,              # nominal tolerance (mm)
        calibration_path=r"C:\Users\Obhash\Desktop\Factory_Day_14\2\Files\camera_calibration_1.json",
        extrinsics_path=r"C:\Users\Obhash\Desktop\Factory_Day_14\2\Files\camera_extrinsics_1.json",
        num_segments=5,
        debug=True,
        profile=True,
        enable_inspection=True,           # defect detection ON
        me_length=0.1,          # additional length tolerance (mm)
        me_width=0.1,            # additional width tolerance (mm)
    )

    result_inspect = inspector_inspect.inspect(original, mask)
    inspector_inspect.print_profiling_stats()

    output_path = r"C:\Users\Obhash\Desktop\imges\result_calibrated_1.png"
    cv2.imwrite(output_path, result_inspect)
    print(f"\n✓ Result saved to: {output_path}")

    cv2.imshow("With Inspection (Defects Highlighted)", result_inspect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()