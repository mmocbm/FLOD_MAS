import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


class LinearFeatureInspector:
    """
    Inspector for straight-feature filtering,
    length inspection, and width inspection using OpenCV overlays.
    """

    def __init__(
        self,
        expected_lengths,
        length_tolerance,
        expected_width,
        width_tolerance,
        num_segments=5,
        straight_threshold=1.05,
        max_assign_distance=20,
    ):
        self.expected_lengths = np.asarray(expected_lengths, dtype=np.float32)
        self.length_tolerance = length_tolerance

        self.expected_width = expected_width
        self.width_tolerance = width_tolerance
        self.num_segments = num_segments
        self.max_assign_distance = max_assign_distance

        self.straight_threshold = straight_threshold

        # Colors
        self.normal_color = np.array([0, 180, 0])
        self.defect_color = np.array([255, 0, 0])
        self.boundary_color = (0, 255, 0)

        self.alpha_ok = 0.15
        self.alpha_defect = 0.55

    # --------------------------------------------------
    def _nearest_expected_length(self, measured):
        idx = np.argmin(np.abs(self.expected_lengths - measured))
        return self.expected_lengths[idx]

    # --------------------------------------------------
    # STEP 1: Straight filtering
    # --------------------------------------------------
    def filter_straight_objects(self, mask):
        binary = mask > 0
        skeleton = skeletonize(binary)

        num_objects, labels, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        straight_only = np.zeros_like(mask)
        skel_coords_dict = {}

        for i in range(1, num_objects):
            obj_mask = labels == i
            skel_obj = skeleton & obj_mask
            coords = np.column_stack(np.where(skel_obj))
            if len(coords) >= 2:
                skel_coords_dict[i] = coords

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

            if (max_dist / euclid) <= self.straight_threshold:
                straight_only[labels == i] = 255

        return straight_only, skel_coords_dict, labels

    # --------------------------------------------------
    # STEP 2: Length inspection
    # --------------------------------------------------
    def inspect_lengths(self, overlay, straight_only):
        h, w = straight_only.shape
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            straight_only, connectivity=8
        )

        for i in range(1, num):
            mask = labels == i
            coords = np.column_stack(np.where(mask))
            if len(coords) < 10:
                continue

            points = np.flip(coords, axis=1).astype(np.float32)
            pca = PCA(n_components=2).fit(points)

            direction = pca.components_[0]
            direction /= np.linalg.norm(direction)
            centroid = pca.mean_

            proj = np.dot(points - centroid, direction)
            p1 = centroid + proj.min() * direction
            p2 = centroid + proj.max() * direction

            pt1 = tuple(np.round(p1).astype(int))
            pt2 = tuple(np.round(p2).astype(int))

            if (
                min(pt1[1], pt2[1]) <= 0
                or max(pt1[1], pt2[1]) >= h - 1
                or min(pt1[0], pt2[0]) <= 0
                or max(pt1[0], pt2[0]) >= w - 1
            ):
                cv2.line(overlay, pt1, pt2, (0, 165, 255), 2)
                continue

            length = np.linalg.norm(p2 - p1)
            expected = self._nearest_expected_length(length)
            defect = abs(length - expected) > self.length_tolerance

            color = (255, 255, 255) if defect else (0, 255, 0)
            label = (
                f"{length:.1f}px"
                if defect
                else f"{length:.1f}px"
            )

            cv2.line(overlay, pt1, pt2, color, 2)
            cv2.circle(overlay, pt1, 4, (255, 0, 0), -1)
            cv2.circle(overlay, pt2, 4, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                label,
                tuple(centroid.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            if defect:
                x, y, bw, bh = stats[i, :4]
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)

    # --------------------------------------------------
    # STEP 3: Width inspection
    # --------------------------------------------------
    def inspect_widths(self, overlay, mask, skel_coords_dict, labels):
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        for i, skel_coords in skel_coords_dict.items():
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
                0,
                self.num_segments - 1,
            )

            pixels = np.column_stack(np.where(object_mask))
            tree = cKDTree(ordered)
            d, idx = tree.query(pixels)
            valid = d < self.max_assign_distance

            for s in range(self.num_segments):
                seg_pixels = pixels[(seg_ids[idx] == s) & valid]
                if len(seg_pixels) == 0:
                    continue

                widths = 2 * dist_transform[
                    seg_pixels[:, 0], seg_pixels[:, 1]
                ]
                mean_w = widths.mean()

                defect = abs(mean_w - self.expected_width) > self.width_tolerance
                color = self.defect_color if defect else self.normal_color
                alpha = self.alpha_defect if defect else self.alpha_ok

                overlay[seg_pixels[:, 0], seg_pixels[:, 1]] = (
                    (1 - alpha) * overlay[seg_pixels[:, 0], seg_pixels[:, 1]]
                    + alpha * color
                ).astype(np.uint8)

                cy, cx = seg_pixels.mean(axis=0).astype(int)
                cv2.putText(
                    overlay,
                    f"{mean_w:.1f}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def inspect(self, original_bgr, mask_gray):
        overlay = original_bgr.copy()
        straight_only, skel_dict, labels = self.filter_straight_objects(mask_gray)
        self.inspect_lengths(overlay, straight_only)
        self.inspect_widths(overlay, mask_gray, skel_dict, labels)
        return overlay


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    image_path = r"get_mesurement\undist_20260211_090340 - Copy.jpg"
    mask_path = r"get_mesurement\mask_result.png"

    original = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    inspector = LinearFeatureInspector(
        expected_lengths=[1200.0, 1325.0, 1450.0, 1094.0, 283.0],
        length_tolerance=2.0,
        expected_width=2.5,
        width_tolerance=1.5,
        num_segments=5,
    )

    result = inspector.inspect(original, mask)

    cv2.imshow("Length + Width Inspection", result)
    cv2.imwrite(r"get_mesurement\result.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()