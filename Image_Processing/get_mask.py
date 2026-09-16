import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from scipy.ndimage import convolve


class SkeletonSegmentationPredictor:
    def __init__(
        self,
        min_component_area=100,
        merge_distance=10,
        min_object_area=1500,
        min_skeleton_area=3000,
        max_allowed_branches=5,
    ):
        self.min_component_area = min_component_area
        self.merge_distance = merge_distance
        self.min_object_area = min_object_area
        self.min_skeleton_area = min_skeleton_area
        self.max_allowed_branches = max_allowed_branches

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.endpoint_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR format (OpenCV style)

        Returns
        -------
        mask : np.ndarray
            Binary mask (0 or 1), same spatial size as input
        """

        if image_bgr is None:
            raise ValueError("Input image is None")

        # --------------------------------------------------
        # STEP 1: Convert to grayscale
        # --------------------------------------------------
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------
        # STEP 2: Contrast enhancement + blur
        # --------------------------------------------------
        img_eq = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

        # --------------------------------------------------
        # STEP 3: Adaptive threshold
        # --------------------------------------------------
        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            101,
            2
        )

        binary = 255 - th

        # --------------------------------------------------
        # STEP 4: Connected components (remove noise)
        # --------------------------------------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        valid_labels = [
            i for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_component_area
        ]

        if len(valid_labels) == 0:
            return np.zeros(gray.shape, dtype=np.uint8)

        # --------------------------------------------------
        # STEP 5: Merge close fragments
        # --------------------------------------------------
        centers = np.array([centroids[i] for i in valid_labels])
        dist_matrix = cdist(centers, centers)

        label_map = {i: i for i in valid_labels}

        for i in range(len(valid_labels)):
            for j in range(i + 1, len(valid_labels)):
                if dist_matrix[i, j] < self.merge_distance:
                    root_i = label_map[valid_labels[i]]
                    root_j = label_map[valid_labels[j]]
                    for k in label_map:
                        if label_map[k] == root_j:
                            label_map[k] = root_i

        merged = np.zeros_like(labels, dtype=np.int32)
        for lbl in valid_labels:
            merged[labels == lbl] = label_map[lbl]

        merged_binary = (merged > 0).astype(np.uint8) * 255

        # --------------------------------------------------
        # STEP 6: Remove small objects
        # --------------------------------------------------
        num_objects, labels2, stats2, _ = cv2.connectedComponentsWithStats(
            merged_binary, connectivity=8
        )

        filtered = np.zeros_like(merged_binary)

        for i in range(1, num_objects):
            if stats2[i, cv2.CC_STAT_AREA] >= self.min_object_area:
                filtered[labels2 == i] = 255

        # --------------------------------------------------
        # STEP 7: Skeleton filtering
        # --------------------------------------------------
        skeleton = skeletonize(filtered > 0).astype(np.uint8)

        neighbor_count = convolve(
            skeleton, self.endpoint_kernel, mode="constant", cval=0
        )

        endpoints = (skeleton == 1) & (neighbor_count == 1)

        num_objects2, labels3, stats3, _ = cv2.connectedComponentsWithStats(
            filtered, connectivity=8
        )

        final_mask = np.zeros_like(filtered)

        for i in range(1, num_objects2):
            object_mask = labels3 == i
            area = stats3[i, cv2.CC_STAT_AREA]

            if area < self.min_skeleton_area:
                continue

            ep_count = np.sum(endpoints & object_mask)
            true_branches = max(0, ep_count - 2)

            if true_branches > self.max_allowed_branches:
                continue

            final_mask[object_mask] = 1

        return final_mask


if __name__ == "__main__":

    predictor = SkeletonSegmentationPredictor()

    IMAGE_PATH = r"C:\Users\Obhash\Desktop\imges\un_1.jpg"
    OUTPUT_PATH = r"C:\Users\Obhash\Desktop\imges\mask_1.png"
    
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print(f"❌ Error: Could not load image at {IMAGE_PATH}")
    else:
        # -------- Run prediction using ARRAY (not path) --------
        mask = predictor.predict(image_bgr)

        # -------- Convert mask for display --------
        mask_vis = (mask * 255).astype(np.uint8)

        # -------- Save result --------
        cv2.imwrite(OUTPUT_PATH, mask_vis)
        print("✅ Mask saved to:", OUTPUT_PATH)

        # -------- Show results --------
        cv2.imshow("Original Image", image_bgr)
        cv2.imshow("Segmentation Mask", mask_vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
