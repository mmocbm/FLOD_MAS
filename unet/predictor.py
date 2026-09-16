import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags


def _fix_image_orientation(image_path):
    """
    Load image from disk with PIL, apply EXIF orientation, return BGR numpy array.
    Only used when predict() receives a file path string.
    """
    pil_image = Image.open(image_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = pil_image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation_value == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation_value == 8:
                pil_image = pil_image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass

    rgb_array = np.array(pil_image)
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    else:
        bgr_array = rgb_array

    return bgr_array


def remove_small_components(mask, min_area=500):
    """
    Remove connected components smaller than min_area from a binary mask.

    Args:
        mask: Binary mask (values 0 or 1, dtype uint8)
        min_area: Components with pixel-count below this are erased.

    Returns:
        Cleaned binary mask (same shape/dtype)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):          # label 0 = background, skip
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 1

    return cleaned_mask


class UNetPredictor:
    """
    U-Net Segmentation Predictor — 4-Crop strategy.

    predict() accepts EITHER:
        • a file-path string  →  loaded & EXIF-corrected automatically
        • a BGR numpy array   →  used directly (live camera / undistorted frames)
    """

    def __init__(self, model_path, img_size=256, threshold=0.5, min_component_area=500):
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold
        self.min_component_area = min_component_area

        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully:", self.model_path)

    # ----------------------------------------------------------
    def predict(self, image):
        """
        Run 4-crop segmentation.

        Args:
            image: str (file path) OR numpy ndarray (BGR, uint8)

        Returns:
            Binary mask (H x W, values 0/1, dtype uint8)
        """

        # ---- resolve input to a BGR uint8 array ----
        if isinstance(image, str):
            img = _fix_image_orientation(image)
            if img is None:
                raise ValueError(f"❌ Unable to read image: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()          # don't mutate the caller's array
        else:
            raise TypeError("image must be a file path (str) or a numpy array (BGR uint8)")

        original_h, original_w = img.shape[:2]

        # ---- split into 4 quadrants ----
        mid_h, mid_w = original_h // 2, original_w // 2

        crops = {
            "top_left":     img[0:mid_h,          0:mid_w],
            "top_right":    img[0:mid_h,          mid_w:original_w],
            "bottom_left":  img[mid_h:original_h, 0:mid_w],
            "bottom_right": img[mid_h:original_h, mid_w:original_w],
        }

        predicted_masks = {}

        # ---- predict each crop independently ----
        for key, crop in crops.items():
            h, w = crop.shape[:2]

            # resize → normalise → add batch dim
            crop_resized = cv2.resize(crop, (self.img_size, self.img_size))
            crop_resized = crop_resized.astype(np.float32) / 255.0
            crop_resized = np.expand_dims(crop_resized, axis=0)   # (1, 256, 256, 3)

            # model inference
            pred = self.model.predict(crop_resized, verbose=0)[0]  # (256, 256, 1)

            # threshold
            mask = (pred > self.threshold).astype(np.uint8)

            # resize back to original crop size (nearest neighbour to keep binary)
            mask_resized = cv2.resize(
                mask.squeeze(),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            predicted_masks[key] = mask_resized

        # ---- stitch quadrants back together ----
        final_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        final_mask[0:mid_h,          0:mid_w]          = predicted_masks["top_left"]
        final_mask[0:mid_h,          mid_w:original_w] = predicted_masks["top_right"]
        final_mask[mid_h:original_h, 0:mid_w]          = predicted_masks["bottom_left"]
        final_mask[mid_h:original_h, mid_w:original_w] = predicted_masks["bottom_right"]

        # ---- post-process: drop tiny noise blobs ----
        final_mask = remove_small_components(final_mask, min_area=self.min_component_area)

        return final_mask

# ------------------------------------------------------------
# Main (Example Usage: Load Image into Function)
# ------------------------------------------------------------
if __name__ == "__main__":

    # -------- Settings --------
    MODEL_PATH = r"unet\models\unet_AllAptern_CoveerdMask.h5"   # change to your model path
    IMAGE_PATH = r"unet\20260203_003735_559270.jpg"   # change to your image path
    OUTPUT_PATH = r"unet\result_mask.png"

    IMG_SIZE = 256
    THRESHOLD = 0.5
    MIN_COMPONENT_AREA = 5000

    # -------- Load image into memory (OpenCV) --------
    image_bgr = cv2.imread(IMAGE_PATH)

    if image_bgr is None:
        raise FileNotFoundError(f"❌ Cannot read image: {IMAGE_PATH}")

    print("✅ Image loaded into memory:", image_bgr.shape)


    # -------- Initialize predictor --------
    predictor = UNetPredictor(
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        threshold=THRESHOLD,
        min_component_area=MIN_COMPONENT_AREA
    )


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