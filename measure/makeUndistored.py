import cv2
import numpy as np
import json
import os


class ImageUndistorter:
    """
    Handles camera lens undistortion using MeasurementApp calibration format.
    """

    def __init__(self, calib_file_path):
        self.calib_file_path = calib_file_path
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None

        self._load_calibration()

    def _load_calibration(self):
        """Load camera matrix and distortion coefficients"""

        if not os.path.exists(self.calib_file_path):
            raise FileNotFoundError(f"Calibration file not found: {self.calib_file_path}")

        with open(self.calib_file_path, "r") as f:
            calib = json.load(f)

        self.camera_matrix = np.array(
            calib["camera_matrix"], dtype=np.float64
        )

        self.dist_coeffs = np.array(
            calib["dist_coeffs"], dtype=np.float64
        )

        image_size = calib.get("image_size")
        if image_size and len(image_size) == 2:
            self.calibration_image_size = (int(image_size[0]), int(image_size[1]))


    def undistort(self, image, crop=False):
        """
        Undistort image using cv2.undistort (same as MeasurementApp)

        Args:
            image: file path (str) OR numpy array (BGR)
            crop: (bool) Whether to crop the image to the valid region (default: False)

        Returns:
            Undistorted image (numpy array)
        """

        # Load image if path is given
        if isinstance(image, str):

            img = cv2.imread(image)

            if img is None:
                raise ValueError(f"Cannot read image: {image}")

        else:
            img = image

        height, width = img.shape[:2]
        camera_matrix = self.camera_matrix.copy()
        if self.calibration_image_size:
            calibrated_width, calibrated_height = self.calibration_image_size
            scale_x = width / calibrated_width
            scale_y = height / calibrated_height
            camera_matrix[0, 0] *= scale_x
            camera_matrix[0, 2] *= scale_x
            camera_matrix[1, 1] *= scale_y
            camera_matrix[1, 2] *= scale_y

        # Keep the same scaled intrinsic matrix in the output image.
        undistorted = cv2.undistort(
            img,
            camera_matrix,
            self.dist_coeffs,
            None,
            camera_matrix,
        )

        return undistorted


# --------------------------------------------------
# Main Example
# --------------------------------------------------
if __name__ == "__main__":

    # Paths (same style as MeasurementApp)
    CALIB_FILE = r"C:\Users\Obhash\Desktop\Factory_Day_14\2\Files\camera_calibration_1.json"
    INPUT_IMAGE = r"C:\Users\Obhash\Desktop\Factory_Day_14\2\captures\cam1\20260214_205426_238314.jpg"
    OUTPUT_IMAGE = r"C:\Users\Obhash\Desktop\imges\un_1.jpg"

    try:

        # Initialize
        undistorter = ImageUndistorter(CALIB_FILE)

        # Load image
        image_bgr = cv2.imread(INPUT_IMAGE)

        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {INPUT_IMAGE}")

        # Undistort
        result = undistorter.undistort(image_bgr)

        # Save result
        cv2.imwrite(OUTPUT_IMAGE, result)
        print("Saved:", OUTPUT_IMAGE)

        # Show result
        cv2.imshow("Undistorted Image", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
