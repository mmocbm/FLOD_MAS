import cv2
import numpy as np

class ColorMaskGenerator:
    """Class to manage region-based color mask generation."""
    
    def __init__(self, hue_range=10, sat_range=80, val_range=80):
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range

    def get_color_from_point(self, hsv_frame, point):
        """Returns the H, S, V values of a specific point."""
        x, y = point
        return hsv_frame[y, x]

    def create_mask(self, frame, region, pick_point):
        """
        Creates a color mask based on a picked point, restricted to a region.
        
        Args:
            frame: The image frame (BGR format).
            region: Tuple (x, y, w, h) of the selected region, or None for the whole image.
            pick_point: Tuple (x, y) coordinates of the color pick point.
            
        Returns:
            The generated mask (numpy array) containing only regions matching the color
            within the specified region boundaries (if provided).
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, s, v = self.get_color_from_point(hsv_frame, pick_point)
        h, s, v = int(h), int(s), int(v)
        
        mask = self._generate_color_mask(hsv_frame, h, s, v)
        
        # Apply region restriction if a region was provided
        if region is not None and len(region) == 4:
            rx, ry, rw, rh = region
            if rw > 0 and rh > 0:
                region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                region_mask[ry:ry+rh, rx:rx+rw] = 255
                mask = cv2.bitwise_and(mask, region_mask)
                
        return mask

    def _generate_color_mask(self, hsv_frame, h, s, v):
        """Internal method to apply hue, saturation, and value ranges for mask generation."""
        lower1 = np.array([0, 0, 0])
        upper1 = np.array([0, 0, 0])
        lower2 = None
        upper2 = None

        # Hue wrap-around for OpenCV HSV where Hue is 0-179
        if h - self.hue_range < 0:
            lower1 = np.array([0, max(s - self.sat_range, 0), max(v - self.val_range, 0)])
            upper1 = np.array([h + self.hue_range, min(s + self.sat_range, 255), min(v + self.val_range, 255)])

            lower2 = np.array([179 + (h - self.hue_range), max(s - self.sat_range, 0), max(v - self.val_range, 0)])
            upper2 = np.array([179, min(s + self.sat_range, 255), min(v + self.val_range, 255)])

        elif h + self.hue_range > 179:
            lower1 = np.array([h - self.hue_range, max(s - self.sat_range, 0), max(v - self.val_range, 0)])
            upper1 = np.array([179, min(s + self.sat_range, 255), min(v + self.val_range, 255)])

            lower2 = np.array([0, max(s - self.sat_range, 0), max(v - self.val_range, 0)])
            upper2 = np.array([(h + self.hue_range) - 179, min(s + self.sat_range, 255), min(v + self.val_range, 255)])

        else:
            lower1 = np.array([h - self.hue_range, max(s - self.sat_range, 0), max(v - self.val_range, 0)])
            upper1 = np.array([h + self.hue_range, min(s + self.sat_range, 255), min(v + self.val_range, 255)])

        mask1 = cv2.inRange(hsv_frame, lower1, upper1)

        if lower2 is not None:
            mask2 = cv2.inRange(hsv_frame, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1

        # Clean mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        return mask
