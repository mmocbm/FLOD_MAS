"""Simple segment grades and calibrated width labels in image coordinates."""
import cv2
import numpy as np

# The adhesive mask is tinted before any grade is drawn. The alpha is low so the
# fabric texture stays readable through the fill, and the colour is deliberately
# none of the red/green/amber used for grades, so a tint can never be mistaken
# for a verdict.
MASK_TINT = (255, 150, 40)
MASK_OUTLINE = (255, 120, 0)
MASK_ALPHA = .30


def tint_mask(image, mask):
    """Translucent fill and outline of the detected adhesive, over a copy.

    The mask is drawn as the model produced it rather than as a fitted polygon,
    so the operator sees exactly the region that was measured. Called only from
    the automatic path: the single-line engine draws its own contours.
    """
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[..., 0]
    binary = (binary > 0).astype(np.uint8)
    # Checked before the empty case: a wrong-sized mask is a coordinate bug even
    # when it happens to be empty, and drawing nothing would hide it.
    if binary.shape != image.shape[:2]:
        raise ValueError('Adhesive mask and image must share the same frame size')
    if not binary.any():
        return image
    tinted = image.copy()
    tinted[binary > 0] = MASK_TINT
    overlay = cv2.addWeighted(tinted, MASK_ALPHA, image, 1 - MASK_ALPHA, 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, MASK_OUTLINE, 2)
    return overlay


def annotated_overlay(image, measurement, mask=None):
    overlay = tint_mask(image, mask) if mask is not None else image.copy()
    for component in measurement['components']:
        for segment in component['segments']:
            samples = [s for s in component['samples'] if s['segment'] == segment['segment']]
            if not samples:
                continue
            # None means this segment had no measurement to grade, so it is
            # marked WARNING at the segment itself rather than pass or fail.
            grade = segment['within_tolerance']
            color = (40, 40, 240) if grade is False else (60, 210, 60) if grade else (0, 190, 255)
            points = np.rint([[s['center_x'], s['center_y']] for s in samples]).astype(np.int32)
            cv2.polylines(overlay, [points], False, color, 5 if grade is False else 2)
            if grade is False:
                valid = [s for s in samples if s['valid']]
                if valid:
                    polygon = np.rint([[s['left_x'], s['left_y']] for s in valid] +
                                     [[s['right_x'], s['right_y']] for s in reversed(valid)]).astype(np.int32)
                    tint = overlay.copy()
                    cv2.fillPoly(tint, [polygon], color)
                    overlay = cv2.addWeighted(tint, .35, overlay, .65, 0)
            width = segment['average_width_mm']
            pixels = segment['average_width_px']
            label = (f'{width:.2f} mm' if width is not None else
                     f'{pixels:.1f} px' if pixels is not None else 'WARNING')
            x, y = points[len(points)//2]
            x = max(2, min(int(x)+14, overlay.shape[1]-130))
            y = max(20, min(int(y), overlay.shape[0]-8))
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    return overlay
