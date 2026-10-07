"""Simple segment grades and calibrated width labels in image coordinates."""
import cv2
import numpy as np


def annotated_overlay(image, measurement):
    overlay = image.copy()
    for component in measurement['components']:
        for segment in component['segments']:
            samples = [s for s in component['samples'] if s['segment'] == segment['segment']]
            if not samples:
                continue
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
                     f'{pixels:.1f} px' if pixels is not None else 'Unmeasured')
            x, y = points[len(points)//2]
            x = max(2, min(int(x)+14, overlay.shape[1]-130))
            y = max(20, min(int(y), overlay.shape[0]-8))
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    return overlay
