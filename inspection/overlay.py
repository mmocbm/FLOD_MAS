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

# The ten graded segments are drawn as one continuous centreline, so the join
# between two of them is invisible until their grades differ. A bar across the
# adhesive at each join shows where one segment ends and the next begins while
# leaving the line unbroken. It is drawn white over a dark outline, like the
# width labels, so the separation reads against either fabric tone without
# introducing a colour that could be mistaken for a grade.
SEPARATOR_COLOR = (255, 255, 255)
SEPARATOR_OUTLINE = (0, 0, 0)


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


def segment_joins(component):
    """Cross-adhesive bars separating each segment from the one before it.

    A segment's first sample lies exactly where the previous segment ended, and
    that sample's own left/right boundary points already span the adhesive, so
    the join can be drawn straight from the measurement rather than estimated.
    The first segment has no join ahead of it and is left out. A segment whose
    samples carry no boundary geometry is skipped, never guessed at.
    """
    samples = component.get('samples', ())
    joins = []
    for segment in list(component.get('segments', ()))[1:]:
        boundary = None
        for sample in samples:
            if sample.get('segment') != segment.get('segment'):
                continue
            points = [sample.get(key) for key in ('left_x', 'left_y', 'right_x', 'right_y')]
            if None in points:
                continue
            # A sample the measurement itself accepted is the boundary; the
            # first sample with geometry is kept only as a fallback, so a lone
            # rejected sample cannot hide the join between graded segments.
            if sample.get('valid'):
                boundary = points
                break
            if boundary is None:
                boundary = points
        if boundary is not None:
            joins.append(((int(boundary[0]), int(boundary[1])),
                          (int(boundary[2]), int(boundary[3]))))
    return joins


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
        for start, end in segment_joins(component):
            cv2.line(overlay, start, end, SEPARATOR_OUTLINE, 4, cv2.LINE_AA)
            cv2.line(overlay, start, end, SEPARATOR_COLOR, 2, cv2.LINE_AA)
    return overlay
