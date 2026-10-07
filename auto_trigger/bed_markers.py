"""Confirm two physical ID-0 markers; duplicate IDs are intentional."""
import cv2


class BedMarkers:
    def __init__(self, confirm_seconds=0.5):
        self.confirm_seconds = confirm_seconds
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        self.reset()

    def reset(self):
        self.since = None

    def update(self, frame, now):
        _, ids, _ = self.detector.detectMarkers(frame)
        count = 0 if ids is None else int((ids.flatten() == 0).sum())
        if count < 2:
            self.reset()
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.confirm_seconds

    def detect_for_display(self, frame, width):
        """Marker outlines for the live preview, in full-frame pixels.

        Display only. It never touches self.since, so arming of the reset above
        is decided by the full-resolution scan and nothing else. The downscale
        here exists because a second full-resolution scan on every frame, rather
        than only while results are pending, is not worth its cost for drawing.
        """
        full_height, full_width = frame.shape[:2]
        scale = 1.0
        target = frame
        if width and full_width > width:
            scale = width / full_width
            target = cv2.resize(frame, (width, max(1, round(full_height * scale))),
                                interpolation=cv2.INTER_AREA)
        corners, ids, _ = self.detector.detectMarkers(target)
        if ids is None:
            return ()
        return tuple(
            tuple((float(x) / scale, float(y) / scale) for x, y in corner.reshape(4, 2))
            for corner, marker_id in zip(corners, ids.flatten()) if marker_id == 0
        )
