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
