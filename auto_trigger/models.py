"""Synchronous inference adapters derived from hand_traker; called off Tk."""
from pathlib import Path
import time
import cv2
import numpy as np
from .fabric_check import _keras_models, load_labels, preprocess

MODEL_DIR = Path(__file__).resolve().parent / 'models'


class HandPresence:
    def __init__(self, settings):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.width = settings['detect_width']
        self.last_timestamp = -1
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=settings['max_hands'],
                min_hand_detection_confidence=settings['hand_confidence'],
                min_hand_presence_confidence=settings['hand_confidence'],
                min_tracking_confidence=settings['hand_confidence']))

    def detect(self, frame):
        """Return (present, hands) with hand landmarks in full-frame pixels.

        One inference call. MediaPipe reports normalised coordinates against the
        image it was handed, which is a uniform rescale of the full frame, so
        multiplying by the full width and height is exact.
        """
        import mediapipe as mp
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, round(h*self.width/w))))
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        timestamp = max(self.last_timestamp + 1, int(time.monotonic()*1000))
        self.last_timestamp = timestamp
        # Inference errors propagate; a failed hand check must not mean "absent".
        found = self.landmarker.detect_for_video(image, timestamp).hand_landmarks
        hands = tuple(tuple((float(point.x * w), float(point.y * h)) for point in hand)
                      for hand in found)
        return bool(hands), hands

    def present(self, frame):
        return self.detect(frame)[0]

    def close(self):
        self.landmarker.close()


class FabricGate:
    def __init__(self, settings):
        self.settings = settings
        self.labels = load_labels(MODEL_DIR / 'labels_2c.txt')
        self.model = _keras_models().load_model(str(MODEL_DIR / 'keras_model_2c.h5'), compile=False)
        if self.labels != ['full_fabric', 'no_fabric']:
            raise ValueError('Fabric labels must be 0 full_fabric and 1 no_fabric')

    def classify(self, frame):
        scores = np.asarray(self.model(preprocess(frame), training=False)).reshape(-1)
        if len(scores) != len(self.labels) or not np.isfinite(scores).all():
            raise ValueError('Invalid fabric classifier output')
        full, empty = map(float, scores)
        accepted = full > empty  # Strict comparison: ties do not trigger.
        label = 'full_fabric' if accepted else 'no_fabric' if empty > full else 'uncertain'
        return {'label': label, 'confidence': max(full, empty),
                'accepted': accepted, 'empty': empty > full,
                'scores': dict(zip(self.labels, map(float, scores)))}
