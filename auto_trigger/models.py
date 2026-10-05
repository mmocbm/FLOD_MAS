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

    def present(self, frame):
        import mediapipe as mp
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, round(h*self.width/w))))
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        timestamp = max(self.last_timestamp + 1, int(time.monotonic()*1000))
        self.last_timestamp = timestamp
        # Inference errors propagate; a failed hand check must not mean "absent".
        return bool(self.landmarker.detect_for_video(image, timestamp).hand_landmarks)

    def close(self):
        self.landmarker.close()


class FabricGate:
    def __init__(self, settings):
        self.settings = settings
        self.labels = load_labels(MODEL_DIR / 'labels.txt')
        self.model = _keras_models().load_model(str(MODEL_DIR / 'keras_model.h5'), compile=False)
        if not {'full_fabric', 'half_fabric', 'no_fabric'} <= set(self.labels):
            raise ValueError('Fabric labels must contain full_fabric, half_fabric and no_fabric')

    def classify(self, frame):
        scores = np.asarray(self.model(preprocess(frame), training=False)).reshape(-1)
        if len(scores) != len(self.labels) or not np.isfinite(scores).all():
            raise ValueError('Invalid fabric classifier output')
        index = int(np.argmax(scores))
        label, confidence = self.labels[index], float(scores[index])
        return {'label': label, 'confidence': confidence,
                'accepted': label in self.settings['accepted_labels']
                            and confidence >= self.settings['fabric_min_confidence'],
                'empty': label == 'no_fabric' and confidence >= self.settings['fabric_min_confidence'],
                'scores': dict(zip(self.labels, map(float, scores)))}
