"""Reference H5 loading and fabric preprocessing, copied from hand_traker.

The model adapter compares full_fabric against no_fabric. This module prepares
the two-class model input using the original preprocessing.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
PAD_COLOR = 'white'


def _keras_models():
    # Teachable Machine H5 exports need legacy Keras 2 deserialization.
    import tf_keras
    return tf_keras.models


def load_labels(path: Path) -> list[str]:
    labels = []
    with open(path, 'r', encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                labels.append(parts[1].strip() if len(parts) == 2 else line)
    return labels


def preprocess(frame_bgr) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = ImageOps.pad(Image.fromarray(rgb), IMAGE_SIZE,
                         method=Image.Resampling.LANCZOS, color=PAD_COLOR)
    array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    return (array / 127.5) - 1
