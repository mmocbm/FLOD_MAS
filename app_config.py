"""Shared configuration loaded after the startup settings window."""
import json
from pathlib import Path
import cv2
from app_storage import DATA_ROOT, data_path, initialize_storage

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
from config_validation import validate_config

validate_config(CONFIG)
CAMERA_COUNT = CONFIG.get('camera_count', len(CONFIG['cameras']))
ACTIVE_CAMERAS = CONFIG['cameras'][:CAMERA_COUNT]
cv2.setNumThreads(CONFIG.get('performance', {}).get('opencv_threads', 1))
initialize_storage()


def camera_config(index):
    return next(c for c in CONFIG["cameras"] if c["index"] == index)


def project_path(path):
    """Compatibility alias: configured data now lives in Documents."""
    return str(data_path(path))
