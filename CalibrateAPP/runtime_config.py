"""Allow the standalone calibration tools to share the root configuration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_config import CONFIG, camera_config, project_path
