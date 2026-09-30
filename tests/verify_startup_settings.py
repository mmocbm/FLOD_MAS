"""Verify startup routing and settings writes without opening cameras."""
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import startup_settings as settings


def verify():
    root = tk.Tk()
    root.withdraw()
    window = settings.StartupWindow(root)
    assert window.timer is not None
    window.settings()
    assert window.timer is None
    assert len(window.fields) > 100
    paths = {path for path, _, _ in window.fields}
    assert ('cameras', 0, 'width') in paths
    assert ('sam_detection', 'send_crops', '1', 0) in paths
    width = next(variable for path, _, variable in window.fields
                 if path == ('cameras', 0, 'width'))
    width.set('invalid')
    with patch.object(settings.messagebox, 'showerror') as error:
        window.save()
        error.assert_called_once()
    assert not window.saved and root.winfo_exists()
    width.set('1920')
    with patch.object(settings, 'save_config') as save:
        window.save()
        assert save.call_args.args[0]['cameras'][0]['width'] == 1920
    assert window.saved

    root = tk.Tk()
    root.withdraw()
    window = settings.StartupWindow(root)
    window.cancel_timer()
    window.deadline = time.monotonic() - 1
    window.tick()
    assert not window.saved

    # Test atomic file replacement independently of OpenCV validation.
    validator = types.ModuleType('config_validation')
    validator.validate_config = lambda config: None
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'config_validation': validator}):
        path = Path(directory) / 'config.json'
        path.write_text('{"original": true}')
        settings.save_config({'width': 1920}, path)
        assert json.loads(path.read_text()) == {'width': 1920}
        validator.validate_config = lambda config: (_ for _ in ()).throw(ValueError('invalid'))
        try:
            settings.save_config({'width': -1}, path)
            raise AssertionError('Invalid settings accepted')
        except ValueError:
            pass
        assert json.loads(path.read_text()) == {'width': 1920}
        assert list(Path(directory).glob('*.tmp')) == []
    print('Startup countdown, form editing, error handling, and atomic save checks passed.')


if __name__ == '__main__':
    verify()
