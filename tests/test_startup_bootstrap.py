"""Headless regression for loading saved settings before app imports."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import startup_settings


class StartupBootstrapTests(unittest.TestCase):
    def test_save_and_timeout_both_continue_to_fresh_config(self):
        tree = ast.parse((ROOT / 'main_1366.py').read_text(encoding='utf-8-sig'))
        bootstrap = compile(ast.Module(body=[tree.body[0]], type_ignores=[]),
                            'startup bootstrap', 'exec')
        config_code = compile((ROOT / 'app_config.py').read_text(encoding='utf-8-sig'),
                              'app_config.py', 'exec')
        for saved in (True, False):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config_path = directory / 'config.json'
                config_path.write_text(json.dumps({'cameras': [{'index': 0}], 'performance': {'opencv_threads': 1}}))
                validator = types.ModuleType('config_validation')
                validator.validate_config = lambda config: None
                cv = types.ModuleType('cv2')
                cv.setNumThreads = lambda count: None
                def startup():
                    if saved:
                        startup_settings.save_config({'cameras': [{'index': 0}], 'performance': {'opencv_threads': 2}}, config_path)
                    return saved
                with patch.dict(sys.modules, {'config_validation': validator, 'cv2': cv}), \
                     patch('app_storage.initialize_storage'), \
                     patch.object(startup_settings, 'run_startup', side_effect=startup) as prompt:
                    exec(bootstrap, {'__name__': '__main__'})
                    namespace = {'__file__': str(directory / 'app_config.py')}
                    exec(config_code, namespace)
                    prompt.assert_called_once()
                    self.assertEqual(namespace['CONFIG']['performance']['opencv_threads'],
                                     2 if saved else 1)

    def test_importing_dashboard_does_not_open_startup(self):
        tree = ast.parse((ROOT / 'main_1366.py').read_text(encoding='utf-8-sig'))
        bootstrap = compile(ast.Module(body=[tree.body[0]], type_ignores=[]),
                            'startup bootstrap', 'exec')
        with patch.object(startup_settings, 'run_startup') as prompt:
            exec(bootstrap, {'__name__': 'main_1366'})
            prompt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
