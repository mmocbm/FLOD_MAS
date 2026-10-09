"""User distance correction, deliberately separate from calibration geometry."""
import json
import math
from pathlib import Path

from app_storage import data_path
from CalibrateAPP.calibration_store import replace_calibration_files


def settings_path():
    return Path(data_path('Files/measurement_adjustment.json'))


def validate_ratio(value):
    if isinstance(value, bool):
        raise ValueError('Ratio must be a positive finite number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Ratio must be a positive finite number')
    return value


def load():
    path = settings_path()
    if not path.exists():
        return {'developer_mode': False, 'ratio': 1.0}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {'developer_mode': bool(data.get('developer_mode', False)),
            'ratio': validate_ratio(data.get('ratio', 1.0))}


def save(*, ratio=None, developer_mode=None):
    data = load()
    if ratio is not None:
        data['ratio'] = validate_ratio(ratio)
    if developer_mode is not None:
        data['developer_mode'] = bool(developer_mode)
    replace_calibration_files({settings_path(): data})
    return data


def current_ratio():
    # Mode controls editing; a saved correction remains active in operator mode.
    return load()['ratio']


def adjust_accuracy_rows(rows):
    ratio = current_ratio()
    adjusted = []
    for row in rows:
        row = dict(row)
        raw = row['measured_mm']
        row.update(raw_measured_mm=raw, distance_ratio=ratio, measured_mm=raw * ratio)
        error = row['measured_mm'] - row['expected_mm']
        row.update(error_mm=error, absolute_error_mm=abs(error))
        if 'error_percent' in row:
            row['error_percent'] = 100 * error / row['expected_mm']
        adjusted.append(row)
    return adjusted
