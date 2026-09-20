"""Generate the two uniquely numbered ChArUco boards used by two-board mode."""

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app_config import CONFIG, ROOT


def generate_two_boards(output_dir=None, pixels_per_square=200):
    board_config = CONFIG['board']
    dual_config = CONFIG['two_board']
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dual_config['dictionary'])
    )
    marker_count = (board_config['squares_x'] * board_config['squares_y']) // 2
    starts = (0, dual_config['second_board_start_id'])
    output_dir = Path(output_dir or ROOT / 'Files' / 'calibration_boards')
    output_dir.mkdir(parents=True, exist_ok=True)
    size = (
        board_config['squares_x'] * pixels_per_square,
        board_config['squares_y'] * pixels_per_square,
    )
    paths = []
    for board_number, start in enumerate(starts, 1):
        marker_ids = np.arange(start, start + marker_count, dtype=np.int32)
        board = cv2.aruco.CharucoBoard(
            (board_config['squares_x'], board_config['squares_y']),
            board_config['square_length_mm'], board_config['marker_length_mm'],
            dictionary, marker_ids,
        )
        path = output_dir / f'charuco_board_{board_number}.png'
        if not cv2.imwrite(str(path), board.generateImage(size, marginSize=0, borderBits=1)):
            raise RuntimeError(f'Could not save {path}')
        paths.append(path)
    return paths


if __name__ == '__main__':
    for generated_path in generate_two_boards():
        print(generated_path)
    width_mm = CONFIG['board']['squares_x'] * CONFIG['board']['square_length_mm']
    height_mm = CONFIG['board']['squares_y'] * CONFIG['board']['square_length_mm']
    print(f'Print each board at exactly {width_mm:.1f} x {height_mm:.1f} mm (no fit-to-page scaling).')
