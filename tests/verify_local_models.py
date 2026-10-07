"""Compare the port to the reference on the SAME input using real local models.

Run: python tests/verify_local_models.py --reference PATH --image IMAGE
Does not start a camera, call a network service or modify the reference folder.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import importlib.util
import json
from time import perf_counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspection.local.engine import LiveInspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT/'config.json').read_text())['local_inspection']
    frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Cannot read comparison image')
    cv2.setNumThreads(1)
    started = perf_counter()
    inspector = LiveInspection(ROOT/config['source_model'], ROOT/config['glue_model'],
                               config['offset_pixels'], config['source_width'])
    print(f'Model initialization: {perf_counter()-started:.2f}s', flush=True)
    result = inspector.run(frame, lambda message: print(message, flush=True))
    sys.path.insert(0, str(args.reference.resolve()))
    spec = importlib.util.spec_from_file_location('_reference_live_inspection', args.reference/'live_inspection.py')
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module.LiveInspection.__new__(reference_module.LiveInspection)
    # Share the exact model sessions. This isolates geometry/integration changes
    # from runtime/model versions and avoids loading two copies of large weights.
    reference.source, reference.glue, reference.offset = inspector.source, inspector.glue, inspector.offset
    expected = reference.run(frame)
    np.testing.assert_array_equal(result['mask'], expected['mask'])
    np.testing.assert_array_equal(result['overlay'], expected['overlay'])
    for key in ('line_count', 'selected_line', 'component_count', 'length_px',
                'longest_length_px', 'mean_width_px', 'segments'):
        assert result[key] == expected[key], (key, result[key], expected[key])
    preview = result['overlay']
    factor = min(1., 1600/max(preview.shape[:2]))
    preview = cv2.resize(preview, (round(preview.shape[1]*factor), round(preview.shape[0]*factor)))
    cv2.imwrite(str(ROOT/'tests/local_inspection_preview.jpg'), preview)
    print(json.dumps({k: result[k] for k in ('line_count','selected_line','component_count',
                                            'length_px','mean_width_px','seconds')}, indent=2))
    print('Exact reference parity: full-size mask, overlay, line selection and pixel measurements.', flush=True)


if __name__ == '__main__':
    main()
