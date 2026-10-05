"""Replay saved images through the automatic inspection stages, offline.

Runs the fabric gate, the segmentation provider and the strip measurement on
image files instead of the live camera, and writes what each stage saw and
returned. It never opens a camera, never touches the application's data folder,
and never changes config.json: settings are overridden per run with ``--set``.

Typical use, from the project folder with the project interpreter::

    .venv\\Scripts\\python.exe tools\\replay_capture.py enhancement_images\\*.jpeg

    # The same images with the low-contrast options switched on, for comparison
    .venv\\Scripts\\python.exe tools\\replay_capture.py enhancement_images\\*.jpeg ^
        --out replay_enhanced ^
        --set segmentation.enhance.enabled=true ^
        --set segmentation.roi=[0.1,0.05,0.95,0.85] ^
        --set segmentation.report_missing_strips=true ^
        --set segmentation.fragment_policy=largest ^
        --set sam_detection.refine_edges=true

Without ``--calibration`` the image is used as it is and widths stay in pixels.
With ``--calibration`` and ``--extrinsics`` it is lens-corrected first and the
widths are reported in millimetres, as in the application.

Each image gets its own folder holding ``gate.json``, ``segmentation_input.jpg``
(the exact picture sent to the model), ``overview.jpg``, ``fabric_NN.jpg`` and
``result.json``.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402 - after the path is prepared


def apply_override(config, text):
    """Apply one ``section.key=value`` override; the value is read as JSON."""
    path, separator, raw = text.partition('=')
    if not separator:
        raise SystemExit(f'--set needs section.key=value, got: {text}')
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw  # a bare word such as largest
    target = config
    keys = path.split('.')
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def classify(frame, config):
    """The fabric gate's verdict, or the reason it could not run."""
    try:
        from auto_trigger.models import FabricGate
        return FabricGate(config['auto_trigger']).classify(frame)
    except Exception as error:  # noqa: BLE001 - reported, the replay continues
        return {'error': f'{type(error).__name__}: {error}'}


def replay(image_path, config, segmenter, undistorter, scale_paths, output, skip_gate):
    from inspection.measurement import measure_instances
    from inspection.pipeline import UNGRADED, check_gradable, keep_largest_fragments
    from segmentation import validate_result

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f'Cannot read image: {image_path}')
    output.mkdir(parents=True, exist_ok=True)
    record = {'image': str(image_path), 'frame_size': [frame.shape[1], frame.shape[0]]}

    if not skip_gate:
        record['fabric_gate'] = classify(frame, config)
        (output / 'gate.json').write_text(
            json.dumps(record['fabric_gate'], indent=2), encoding='utf-8')

    working = undistorter.undistort(frame) if undistorter is not None else frame
    settings = config['segmentation']
    try:
        result = validate_result(segmenter.segment(working), working)
    finally:
        sent = getattr(segmenter, 'last_input', None)
        if sent is not None:
            cv2.imwrite(str(output / 'segmentation_input.jpg'), sent)
    record['raw_instances'] = [
        {'confidence': float(i.confidence), 'label': i.label,
         'has_strip': i.polygon is not None,
         'panel_box': None if i.panel_box is None else [float(v) for v in i.panel_box]}
        for i in result.instances]
    if settings.get('fragment_policy', 'error') == 'largest':
        result = keep_largest_fragments(result)
    check_gradable(result, settings['max_fabrics'])

    scale = None
    if scale_paths is not None:
        import plane_scale
        scale = plane_scale.load_frame_scale(1, result.frame_size, *scale_paths)
    inspection = config['inspection']
    fabrics = measure_instances(
        working, result, scale, config['sam_detection'], inspection['strip_width_mm'],
        inspection['strip_width_tolerance_mm'], settings.get('min_confidence', 0.0))

    overview = working.copy()
    for fabric in fabrics:
        item = fabric.record
        x1, y1, x2, y2 = item['display_box']
        cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 220, 220), 3)
        cv2.putText(overview, f"Fabric {item['fabric']}: {item['status']}",
                    (x1, max(25, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 220), 2)
        cv2.imwrite(str(output / f"fabric_{item['fabric']:02d}.jpg"), fabric.image)
    cv2.imwrite(str(output / 'overview.jpg'), overview)

    states = [fabric.record['status'] for fabric in fabrics]
    record.update(
        fabrics=[fabric.record for fabric in fabrics],
        warnings=[text for status, text in UNGRADED.items() if status in states])
    (output / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def summarise(record):
    gate = record.get('fabric_gate')
    if gate is None:
        gate_text = 'gate skipped'
    elif 'error' in gate:
        gate_text = f"gate unavailable ({gate['error']})"
    else:
        gate_text = f"gate {gate['label']} {gate['confidence']:.0%}"
    lines = [f"{Path(record['image']).name}: {gate_text}, "
             f"{len(record['fabrics'])} fabric(s)"]
    for item in record['fabrics']:
        measurement = item['measurement']
        if measurement is None:
            detail = 'no measurement'
        elif measurement['metric']:
            detail = (f"{measurement['average_width_mm']:.2f} mm "
                      f"({measurement['minimum_width_mm']:.2f}-"
                      f"{measurement['maximum_width_mm']:.2f})")
        else:
            detail = f"{measurement['average_width_px']:.1f} px"
        refinement = item.get('edge_refinement')
        if refinement and refinement['refined_fraction'] is not None:
            detail += f", edges located on {refinement['refined_fraction']:.0%}"
        lines.append(f"  fabric {item['fabric']}: {item['status']:<14} "
                     f"conf {item['confidence']:.2f}  {detail}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Replay saved images through gate, segmentation and measurement.')
    parser.add_argument('images', nargs='+', help='image files or wildcard patterns')
    parser.add_argument('--out', default='replay_output', help='folder for the results')
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE', help='override one setting for this run')
    parser.add_argument('--calibration', help='camera calibration JSON; enables lens correction')
    parser.add_argument('--extrinsics', help='measurement surface JSON; enables millimetres')
    parser.add_argument('--skip-gate', action='store_true',
                        help='do not load or run the fabric classifier')
    args = parser.parse_args(argv)

    config = copy.deepcopy(json.loads(Path(args.config).read_text(encoding='utf-8')))
    for override in args.overrides:
        apply_override(config, override)

    paths = []
    for pattern in args.images:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in (matches or [pattern]))

    undistorter, scale_paths = None, None
    if args.calibration:
        from measure.makeUndistored import ImageUndistorter
        undistorter = ImageUndistorter(args.calibration)
        if args.extrinsics:
            scale_paths = (args.calibration, args.extrinsics)
    elif args.extrinsics:
        raise SystemExit('--extrinsics needs --calibration as well')

    from segmentation import create_segmenter
    segmenter = create_segmenter(config['segmentation'])

    failures = 0
    for path in paths:
        try:
            record = replay(path, config, segmenter, undistorter, scale_paths,
                            Path(args.out) / path.stem, args.skip_gate)
            print(summarise(record))
        except Exception as error:  # noqa: BLE001 - one bad image must not end the run
            failures += 1
            print(f'{path.name}: FAILED - {type(error).__name__}: {error}')
    print(f'Results written to {Path(args.out).resolve()}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
