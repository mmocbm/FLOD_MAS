"""One immutable capture -> segmentation -> existing measurements -> saved set."""
from dataclasses import replace
import cv2
import numpy as np
import plane_scale
from segmentation import SegmentationResult, validate_result
from .measurement import measure_instances

# Statuses that are shown but never graded; each one is also raised as a warning.
UNGRADED = {
    'UNMEASURED': 'Some strips could not be measured in millimetres',
    'NO STRIP': 'A fabric was found with no glue strip detected',
    'FRAGMENTED': 'A glue strip came back in pieces; only its largest piece is shown',
    'LOW CONFIDENCE': 'A glue strip was detected with low confidence and was not graded',
}


def keep_largest_fragments(result):
    """One strip per panel: keep the largest piece and mark it as fragmented."""
    groups = {}
    for instance in result.instances:
        if instance.panel_box is not None and instance.polygon is not None:
            groups.setdefault(instance.panel_box, []).append(instance)
    kept = {}
    for box, pieces in groups.items():
        if len(pieces) > 1:
            largest = max(pieces, key=lambda item: cv2.contourArea(
                np.asarray(item.polygon, np.float32)))
            kept[box] = replace(largest, note='fragmented')
    if not kept:
        return result
    instances = []
    for instance in result.instances:
        chosen = kept.get(instance.panel_box) if instance.polygon is not None else None
        if chosen is None:
            instances.append(instance)
        elif instance.polygon is chosen.polygon:
            instances.append(chosen)
    return SegmentationResult(tuple(instances), result.frame_size)

def check_gradable(result, max_fabrics):
    """Refuse a result that cannot be graded fabric by fabric."""
    if len(result.instances) > max_fabrics:
        raise ValueError(f'{len(result.instances)} strips detected; expected at most '
                         f"{max_fabrics}. Check segmentation.")
    # Multiple disjoint predictions inside one panel are ambiguous, not
    # separate fabrics. Report them rather than silently grading fragments.
    boxes = [i.panel_box for i in result.instances if i.panel_box is not None]
    if len(boxes) != len(set(boxes)):
        raise ValueError('Multiple strip fragments returned for one fabric; cannot grade reliably')


class InspectionPipeline:
    def __init__(self, segmenter, store, config):
        self.segmenter, self.store, self.config = segmenter, store, config

    def run(self, original, undistorted, metadata, target_width, tolerance, capture_path=None):
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'fabrics': []}
        try:
            settings = self.config['segmentation']
            try:
                result = validate_result(self.segmenter.segment(undistorted), undistorted)
            finally:
                sent = getattr(self.segmenter, 'last_input', None)
                if settings.get('save_input', False) and sent is not None:
                    self.store.save_preview(path, 'segmentation_input', sent)
            if settings.get('fragment_policy', 'error') == 'largest':
                result = keep_largest_fragments(result)
            check_gradable(result, settings['max_fabrics'])
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, result.frame_size)
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            fabrics = measure_instances(undistorted, result, scale,
                                         self.config['sam_detection'], target_width, tolerance,
                                         settings.get('min_confidence', 0.0))
            overview = undistorted.copy()
            for fabric in fabrics:
                item = fabric.record
                x1, y1, x2, y2 = item['display_box']
                cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 220, 220), 3)
                cv2.putText(overview, f"Fabric {item['fabric']}: {item['status']}",
                            (x1, max(25, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 220), 2)
                self.store.save_preview(path, f"fabric_{item['fabric']:02d}", fabric.image)
            self.store.save_preview(path, 'overview', overview)
            if not fabrics:
                warnings.append('Fabric accepted, but no glue strip was detected')
            states = {f.record['status'] for f in fabrics}
            warnings.extend(text for status, text in UNGRADED.items() if status in states)
            record.update(status='complete', frame_size=list(result.frame_size),
                          coordinate_space='undistorted input frame', warnings=warnings,
                          fabrics=[f.record for f in fabrics])
            return {'fabrics': fabrics, 'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)
