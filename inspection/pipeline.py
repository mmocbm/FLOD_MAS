"""One immutable capture -> segmentation -> existing measurements -> saved set."""
import cv2
import plane_scale
from segmentation import validate_result
from .measurement import measure_instances


class InspectionPipeline:
    def __init__(self, segmenter, store, config):
        self.segmenter, self.store, self.config = segmenter, store, config

    def run(self, original, undistorted, metadata, target_width, tolerance, capture_path=None):
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'fabrics': []}
        try:
            result = validate_result(self.segmenter.segment(undistorted), undistorted)
            if len(result.instances) > self.config['segmentation']['max_fabrics']:
                raise ValueError(f'{len(result.instances)} strips detected; expected at most '
                                 f"{self.config['segmentation']['max_fabrics']}. Check segmentation.")
            # Multiple disjoint predictions inside one panel are ambiguous, not
            # separate fabrics. Report them rather than silently grading fragments.
            boxes = [i.panel_box for i in result.instances if i.panel_box is not None]
            if len(boxes) != len(set(boxes)):
                raise ValueError('Multiple strip fragments returned for one fabric; cannot grade reliably')
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, result.frame_size)
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            fabrics = measure_instances(undistorted, result, scale,
                                         self.config['sam_detection'], target_width, tolerance)
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
            if any(f.record['status'] == 'UNMEASURED' for f in fabrics):
                warnings.append('Some strips could not be measured in millimetres')
            record.update(status='complete', frame_size=list(result.frame_size),
                          coordinate_space='undistorted input frame', warnings=warnings,
                          fabrics=[f.record for f in fabrics])
            return {'fabrics': fabrics, 'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)
