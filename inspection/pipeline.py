"""Immutable capture -> local ONNX unfolding/refolding -> calibrated measurements."""
import plane_scale
from .line_measurement import measurement_record


class InspectionPipeline:
    def __init__(self, inspector, store, config):
        self.inspector, self.store, self.config = inspector, store, config

    def run(self, original, undistorted, metadata, target_width, tolerance, capture_path=None,
            progress=lambda text: None):
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_rightmost_line'}
        try:
            result = self.inspector.run(undistorted, progress)
            warnings = []
            height, width = undistorted.shape[:2]
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            measurement = measurement_record(result, scale, target_width, tolerance)
            if measurement['status'] == 'UNMEASURED':
                warnings.append('Pixel measurements available; calibrated widths incomplete or unavailable')
            self.store.save_mask(path, result['mask'])
            self.store.save_overlay(path, result['overlay'])
            self.store.save_preview(path, 'overview', result['overlay'])
            record.update(status='complete', frame_size=[width, height], warnings=warnings,
                          settings=self.config['local_inspection'], measurement=measurement)
            return {'measurement': measurement, 'overlay': result['overlay'], 'mask': result['mask'],
                    'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)
