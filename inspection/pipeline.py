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
            measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)))
            if measurement['status'] == 'WARNING':
                warnings.append('Pixel measurements available; calibrated widths incomplete or unavailable')
            self.store.save_mask(path, result['mask'])
            self.store.save_overlay(path, result['overlay'])
            self.store.save_preview(path, 'overview', result['overlay'])
            record.update(status='complete', frame_size=[width, height], warnings=warnings,
                          settings=self.config['local_inspection'], measurement=measurement)
            return {'measurement': measurement, 'overlay': result['overlay'], 'mask': result['mask'],
                    'source_image': undistorted,
                    'warnings': warnings, 'path': str(path)}
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)

    def run_batch(self, original, undistorted, metadata, target_width, tolerance,
                  capture_path=None, on_result=lambda result: None, on_error=lambda message: None):
        from .overlay import annotated_overlay
        path = capture_path or self.store.begin(original, undistorted, metadata)
        self.store.update_metadata(path, metadata)
        record = {'status': 'error', 'capture': metadata, 'method': 'local_onnx_lines', 'lines': []}
        try:
            height, width = undistorted.shape[:2]
            warnings = []
            try:
                scale = plane_scale.load_frame_scale(1, (width, height))
            except plane_scale.PlaneScaleError as error:
                scale = None
                warnings.append(f'Millimetre calibration unavailable: {error}')
            failures = []
            def failed_line(index, error):
                message = f'Line {index}: {error}'
                failures.append(message)
                on_error(message)
            for result in self.inspector.run_lines(
                    undistorted, inspected=metadata.get('inspected_lines', ()),
                    reserved=metadata.get('queued_captures', 0), on_error=failed_line):
                measurement = measurement_record(result, scale, target_width, tolerance,
                    metadata.get('end_exclusion_percent', self.config['inspection'].get('end_exclusion_percent', 5.0)))
                # The mask is what the operator sees as the strip, so it is
                # drawn under the graded centrelines rather than instead of them.
                overlay = annotated_overlay(undistorted, measurement, result['mask'])
                line_path = path / f"line_{result['selected_line']}"
                line_path.mkdir()
                self.store.save_mask(line_path, result['mask'])
                self.store.save_overlay(line_path, overlay)
                self.store.save_preview(line_path, 'overview', overlay)
                self.store.finish(line_path, {'status': 'complete', 'measurement': measurement})
                record['lines'].append(measurement)
                line_warnings = list(warnings)
                if measurement['status'] == 'WARNING':
                    line_warnings.append('Calibrated widths incomplete or unavailable')
                on_result({'measurement': measurement, 'overlay': overlay, 'mask': result['mask'],
                           'source_image': undistorted,
                           'warnings': line_warnings, 'path': str(line_path)})
            record.update(status='complete_with_errors' if failures else 'complete',
                          errors=failures, warnings=warnings, frame_size=[width, height])
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.store.finish(path, record)
