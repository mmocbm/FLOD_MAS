"""In-memory source-line inspection with shared source detection per capture."""
from time import perf_counter
import numpy as np
from .geometry import (cv2, extract_lines, filter_wave_lines, offset_lines,
    region_mask, original_dimensions, unfold_maps, enhance_image, clean_mask,
    individual_crop, mapped_mask)
from .width_measurement import measure_component


class LiveInspection:
    def __init__(self, source_path, glue_path, offset=100., source_width=1152):
        from .source_model import SourceMaskOnnx
        from .glue_model import SegformerOnnx
        self.source = SourceMaskOnnx(source_path, width=source_width, flip=False)
        self.glue = SegformerOnnx(glue_path)
        self.offset = float(offset)
        if not np.isfinite(self.offset) or self.offset <= 0:
            raise ValueError('Inspection offset must be positive and finite')

    def run(self, photo, progress=lambda text: None):
        return next(self.run_lines(photo, progress, "rightmost"))

    def run_lines(self, photo, progress=lambda text: None, selection="rightmost", on_error=None):
        if selection not in ("all", "leftmost", "rightmost"):
            raise ValueError("Invalid line selection")
        started = perf_counter()
        progress('AI source mask')
        source = self.source.predict(photo)
        chains = extract_lines(source, hook_zone=.04, hook_angle=65)
        waves = np.zeros_like(source)
        for chain in chains:
            cv2.polylines(waves, [chain], False, 255, 1, cv2.LINE_8)
        filtered, _, _ = filter_wave_lines(waves, source, .008, .1, .9, make_preview=False)
        _, _, lines = offset_lines(filtered, self.offset, 'inside', source_mask=source, make_preview=False)
        if not lines:
            raise ValueError('No accepted source lines. Check placement or the source mask model.')
        lines.sort(key=lambda line: float(np.mean(np.asarray(line['original_xy'])[:, 0])))
        indices = range(len(lines)) if selection == 'all' else [0 if selection == 'leftmost' else len(lines)-1]
        for index in indices:
            progress(f'Inspecting line {index+1} of {len(lines)}')
            try:
                yield self._inspect_line(photo, source, lines[index], index+1, len(lines), started, progress)
            except Exception as error:
                if on_error is None:
                    raise
                on_error(index+1, error)

    def _inspect_line(self, photo, source, line, index, line_count, started, progress):
        selected = region_mask(source.shape, [line])
        width, height = original_dimensions(line, self.offset)
        mx, my, _ = unfold_maps(line, width, height, 'horizontal')
        progress('Enhancing and unfolding the selected inward band')
        flat = cv2.remap(enhance_image(photo, 2., 16, 16), mx, my, cv2.INTER_LINEAR)
        allowed = cv2.remap(selected, mx, my, cv2.INTER_NEAREST)
        flat[allowed == 0] = 0
        progress('Local SegFormer adhesive detection')
        predicted = self.glue.predict(flat)
        scale = width / self.glue.width
        predicted = clean_mask(predicted, max(1, round(9*scale)), max(0, round(12*scale)), .15)
        predicted[allowed == 0] = 0
        progress('Cleaning and refolding adhesive into the undistorted frame')
        _, crop_mask, bbox = individual_crop(photo, line)
        local, _, _, _ = mapped_mask(predicted, mx, my, bbox, crop_mask)
        mask = np.zeros_like(source)
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = local
        mask[selected == 0] = 0
        if not mask.any():
            raise ValueError(f'Line {index}: adhesive model returned an empty mask.')
        progress('Measuring strip in undistorted frame pixels')
        # Measure each real component. Do not bridge disconnected glue across long gaps.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        measurements = []
        overlay = photo.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        for i in range(1, count):
            cx, cy, cw, ch, area = stats[i]
            if area < 100:
                continue
            component = np.uint8(labels[cy:cy+ch, cx:cx+cw] == i)*255
            component = np.pad(component, 2)
            try:
                result = measure_component(component)
            except ValueError:
                continue
            shift = np.array([cx-2, cy-2])
            points = np.array([[s['center_x'], s['center_y']] for s in result['samples']]) + shift
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], False, (0, 255, 255), 2)
            for sample in result['samples']:
                for key, delta in (('center_x', cx-2), ('center_y', cy-2),
                                   ('left_x', cx-2), ('left_y', cy-2),
                                   ('right_x', cx-2), ('right_y', cy-2)):
                    if sample[key] is not None:
                        sample[key] += int(delta)
            measurements.append(result)
        if not measurements:
            raise ValueError('Detected adhesive is too small to measure reliably.')
        main = max(measurements, key=lambda item: item['length_px'])
        widths = [s['width_px'] for s in main['samples'] if s['valid']]
        return dict(mask=mask, overlay=overlay, line_count=line_count, selected_line=index,
                    length_px=sum(m['length_px'] for m in measurements),
                    longest_length_px=main['length_px'], component_count=len(measurements),
                    mean_width_px=float(np.mean(widths)) if widths else None,
                    segments=main['segments'], components=measurements,
                    selected_boundary=line, seconds=perf_counter()-started)
