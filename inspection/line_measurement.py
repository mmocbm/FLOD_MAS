"""Calibrate the new refolded centreline and boundary intersections."""
import copy
import numpy as np
from measurement_adjustment import current_ratio


def with_distance_ratio(measurement, ratio):
    """Recalculate a displayed result from its recorded ratio, without mutation."""
    from measurement_adjustment import validate_ratio
    ratio = validate_ratio(ratio)
    factor = ratio / validate_ratio(measurement.get('distance_ratio', 1.0))
    result = copy.deepcopy(measurement)
    def multiply(item, names):
        for name in names:
            if item.get(name) is not None:
                item[name] *= factor
    multiply(result, ('length_mm', 'longest_length_mm', 'mean_width_mm'))
    grades = []
    for component in result['components']:
        multiply(component, ('length_mm',))
        for sample in component['samples']:
            multiply(sample, ('width_mm',))
        for segment in component['segments']:
            multiply(segment, ('average_width_mm', 'minimum_width_mm', 'maximum_width_mm'))
            average = segment.get('average_width_mm')
            segment['within_tolerance'] = (None if average is None else
                result['target_width_mm']-result['tolerance_mm'] <= average <=
                result['target_width_mm']+result['tolerance_mm'])
            grades.append(segment['within_tolerance'])
    main = max(result['components'], key=lambda c: c['length_px'])
    result['segments'] = main['segments']
    result['distance_ratio'] = ratio
    result['status'] = ('WARNING' if not result['metric'] or any(g is None for g in grades)
                        else 'PASS' if all(grades) else 'FAIL')
    return result


def measurement_record(result, scale, target_width, tolerance, end_exclusion_percent=0.0):
    """Keep reference pixel measurements; map points, never use a global px scale."""
    if (isinstance(end_exclusion_percent, bool) or not np.isfinite(end_exclusion_percent)
            or not 0 <= end_exclusion_percent < 50):
        raise ValueError('End exclusion must be at least 0 and less than 50 percent')
    ratio = current_ratio()
    components = copy.deepcopy(result['components'])
    for component in components:
        samples = component['samples']
        low = component['length_px'] * end_exclusion_percent / 100.0
        high = component['length_px'] - low
        component['excluded_each_end_px'] = low
        component['inspected_length_px'] = high-low
        count = len(component['segments'])
        for sample in samples:
            excluded = not low <= sample['arc_px'] <= high
            sample['excluded_end'] = excluded
            if excluded:
                sample.update(segment=0, valid=False, width_px=None)
            elif end_exclusion_percent:
                sample['segment'] = min(count, int((sample['arc_px']-low)/(high-low)*count)+1)
        for segment in component['segments']:
            segment['start_arc_px'] = low+(segment['segment']-1)*(high-low)/count
            segment['end_arc_px'] = low+segment['segment']*(high-low)/count
            pixels = [s['width_px'] for s in samples if s['segment'] == segment['segment'] and s['valid']]
            segment['sample_count'] = len(pixels)
            for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
                segment[f'{name}_width_px'] = float(operation(pixels)) if pixels else None
        if scale is not None:
            centres = scale.to_mm([[s['center_x'], s['center_y']] for s in samples])
            component['length_mm'] = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()) * ratio
        else:
            component['length_mm'] = None
        for sample in samples:
            sample['width_mm'] = None
            if scale is not None and sample['valid']:
                endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                         [sample['right_x'], sample['right_y']]])
                width = float(np.linalg.norm(endpoints[1] - endpoints[0])) * ratio
                if np.isfinite(width):
                    sample['width_mm'] = width
        for segment in component['segments']:
            widths = [s['width_mm'] for s in samples
                      if s['segment'] == segment['segment'] and s['width_mm'] is not None]
            for name, operation in [('average', np.mean), ('minimum', np.min), ('maximum', np.max)]:
                segment[f'{name}_width_mm'] = float(operation(widths)) if widths else None
            average = segment['average_width_mm']
            segment['within_tolerance'] = (target_width-tolerance <= average <= target_width+tolerance
                                           if average is not None else None)
    main = max(components, key=lambda c: c['length_px'])
    widths = [s['width_mm'] for s in main['samples'] if s['width_mm'] is not None]
    pixel_widths = [s['width_px'] for s in main['samples'] if s['valid']]
    graded = [s['within_tolerance'] for component in components for s in component['segments']]
    metric = scale is not None and bool(widths)
    # A segment that could not be measured is reported as WARNING, naming no
    # verdict of its own. PASS/FAIL are reserved for fabrics whose every segment
    # was actually graded, so an ungradeable segment can never read as a failure.
    status = ('WARNING' if not metric or any(s is None for s in graded)
              else 'PASS' if all(graded) else 'FAIL')
    return dict(status=status, metric=metric, distance_ratio=ratio,
                line_count=result['line_count'], selected_line=result['selected_line'],
                component_count=result['component_count'], length_px=result['length_px'],
                longest_length_px=result['longest_length_px'],
                mean_width_px=float(np.mean(pixel_widths)) if pixel_widths else None,
                end_exclusion_percent=float(end_exclusion_percent),
                length_mm=sum(c['length_mm'] for c in components) if scale is not None else None,
                longest_length_mm=main['length_mm'],
                mean_width_mm=float(np.mean(widths)) if widths else None,
                target_width_mm=float(target_width), tolerance_mm=float(tolerance),
                segments=main['segments'], components=components,
                selected_boundary=result['selected_boundary'], seconds=result['seconds'],
                coordinate_space='undistorted input frame',
                segment_basis='ten equal pixel arc-length intervals of the retained centreline in each component',
                length_basis='sum of separately measured adhesive components; no gap bridging')
