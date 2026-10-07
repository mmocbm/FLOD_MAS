"""Calibrate the new refolded centreline and boundary intersections."""
import copy
import numpy as np


def measurement_record(result, scale, target_width, tolerance):
    """Keep reference pixel measurements; map points, never use a global px scale."""
    components = copy.deepcopy(result['components'])
    for component in components:
        samples = component['samples']
        if scale is not None:
            centres = scale.to_mm([[s['center_x'], s['center_y']] for s in samples])
            component['length_mm'] = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
        else:
            component['length_mm'] = None
        for sample in samples:
            sample['width_mm'] = None
            if scale is not None and sample['valid']:
                endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                         [sample['right_x'], sample['right_y']]])
                width = float(np.linalg.norm(endpoints[1] - endpoints[0]))
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
    graded = [s['within_tolerance'] for s in main['segments']]
    metric = scale is not None and bool(widths)
    status = ('UNMEASURED' if not metric or any(s is None for s in graded)
              else 'PASS' if all(graded) else 'FAIL')
    return dict(status=status, metric=metric,
                line_count=result['line_count'], selected_line=result['selected_line'],
                component_count=result['component_count'], length_px=result['length_px'],
                longest_length_px=result['longest_length_px'], mean_width_px=result['mean_width_px'],
                length_mm=sum(c['length_mm'] for c in components) if scale is not None else None,
                longest_length_mm=main['length_mm'],
                mean_width_mm=float(np.mean(widths)) if widths else None,
                target_width_mm=float(target_width), tolerance_mm=float(tolerance),
                segments=main['segments'], components=components,
                selected_boundary=result['selected_boundary'], seconds=result['seconds'],
                coordinate_space='undistorted input frame',
                segment_basis='ten equal pixel arc-length intervals of the longest component',
                length_basis='sum of separately measured adhesive components; no gap bridging')
