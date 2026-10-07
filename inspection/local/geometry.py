"""Geometry copied from anotate_main_line; no file/UI processing dependencies."""
import cv2
import numpy as np

def trim_endpoint_hooks(chain, hook_zone=.04, hook_angle=65., trim_top=0, trim_bottom=0):
    """Only slice endpoint cap returns; never filter or interpolate the wave."""
    if hook_zone <= 0:
        return trim_endpoint_pixels(chain, trim_top, trim_bottom)
    if len(chain) < 20:
        ys = chain[:, 1]
        chain = chain[int(np.flatnonzero(ys == ys.min())[-1]):int(np.flatnonzero(ys == ys.max())[0]) + 1].copy()
        return trim_endpoint_pixels(chain, trim_top, trim_bottom)
    height = float(np.ptp(chain[:, 1]))
    window = max(5, int(height * .002))
    zone = max(window * 2, int(height * hook_zone))
    corners = []
    for i in range(window, len(chain) - window):
        if i > zone and i < len(chain) - zone:
            continue
        before = chain[i].astype(float) - chain[i - window]
        after = chain[i + window].astype(float) - chain[i]
        norm = np.linalg.norm(before) * np.linalg.norm(after)
        if norm and np.dot(before, after) / norm < np.cos(np.deg2rad(hook_angle)):
            corners.append(i)
    top = [i for i in corners if i <= zone]
    bottom = [i for i in corners if i >= len(chain) - zone]
    start = max(top) if top else 0
    end = min(bottom) + 1 if bottom else len(chain)
    chain = chain[start:end]
    ys = chain[:, 1]
    start = int(np.flatnonzero(ys == ys.min())[-1])
    end = int(np.flatnonzero(ys == ys.max())[0]) + 1
    return trim_endpoint_pixels(chain[start:end].copy(), trim_top, trim_bottom)

def trim_endpoint_pixels(chain, top, bottom):
    """Trim by distance along the original path, measured in mask pixels."""
    if len(chain) < 2:
        return chain
    distance = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(chain.astype(float), axis=0), axis=1))))
    keep = (distance >= top) & (distance <= distance[-1] - bottom)
    return chain[keep].copy()

def extract_lines(mask, min_area=100, hook_zone=.04, hook_angle=65., trim_top=0, trim_bottom=0):
    """Drop straight backing edges/caps; keep full-resolution curved chains.

    These masks describe roughly vertical seams. A component can contain one
    curved side or two, so components are never reduced to one centerline.
    """
    binary = np.uint8(mask > 127) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = contour[:, 0, :]
        height = np.ptp(points[:, 1])
        if height < 20:
            continue
        polygon = cv2.approxPolyDP(contour, max(1., height * .002), True)[:, 0, :]
        # Polygon vertices are original contour points: retain the exact curved
        # boundary instead of drawing the simplified polygon.
        indices = [int(np.flatnonzero(np.all(points == p, axis=1))[0]) for p in polygon]
        indices.sort()
        excluded = np.zeros(len(points), dtype=bool)
        for start, end in zip(indices, indices[1:] + indices[:1]):
            delta = points[end].astype(float) - points[start]
            length = np.linalg.norm(delta)
            cap = abs(delta[1]) < abs(delta[0]) * .65 and length > height * .015
            backing = length > height * .5
            if cap or backing:
                # Include the edge interior but preserve endpoints for adjoining curves.
                arc = np.arange(start + 1, end if end > start else end + len(points)) % len(points)
                excluded[arc] = True
        if not excluded.any():
            # No confidently identified caps: report rather than output a closed outline.
            continue
        anchor = int(np.flatnonzero(excluded)[0])
        ordered = (np.arange(1, len(points) + 1) + anchor) % len(points)
        run = []
        for index in ordered:
            if excluded[index]:
                if run:
                    chain = points[run]
                    # A backing border can be split by raster stair steps. Reject
                    # the whole remaining chain if it still fits a straight line.
                    trend = np.polyfit(chain[:, 1], chain[:, 0], 1) if len(chain) > 1 else [0, 0]
                    residual = chain[:, 0] - np.polyval(trend, chain[:, 1])
                    curved = np.sqrt(np.mean(residual ** 2)) > max(2., height * .008)
                    if np.ptp(chain[:, 1]) > height * .55 and curved:
                        if chain[0, 1] > chain[-1, 1]:
                            chain = chain[::-1]
                        trimmed = trim_endpoint_hooks(chain, hook_zone, hook_angle, trim_top, trim_bottom)
                        if len(trimmed) >= 2:
                            lines.append(trimmed)
                    run = []
            else:
                run.append(index)
    return sorted(lines, key=lambda line: float(np.mean(line[:, 0])))

def filter_wave_lines(waves, source_mask, min_wave=.008, min_span=.1, min_support=.9, make_preview=True):
    if waves.shape != source_mask.shape:
        raise ValueError('Wave mask and source mask must have the same dimensions')
    binary = np.uint8(waves > 127)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    source = np.uint8(source_mask > 127)
    boundary = cv2.morphologyEx(source, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(np.uint8(boundary == 0), cv2.DIST_L2, 5)
    result = np.zeros_like(waves)
    preview = cv2.cvtColor(source_mask, cv2.COLOR_GRAY2BGR) if make_preview else None
    report = []
    for component in range(1, n):
        x, y, width, height, area = stats[component]
        region = labels[y:y+height, x:x+width] == component
        yy, xx = np.where(region)
        points = np.column_stack((xx + x, yy + y)).astype(float)
        centered = points - points.mean(axis=0)
        # PCA removes overall tilt. Measurements never modify the saved pixels.
        if len(points) > 1:
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            along = centered @ axes[0]
            across = centered @ axes[1]
        else:
            along = across = np.zeros(1)
        span = float(np.ptp(along))
        rms = float(np.sqrt(np.mean(across ** 2)))
        wave_ratio = rms / max(span, 1.)
        # Approximate stroke width for source-boundary comparison. Thick strokes
        # are allowed on both sides of the original boundary.
        stroke_width = area / max(span, 1.)
        tolerance = max(2., stroke_width / 2 + 1.)
        support = float(np.mean(distance[yy+y, xx+x] <= tolerance))
        reasons = []
        if span < min_span * max(waves.shape):
            reasons.append('too short')
        if wave_ratio < min_wave:
            reasons.append('nearly straight, no significant wave')
        if support < min_support:
            reasons.append('does not follow source-mask boundary')
        kept = not reasons
        target = result[y:y+height, x:x+width]
        target[region] = waves[y:y+height, x:x+width][region] if kept else 0
        if preview is not None:
            preview[y:y+height, x:x+width][region] = (0, 255, 0) if kept else (0, 0, 255)
        report.append({'component': component, 'kept': kept, 'reason': reasons or ['wave boundary'],
                       'wave_ratio': wave_ratio, 'span_pixels': span, 'boundary_support': support,
                       'bbox_xywh': [int(x), int(y), int(width), int(height)]})
    return result, preview, report

def offset_path(points, pixels, side='right', tangent_window=5):
    """Points must be ordered top to bottom. Image y increases downwards."""
    indices = np.arange(len(points))
    before = points[np.maximum(indices - tangent_window, 0)]
    after = points[np.minimum(indices + tangent_window, len(points) - 1)]
    tangent = after - before
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths == 0):
        raise ValueError('Cannot offset a path with coincident tangent points')
    # Right of a downward-directed line is +x in image coordinates.
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / lengths[:, None]
    if side == 'left':
        normal = -normal
    return points + pixels * normal

def mask_interior_side(points, source_mask, tangent_window=5):
    """Vote for the white-region side of a boundary, using interior samples."""
    scores = {}
    margin = max(1, len(points) // 20)
    for side in ('left', 'right'):
        votes = []
        for distance in (3., 6., 12.):
            samples = np.rint(offset_path(points, distance, side, tangent_window)[margin:-margin]).astype(int)
            valid = ((samples[:, 0] >= 0) & (samples[:, 0] < source_mask.shape[1]) &
                     (samples[:, 1] >= 0) & (samples[:, 1] < source_mask.shape[0]))
            if valid.any():
                samples = samples[valid]
                votes.extend((source_mask[samples[:, 1], samples[:, 0]] > 127).tolist())
        scores[side] = float(np.mean(votes)) if votes else 0.
    if abs(scores['left'] - scores['right']) < .2:
        raise ValueError(f'Cannot confidently identify mask interior for a line: {scores}')
    return max(scores, key=scores.get), scores

def offset_lines(mask, pixels, side='inside', thickness=0, tangent_window=5, source_mask=None, make_preview=True):
    if side in ('inside', 'outside') and (source_mask is None or source_mask.shape != mask.shape):
        raise ValueError('Inside/outside offsets require the matching source mask at the same resolution')
    binary = np.uint8(mask > 127)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(mask)
    paths = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if height < 2:
            continue
        region = np.uint8(labels[y:y+height, x:x+width] == component)
        # These seams have one x position per row. Thick strokes are reduced to
        # their row centers; the wave profile is not fitted or smoothed.
        points = np.array([[x + np.median(np.flatnonzero(row)), y + i]
                           for i, row in enumerate(region) if row.any()], dtype=float)
        scores = None
        line_side = side
        if side in ('inside', 'outside'):
            line_side, scores = mask_interior_side(points, source_mask, tangent_window)
            if side == 'outside':
                line_side = 'right' if line_side == 'left' else 'left'
        shifted = offset_path(points, pixels, line_side, tangent_window)
        if thickness:
            stroke = thickness
        else:
            # Infer the OpenCV drawing width from cross-section diameters.
            local = np.pad(region, 1)
            distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
            centers = np.rint(points - [x, y]).astype(int) + 1
            radius = float(np.median(distance[centers[:, 1], centers[:, 0]]))
            stroke = max(1, int(round(2 * radius - 3)))
        raster = np.rint(shifted).astype(np.int32)
        cv2.polylines(result, [raster], False, 255, stroke, cv2.LINE_8)
        clipped = np.any((shifted[:, 0] < 0) | (shifted[:, 0] >= mask.shape[1]) |
                         (shifted[:, 1] < 0) | (shifted[:, 1] >= mask.shape[0]))
        paths.append({'component': component, 'thickness': stroke, 'selected_side': line_side,
                      'mask_interior_scores': scores, 'clipped_at_image_border': bool(clipped),
                      'original_xy': points.tolist(), 'offset_xy': shifted.tolist()})
    # Color both layers after all lines are drawn, preserving the original mask
    # pixels. Mark overlap explicitly instead of hiding either layer.
    preview = None
    if make_preview:
        preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
        original_pixels = mask > 127
        offset_pixels = result > 127
        preview[original_pixels] = (255, 0, 0)  # base: blue (BGR)
        preview[offset_pixels] = (0, 0, 255)  # offset: red
        preview[original_pixels & offset_pixels] = (0, 255, 255)  # overlap: yellow
    return result, preview, paths

def region_mask(shape, lines):
    """Join paired endpoints; fill the unsmoothed base/offset polygon for each line."""
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        base = np.asarray(line['original_xy'], dtype=float)
        offset = np.asarray(line['offset_xy'], dtype=float)
        if base.ndim != 2 or base.shape[1] != 2 or offset.shape != base.shape or len(base) < 2:
            raise ValueError('Invalid paired base/offset coordinates')
        if not np.isfinite(base).all() or not np.isfinite(offset).all():
            raise ValueError('Nonfinite line coordinates')
        polygon = np.rint(np.concatenate((base, offset[::-1]))).astype(np.int32)
        # Fill individually so overlapping bands form a union, not an XOR.
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_8)
    return mask

def original_dimensions(line, offset_pixels, orientation='horizontal'):
    middle=(np.asarray(line['original_xy'],dtype=float)+np.asarray(line['offset_xy'],dtype=float))/2
    length=float(np.linalg.norm(np.diff(middle,axis=0),axis=1).sum())
    across=abs(float(offset_pixels))
    if not np.isfinite(length) or not np.isfinite(across) or length<=0 or across<=0:
        raise ValueError('Original-size unfolding needs positive finite length and offset.')
    along=max(2,int(np.floor(length+.5)))
    across=max(2,int(np.floor(across+.5)))
    return (along,across) if orientation=='horizontal' else (across,along)

def unfold_maps(line, width, height, orientation='vertical'):
    base = np.asarray(line['original_xy'], dtype=float)
    offset = np.asarray(line['offset_xy'], dtype=float)
    if base.ndim != 2 or base.shape[1] != 2 or offset.shape != base.shape or len(base) < 2:
        raise ValueError('Invalid paired boundaries')
    if not np.isfinite(base).all() or not np.isfinite(offset).all():
        raise ValueError('Invalid coordinate values')
    middle = (base + offset) / 2
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=1))))
    unique = np.concatenate(([True], np.diff(arc) > 1e-8))
    if arc[-1] <= 0 or np.max(np.linalg.norm(offset-base, axis=1)) < .5:
        raise ValueError('Region has zero length or width; choose a nonzero offset')
    # Uniform distance down the middle of the strip, and uniform sampling across
    # its width. The left column is always the base, even for mirrored seams.
    length_samples = width if orientation == 'horizontal' else height
    cross_samples = height if orientation == 'horizontal' else width
    rows = np.linspace(0, arc[-1], length_samples)
    left = np.column_stack([np.interp(rows, arc[unique], base[unique, i]) for i in range(2)])
    right = np.column_stack([np.interp(rows, arc[unique], offset[unique, i]) for i in range(2)])
    fraction = np.linspace(0, 1, cross_samples)[None, :, None]
    grid = left[:, None, :] * (1 - fraction) + right[:, None, :] * fraction
    if orientation == 'horizontal':
        # Source top runs left to right; the base is the top edge of the rectangle.
        grid = grid.transpose(1, 0, 2)
    return grid[:, :, 0].astype(np.float32), grid[:, :, 1].astype(np.float32), float(arc[-1])

def individual_crop(image, line):
    base = np.asarray(line['original_xy'], dtype=float)
    offset = np.asarray(line['offset_xy'], dtype=float)
    vertices = np.concatenate((base, offset))
    low = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(vertices.max(axis=0)).astype(int) + 1, [image.shape[1], image.shape[0]])
    x, y = low
    xx, yy = high
    if xx <= x or yy <= y:
        raise ValueError('Region lies outside the image')
    local = {'original_xy': (base - low).tolist(), 'offset_xy': (offset - low).tolist()}
    mask = region_mask((yy-y, xx-x), [local])
    crop = image[y:yy, x:xx] * (mask[:, :, None] // 255)
    return crop, mask, [int(x), int(y), int(xx-x), int(yy-y)]

def enhance_image(image, clip_limit=2., tiles_x=16, tiles_y=16):
    # OpenCV images are BGR. Equalizing L rather than individual color channels
    # avoids independently changing the red/green/blue contrast.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles_x, tiles_y))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Keep geometric padding outside the original image black.
    enhanced[np.all(image == 0, axis=2)] = 0
    return enhanced

def mapped_mask(binary,map_x,map_y,bbox,crop_mask):
    if binary.shape!=map_x.shape or map_y.shape!=map_x.shape:
        raise ValueError('Manual mask dimensions differ from the saved unfolded mapping; do not resize masks.')
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    local=np.zeros(crop_mask.shape,np.uint8);mapped=[];global_contours=[]
    origin=np.array(bbox[:2])
    for contour in contours:
        xy=contour[:,0,:]
        points=np.column_stack((map_x[xy[:,1],xy[:,0]],map_y[xy[:,1],xy[:,0]]))
        if not np.isfinite(points).all():raise ValueError('Mapping contains invalid coordinates.')
        global_contours.append(points.tolist())
        mapped.append(np.rint(points-origin).astype(np.int32)[:,None,:])
    if mapped:
        # Fill all nested contours together to preserve holes in manual masks.
        cv2.drawContours(local,mapped,-1,255,cv2.FILLED,hierarchy=hierarchy)
    local[crop_mask==0]=0
    return local,contours,hierarchy,global_contours

def median_profile(values,window):
    window=max(1,int(window)//2*2+1)
    if window==1:return values.copy()
    padded=np.pad(values,(window//2,window//2),mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(padded,window),axis=1)

def smooth_profile(values,window):
    median=median_profile(values,window)
    if window<=1:return median
    # Light smoothing after rejecting isolated boundary spikes. No global width fit.
    sigma=max(.5,window/6)
    return cv2.GaussianBlur(median.astype(np.float64)[None,:],(0,0),sigmaX=sigma,
                            sigmaY=0,borderType=cv2.BORDER_REPLICATE)[0]

def clean_mask(mask,window=9,max_gap=12,width_tolerance=.15,round_tips=True):
    """Operate at unfolded resolution; window/gap are measured in these pixels.

    Preserve local thickness trends. Width tolerance bounds the additional
    Gaussian smoothing relative to the robust local width, not absolute width.
    Long missing intervals are never filled.
    """
    if mask.ndim!=2:raise ValueError('Cleanup expects a 2-D mask.')
    if window<1 or max_gap<0 or not 0<=width_tolerance<=1:raise ValueError('Invalid cleanup settings.')
    binary=np.uint8(mask>127)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    result=np.zeros_like(binary)
    if count<=1:return result
    # Prefer longitudinal support to small, fat noise blobs.
    main=max(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_WIDTH]*np.sqrt(stats[i,cv2.CC_STAT_AREA]))
    ys,xs=np.where(labels==main)
    x0,x1=int(xs.min()),int(xs.max())
    columns=np.arange(x0,x1+1)
    centers=np.array([np.median(np.flatnonzero(labels[:,x]==main)) if np.any(labels[:,x]==main) else np.nan for x in columns])
    valid=np.isfinite(centers)
    centers=np.interp(columns,columns[valid],centers[valid])
    main_widths=np.array([np.count_nonzero(labels[:,x]==main) for x in columns])
    typical=max(1.,float(np.median(main_widths[main_widths>0])))
    accepted=np.zeros(count,bool);accepted[main]=True
    # Retain plausible broken pieces along the same strip; reject isolated speckles.
    for i in range(1,count):
        if i==main:continue
        x,y,w,h,area=stats[i]
        if w<max(3,typical*.25) or area<max(4,typical*2):continue
        yy,xx=np.where(labels==i)
        reference=np.interp(xx,columns,centers)
        if np.median(np.abs(yy-reference))<=typical*.8:accepted[i]=True
    selected=accepted[labels]
    present=np.flatnonzero(selected.any(axis=0))
    if not len(present):return result
    upper=np.array([np.flatnonzero(selected[:,x])[0] for x in present],float)
    lower=np.array([np.flatnonzero(selected[:,x])[-1] for x in present],float)
    # Divide at substantial gaps so cleanup never invents a long missing strip.
    splits=np.flatnonzero(np.diff(present)>max_gap+1)+1
    for indices in np.split(np.arange(len(present)),splits):
        xp=present[indices];start,end=int(xp[0]),int(xp[-1])
        xx=np.arange(start,end+1)
        top=np.interp(xx,xp,upper[indices]);bottom=np.interp(xx,xp,lower[indices])
        local_window=min(int(window)//2*2+1,max(1,len(xx)//2*2-1))
        # Median removes narrow spikes; genuine sustained changes remain in the profile.
        top=median_profile(top,local_window);bottom=median_profile(bottom,local_window)
        center=(top+bottom)/2;half=np.maximum(0,(bottom-top)/2)
        center=smooth_profile(center,local_window)
        smoothed=smooth_profile(half,local_window)
        limit=width_tolerance*np.maximum(half,.5)
        half=half+np.clip(smoothed-half,-limit,limit)
        if round_tips and len(xx)>4:
            radius=min(max(1,int(round(np.median(half)))),max(1,(len(xx)-1)//4))
            for left in (True,False):
                join=radius if left else len(xx)-1-radius
                positions=np.arange(radius+1) if left else np.arange(len(xx)-1-radius,len(xx))
                distance=np.abs(positions-join)/radius
                half[positions]=half[join]*np.sqrt(np.maximum(0,1-distance**2))
        top=np.clip(np.rint(center-half).astype(int),0,mask.shape[0]-1)
        bottom=np.clip(np.rint(center+half).astype(int),0,mask.shape[0]-1)
        for x,a,b in zip(xx,top,bottom):result[a:b+1,x]=255
    return result
