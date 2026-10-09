"""Pre-processing steps for the Pipeline Lab, and running them in a chosen order.

Each step is a plain function ``(image_bgr, roi_mask, params) -> image_bgr`` with
a description of its settings, so the same steps can be used from the window
(``tools/pipeline_lab.py``), from a script, or from a test. Nothing here needs a
display.

A pipeline is a list of ``{"step": name, "enabled": bool, "params": {...}}``
entries, applied top to bottom. It is saved and loaded as JSON.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402

from segmentation import glue_line  # noqa: E402


@dataclass(frozen=True)
class Setting:
    """One adjustable value of a step."""

    name: str
    label: str
    default: float
    low: float
    high: float
    step: float = 1.0
    whole: bool = False        # an integer setting
    pixels: bool = False       # a size in pixels: scaled with the picture for previews


@dataclass(frozen=True)
class Step:
    name: str
    label: str
    help: str
    settings: tuple
    run: callable
    resizes: bool = False      # changes the image size (the ROI is resized with it)


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _bgr(gray):
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _inside(roi):
    return roi > 0 if roi is not None and (roi > 0).any() else None


def _stretch(values, roi):
    """Scale to 0..255 using the range found inside the ROI."""
    inside = _inside(roi)
    sample = values[inside] if inside is not None else values.ravel()
    low, high = np.percentile(sample[::7] if sample.size > 7000 else sample, [0.5, 99.5])
    return np.clip((values - low) / max(high - low, 1e-6) * 255.0, 0, 255)


def resize(image, roi, params):
    factor = float(params['factor'])
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)


def denoise(image, roi, params):
    result = cv2.bilateralFilter(image, 9, float(params['strength']), 5)
    size = int(params['median']) | 1
    return cv2.medianBlur(result, size) if size >= 3 else result


def flatten(image, roi, params):
    gray = _gray(image).astype(np.float32)
    return _bgr(_stretch(gray - cv2.GaussianBlur(gray, (0, 0), float(params['radius'])), roi))


def clahe(image, roi, params):
    grid = int(params['grid'])
    tool = cv2.createCLAHE(clipLimit=float(params['clip']), tileGridSize=(grid, grid))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = tool.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def bead_enhance(image, roi, params):
    gray = _gray(image)
    result = gray.astype(np.float32)
    largest = int(params['largest'])
    for size in sorted({max(3, largest // 4) | 1, max(3, largest // 2) | 1, largest | 1}):
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result += cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, element)
        result -= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, element)
    return _bgr(_stretch(result, roi))


def local_gain(image, roi, params):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    base = cv2.GaussianBlur(light, (0, 0), float(params['radius']))
    smooth = float(params['smooth'])
    fine = cv2.GaussianBlur(light, (0, 0), smooth) if smooth > 0 else light
    lab[:, :, 0] = np.clip(base + float(params['gain']) * (fine - base), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


ALONG_DIRECTIONS = 16


def smooth_along(image, roi, params):
    # Average along the local direction of lines only. The direction comes from
    # the structure tensor; one line-shaped kernel per direction is applied and
    # each pixel takes the result for its own direction. A bead is averaged with
    # itself, not with the fabric beside it, which a blur cannot do.
    length = max(3, int(round(float(params['length']))) | 1)
    inside = _inside(roi)
    if inside is not None:                      # work on the ROI's surroundings only
        x, y, w, h = cv2.boundingRect(inside.astype(np.uint8))
        margin = length + 20
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
    else:
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
    part = image[y0:y1, x0:x1]
    gray = cv2.GaussianBlur(_gray(part).astype(np.float32), (0, 0), 2.0)
    across_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    across_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    scale = float(params['scale'])
    xx = cv2.GaussianBlur(across_x * across_x, (0, 0), scale)
    yy = cv2.GaussianBlur(across_y * across_y, (0, 0), scale)
    xy = cv2.GaussianBlur(across_x * across_y, (0, 0), scale)
    along = 0.5 * np.arctan2(2.0 * xy, xx - yy) + np.pi / 2.0          # direction of the line
    index = np.round(along / np.pi * ALONG_DIRECTIONS).astype(int) % ALONG_DIRECTIONS
    result = part.copy()
    half = length // 2
    for number in range(ALONG_DIRECTIONS):
        angle = number * np.pi / ALONG_DIRECTIONS
        kernel = np.zeros((length, length), np.float32)
        cv2.line(kernel, (int(round(half - half * np.cos(angle))), int(round(half - half * np.sin(angle)))),
                 (int(round(half + half * np.cos(angle))), int(round(half + half * np.sin(angle)))), 1.0, 1)
        chosen = index == number
        if chosen.any():
            result[chosen] = cv2.filter2D(part, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT)[chosen]
    output = image.copy()
    output[y0:y1, x0:x1] = result
    return output


def kernel_sharpen(image, roi, params):
    # The classic sharpening kernel: a positive centre with negative neighbours.
    # ``reach`` spreads the neighbours out, so that it acts on structures of the
    # bead's size and not only on single-pixel detail.
    reach = max(1, int(round(float(params['reach']))))
    strength = float(params['strength'])
    kernel = np.zeros((2 * reach + 1, 2 * reach + 1), np.float32)
    kernel[reach, reach] = 1.0 + 4.0 * strength
    for row, column in ((0, reach), (2 * reach, reach), (reach, 0), (reach, 2 * reach)):
        kernel[row, column] = -strength
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def negative(image, roi, params):
    return 255 - image


def sharpen(image, roi, params):
    soft = cv2.GaussianBlur(image, (0, 0), float(params['radius']))
    amount = float(params['amount'])
    return cv2.addWeighted(image, 1.0 + amount, soft, -amount, 0)


def quantise(image, roi, params):
    gray = _gray(image)
    inside = _inside(roi)
    samples = (gray[inside] if inside is not None else gray.ravel()).reshape(-1, 1).astype(np.float32)
    samples = samples[::max(1, len(samples) // 200000)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(samples, int(params['levels']), None, criteria, 2,
                               cv2.KMEANS_PP_CENTERS)
    centres = np.sort(centres.ravel())
    # Nearest level by lookup table: 256 comparisons instead of one per pixel.
    table = centres[np.abs(np.arange(256, dtype=np.float32)[:, None] - centres[None, :]).argmin(axis=1)]
    return _bgr(table[gray])


def shine_compress(image, roi, params):
    gray = _gray(image).astype(np.float32)
    inside = _inside(roi)
    base = float(np.median(gray[inside])) if inside is not None else float(np.median(gray))
    knee = float(params['knee'])
    above = gray - base
    squeezed = np.where(above > knee, knee + np.log1p(np.maximum(above - knee, 0)) * 4.0, above)
    return _bgr(base + squeezed)


def dim_outside(image, roi, params):
    inside = _inside(roi)
    if inside is None:
        return image
    result = image.copy()
    result[~inside] = np.median(image[inside], axis=0).astype(np.uint8)
    return result


STEPS = {step.name: step for step in (
    Step('resize', 'Resize (size increase)', 'Enlarge or shrink the image. Above 1 makes the '
         'glue line wider in pixels for the segmenter.',
         (Setting('factor', 'Factor', 2.0, 0.25, 4.0, 0.25),), resize, resizes=True),
    Step('denoise', 'Denoise', 'Edge-preserving smoothing: removes weave and compression '
         'blocks, keeps edges.',
         (Setting('strength', 'Strength', 18, 2, 80, 2), Setting('median', 'Median size', 5, 1, 11, 2, True, True)),
         denoise),
    Step('flatten', 'Flatten lighting', 'Remove the slow lighting gradient so local detail '
         'stands out.', (Setting('radius', 'Radius', 25, 5, 120, 5, pixels=True),), flatten),
    Step('clahe', 'CLAHE contrast', 'Local contrast stretch on lightness.',
         (Setting('clip', 'Clip limit', 2.0, 0.5, 8.0, 0.5), Setting('grid', 'Tile grid', 8, 2, 32, 1, True)),
         clahe),
    Step('bead_enhance', 'Bead enhance', 'Multi-scale top-hat / black-hat: pushes thin bright '
         'and dark lines apart from flat fabric.',
         (Setting('largest', 'Largest scale', 31, 7, 81, 2, True, True),), bead_enhance),
    Step('local_gain', 'Local contrast gain', 'Multiply the lightness difference from the '
         'surroundings, keeping the colour. On pale fabric the glue is only a few levels '
         'darker than the fabric; this makes that step as strong as on dark fabric.',
         (Setting('gain', 'Gain', 6.0, 1.0, 12.0, 0.5), Setting('radius', 'Radius', 40, 10, 120, 5, pixels=True),
          Setting('smooth', 'Smoothing', 1.5, 0.0, 5.0, 0.5, pixels=True)), local_gain),
    Step('sharpen', 'Sharpen', 'Unsharp mask.',
         (Setting('amount', 'Amount', 1.0, 0.1, 4.0, 0.1), Setting('radius', 'Radius', 2.0, 0.5, 10.0, 0.5, pixels=True)),
         sharpen),
    Step('smooth_along', 'Smooth along the line', 'Average along the local direction of lines '
         'only: weave and noise go, a line keeps its edges. Length is how far along; scale is '
         'the size of the neighbourhood the direction is read from.',
         (Setting('length', 'Length', 31, 7, 81, 2, True, True), Setting('scale', 'Direction scale', 6.0, 2.0, 20.0, 1.0, pixels=True)),
         smooth_along),
    Step('kernel_sharpen', 'Sharpen (negative kernel)', 'Sharpening kernel: positive centre, '
         'negative neighbours. Reach sets how far out the neighbours lie.',
         (Setting('strength', 'Strength', 1.0, 0.1, 5.0, 0.1), Setting('reach', 'Reach', 1, 1, 9, 1, True, True)),
         kernel_sharpen),
    Step('negative', 'Negative', 'Invert the picture: pale fabric becomes dark.', (), negative),
    Step('quantise', 'Quantise', 'Reduce to a few brightness levels (k-means inside the ROI).',
         (Setting('levels', 'Levels', 4, 2, 16, 1, True),), quantise),
    Step('shine_compress', 'Shine compress', 'Squeeze highlights brighter than the ROI median '
         'by more than the knee.', (Setting('knee', 'Knee', 40, 5, 120, 5),), shine_compress),
    Step('dim_outside', 'Dim outside ROI', 'Flatten everything outside the ROI to the ROI '
         'median colour.', (), dim_outside),
)}


def new_entry(name):
    """A pipeline entry for a step, with its default settings."""
    step = STEPS[name]
    return {'step': name, 'enabled': True,
            'params': {setting.name: setting.default for setting in step.settings}}


def run_pipeline(image, roi, pipeline, keep_stages=False, pixel_scale=1.0):
    """Apply the enabled steps in order.

    Returns ``(image, roi, stages)``. The ROI mask follows the image through any
    resize. ``stages`` holds ``(label, image)`` after each enabled step when asked.
    ``pixel_scale`` is the size of ``image`` relative to the full picture: settings
    that are sizes in pixels are scaled by it, so a reduced preview shows what the
    full-size run will do.
    """
    current = image
    mask = roi
    stages = []
    for number, entry in enumerate(pipeline, 1):
        if not entry.get('enabled', True):
            continue
        step = STEPS[entry['step']]
        params = {setting.name: entry.get('params', {}).get(setting.name, setting.default)
                  for setting in step.settings}
        if pixel_scale != 1.0:
            for setting in step.settings:
                if setting.pixels:
                    params[setting.name] = max(1.0, params[setting.name] * pixel_scale)
        current = step.run(current, mask, params)
        if mask is not None and mask.shape[:2] != current.shape[:2]:
            mask = cv2.resize(mask, (current.shape[1], current.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if keep_stages:
            stages.append((f'{number}. {step.label}', current))
    return current, mask, stages


def save_recipe(path, recipe):
    Path(path).write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')


def load_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding='utf-8'))
    for entry in recipe.get('pipeline', []):
        if entry.get('step') not in STEPS:
            raise ValueError(f"Unknown step in recipe: {entry.get('step')!r}")
    return recipe


def ring_stats(ring):
    """Length and width of an outlined strip, from its own outline, in pixels."""
    ring = np.round(np.asarray(ring)).astype(np.int32)
    x, y, w, h = cv2.boundingRect(ring)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.fillPoly(mask, [ring - (x - 2, y - 2)], 1)
    from skimage.morphology import skeletonize
    spine = skeletonize(mask > 0)
    if not spine.any():
        return {'length_px': 0.0, 'width_px': 0.0, 'width_p10': 0.0, 'width_p90': 0.0}
    across = 2.0 * cv2.distanceTransform(mask, cv2.DIST_L2, 5)[spine]
    low, high = np.percentile(across, [10, 90])
    return {'length_px': float(spine.sum()), 'width_px': float(np.median(across)),
            'width_p10': float(low), 'width_p90': float(high)}


# --- segmenters ---------------------------------------------------------------

DEFAULT_ROI_MODEL = ('model_for_segment_bra_roi places/model_test_and_running_scripts/'
                     'models/strip_unet_resnet34.onnx')


def model_roi(image, model_path=DEFAULT_ROI_MODEL, input_width=1152, on_gpu=False):
    """The glue-bearing panel regions from the trained model, cardboard removed.

    ``on_gpu`` runs the model's PyTorch checkpoint (the ``.pt`` beside the ONNX
    file) with torch, on the GPU when there is one, instead of ONNX Runtime."""
    if on_gpu:
        model_path = str(Path(model_path).with_suffix('.pt'))
    region = glue_line.glue_region(image, model_path, 0.0, input_width, True)
    return region & (1 - glue_line.cardboard_mask(image, 60.0))


def segment_local(image, roi, guided):
    """Glue lines from the local detector. ``guided`` looks for one line along the
    curved side of each ROI region (for panel-shaped ROIs); otherwise any bead
    inside the ROI is reported."""
    settings = {'glue_line': {'background': 'any', 'roi_guided': bool(guided),
                              'guided_skip_frame_filters': True,
                              'roi_strict': True, 'min_length_px': 400.0}}
    segmenter = glue_line.GlueLineSegmenter(settings)
    segmenter.region_override = roi if roi is not None else np.ones(image.shape[:2], np.uint8)
    result = segmenter.segment(image)
    return [{'ring': instance.polygon, 'source': 'local', 'confidence': instance.confidence}
            for instance in result.instances]


def _centre(ring):
    half = len(ring) // 2
    return (ring[:half] + ring[half:][::-1]) / 2.0


def fabric_colour(image, roi):
    """'dark', 'pink' or 'white': the fabric inside the ROI, from its brightness and
    saturation. Shown with the result; which filters suit which colour was measured
    per colour (see AUTOMATIC_INSPECTION.md, Pipeline Lab)."""
    inside = _inside(roi)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sample = hsv[inside] if inside is not None else hsv.reshape(-1, 3)
    sample = sample[::max(1, len(sample) // 200000)]
    if np.median(sample[:, 2]) < 110:
        return 'dark'
    return 'pink' if np.median(sample[:, 1]) > 25 else 'white'


RIBBON_MAX_MISFIT = 6.0   # pixels: a ribbon further than this from its evidence is not used
RIBBON_MIN_COVER = 0.7    # nor one covering less than this share of the outline it came from
RIBBON_CHECK_MISFIT = 2.0  # above this, or below RIBBON_CHECK_BACKED, the result says 'check'
RIBBON_CHECK_BACKED = 0.8
NEAR_OUTLINE = 22       # the band's middle is looked for this close to an outline's middle
CLEAN_SHARE = 0.8      # a line SAM outlined this evenly sets the frame's usual bead width
ASK_AGAIN_BELOW = 0.95  # a line outlined less evenly than this is put to SAM again
SECOND_LOOKS = ((560, 110), (1000, 200))   # tile stride and point spacing of the further questions


def edge_methods_available():
    """Ways of placing the band's edges that can be used now: the trained trees
    only once their model file exists and LightGBM is installed."""
    import importlib.util
    from segmentation import bead_ml
    names = ['slope', 'profile_fit']
    if bead_ml.MODEL_FILE.is_file() and importlib.util.find_spec('lightgbm') is not None:
        names.insert(0, 'trees')
    return names


def gpu_available():
    """Whether SAM and the ROI model can run on this computer's own GPU."""
    import importlib.util
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('transformers') is None:
        return False
    import torch
    return bool(torch.cuda.is_available())


def segment_sam_from_lines(image, roi, lines, model, client=None, max_width_px=80.0, scale=1.0,
                           progress=None, prepare=None, second_looks=SECOND_LOOKS):
    """Outline each roughly found line with SAM, prompted by points along it.

    Each line is handled inside its own ROI region: SAM's mask is cut to the ROI
    (the glue is always inside it) and read as a ribbon along the rough line.
    Where SAM spilled into the band or mesh beside the bead, that stretch takes
    the line's steady course instead; ``seen`` is the share of the two edges SAM
    outlined itself. A line SAM took too wide over most of its length is corrected
    with the width it gave the frame's evenly outlined lines.

    Dark fabric is outlined evenly at the first asking. Pale fabric is not, and a
    filter does not change that (several were measured); asking again with other
    tiles and points does, because SAM then spills somewhere else and the steady
    parts of the answers are put together. Only lines that need it are asked again.
    ``scale`` is the picture's enlargement by the steps, for the tile sizes.
    ``second_looks`` are the tile stride and point spacing of the further askings;
    empty to ask once only. ``model`` beginning ``local:`` runs on this computer.
    """
    from segmentation import sam_refine
    say = progress or (lambda text: None)
    inside = None if roi is None else (roi > 0).astype(np.uint8)

    def ask(centre, **layout):
        mask, confidences = sam_refine.refine_line(image, centre, max_width_px, client=client,
                                                   region=roi, model=model, prepare=prepare, **layout)
        return (mask & inside if inside is not None else mask), confidences

    answers = []
    for number, line in enumerate(lines, 1):
        say(f'Asking SAM to outline line {number} of {len(lines)}...')
        centre = _centre(line['ring'])
        mask, confidences = ask(centre)
        answers.append([centre, [mask], confidences, sam_refine.trim_ring(mask, centre)])

    def usual_width():
        widths = ([float(np.median(trimmed[2])) for *_, trimmed in answers
                   if trimmed and trimmed[1] >= CLEAN_SHARE]
                  or [float(np.median(trimmed[2])) for *_, trimmed in answers if trimmed])
        return float(np.median(widths)) if widths else None

    def retrim(answer, usual):
        centre, masks, _, trimmed = answer
        if usual and (trimmed is None or trimmed[1] < ASK_AGAIN_BELOW or len(masks) > 1):
            again = sam_refine.trim_ring(masks, centre, usual)
            if again and (trimmed is None or again[1] >= trimmed[1]
                          or abs(np.median(trimmed[2]) - usual) > 0.25 * usual):
                answer[3] = again

    usual = usual_width()
    for number, answer in enumerate(answers, 1):
        retrim(answer, usual)
        for stride, spacing in second_looks:
            if answer[3] is not None and answer[3][1] >= ASK_AGAIN_BELOW:
                break
            say(f'Line {number}: asking SAM again with other tiles...')
            mask, confidences = ask(answer[0], stride=int(stride * scale), spacing=int(spacing * scale))
            answer[1].append(mask)
            answer[2] = answer[2] + confidences
            retrim(answer, usual)
    found = []
    for line, (centre, masks, confidences, trimmed) in zip(lines, answers):
        if trimmed is None or np.median(trimmed[2]) > max_width_px:
            found.append(dict(line, source='local (SAM answer refused)', seen=0.0))
            continue
        ring, seen, _ = trimmed
        evidence = {}
        if sam_refine.trim_ring(masks, centre, usual, details=evidence) is None:
            sam_refine.trim_ring(masks, centre, details=evidence)     # read against its own width
        found.append({'ring': ring, 'source': model if seen >= CLEAN_SHARE else f'{model} (part repaired)',
                      'seen': seen, 'asked': len(masks), 'evidence': evidence, 'rough': centre,
                      'confidence': float(np.mean(confidences)) if confidences else 0.0})
    return found


def finish_on_bead_edges(image, roi, lines, scale=1.0):
    """Replace each line's outline by the bead's own two edges.

    SAM (or the local detector) says where the line runs; on pale fabric its
    outline drifts off the bead and runs past its ends. The edges themselves are
    read from the original picture along that line (``segmentation.bead_edges``).
    ``image`` and ``roi`` are the original picture; ``scale`` is the enlargement of
    the picture the lines are in. The earlier outline is kept as ``first_ring``.
    """
    from segmentation import bead_edges
    finished = []
    for line in lines:
        answer = bead_edges.bead_edges(image, _centre(np.asarray(line['ring']) / scale), roi)
        if answer is None:
            finished.append(dict(line, edges='not found: outline left as it was'))
            continue
        ring, seen, _ = answer
        finished.append(dict(line, ring=ring * scale, first_ring=line['ring'],
                             first_source=line['source'], source=line['source'] + ' + bead edges',
                             edges_seen=seen))
    return finished


def finish_as_ribbon(image, roi, lines, scale=1.0, sources=('sam', 'edges'), edge_method='slope'):
    """Replace each line's outline by one smooth ribbon of even width.

    The edges found so far are evidence: SAM's own edges (kept from the outline
    step) and the bead's edges read from the original picture, each weighted by
    what the picture is like at the place (shiny, matte, one-sided, weak). The
    ribbon is fitted to them (``segmentation.ribbon``); its ends are where the
    bead's dark band ends. ``image`` and ``roi`` are the original picture;
    ``scale`` is the enlargement of the picture the lines are in. ``edge_method``
    is how the band's edges are placed: ``slope`` (steepest slope), ``profile_fit``
    or ``trees`` (the trained corrector, ``segmentation.bead_ml``).

    Each line gains ``track`` (sample points and their state, for drawing),
    ``backed``, ``misfit``, ``rough`` and ``deviations`` (stretches where the
    evidence is steadily wider or narrower than the ribbon: ``(points, pixels)``).
    """
    from segmentation import bead_edges, edge_methods, ribbon
    weights = {name: ribbon.STATE_WEIGHT[name] for name in sources}
    placing = None
    if edge_method not in (None, 'slope'):
        if edge_method == 'trees':
            import importlib
            importlib.import_module('segmentation.bead_ml')      # registers the method
            # The trees already allow for the core lying at an edge, which is why the
            # steepest slope is given little weight there.
            weights = dict(weights, edges=dict(weights.get('edges', ribbon.STATE_WEIGHT['edges']),
                                               **{'one-sided': 1.0}))
        placing = edge_methods.METHODS[edge_method]
    finished = []
    for line in lines:
        # Along the middle of the outline found so far, and close to it: SAM is
        # rarely on the wrong stripe, the band tracker left to itself sometimes is.
        middle = _centre(np.asarray(line['ring'])) / scale
        measured = bead_edges.measure(image, middle, roi, edge_method=placing,
                                      search=NEAR_OUTLINE if line.get('evidence') else bead_edges.SEARCH)
        if measured is None:
            finished.append(dict(line, ribbon='no band found: outline left as it was'))
            continue
        sam = line.get('evidence')
        if sam and scale != 1.0:
            sam = dict(sam, points=sam['points'] / scale, low=sam['low'] / scale, high=sam['high'] / scale)
        low, high = ribbon.gather(measured, sam, weights)
        span = measured['span']
        fitted = ribbon.fit_ribbon(measured['points'][span], low, high)
        if fitted is None:
            finished.append(dict(line, ribbon='too little evidence: outline left as it was'))
            continue
        centre = fitted['centre']
        covered = len(centre) * bead_edges.STEP / max(1.0, float(
            np.linalg.norm(np.diff(middle, axis=0), axis=1).sum()))
        if fitted['misfit'] > RIBBON_MAX_MISFIT or covered < RIBBON_MIN_COVER:
            finished.append(dict(line, ribbon=f"does not fit the evidence (misfit {fitted['misfit']:.0f} px, "
                                              f"{covered:.0%} of the line): outline left as it was"))
            continue
        finished.append(dict(
            line, ring=fitted['ring'] * scale, first_ring=line['ring'], first_source=line['source'],
            source=line['source'] + ' + ribbon', backed=fitted['backed'], misfit=fitted['misfit'],
            rough_px=fitted['rough'], width_noise=fitted.get('width_noise', 0.0),
            doubtful=fitted['misfit'] > RIBBON_CHECK_MISFIT or fitted['backed'] < RIBBON_CHECK_BACKED,
            track={'points': measured['points'][span] * scale, 'state': measured['state'][span]},
            deviations=[(centre[first:last + 1] * scale, pixels)
                        for first, last, pixels in fitted['deviations']]))
    return finished


def segment_sam_from_clicks(image, points, model, client=None):
    """One SAM call on the area around hand-placed points.

    ``points`` is a list of ``(x, y, positive)`` in image pixels. The area sent is
    the points' bounding box with a margin, at least 1024 pixels square.
    """
    from segmentation import sam_refine
    if not any(positive for _, _, positive in points):
        raise ValueError('Click at least one point on the glue line (left click).')
    if client is None and not model.startswith('local:'):
        client = sam_refine._client()
    xy = np.array([(x, y) for x, y, _ in points], np.float64)
    height, width = image.shape[:2]
    size = int(max(1024, np.ptp(xy[:, 0]) + 400, np.ptp(xy[:, 1]) + 400))
    middle = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    x0 = int(np.clip(middle[0] - size / 2, 0, max(0, width - size)))
    y0 = int(np.clip(middle[1] - size / 2, 0, max(0, height - size)))
    tile = np.ascontiguousarray(image[y0:y0 + size, x0:x0 + size])
    prompt = [{'points': [{'x': float(x - x0), 'y': float(y - y0), 'positive': bool(positive)}
                          for x, y, positive in points]}]
    mask, confidence = sam_refine.reply_mask(
        sam_refine.segment_points(client, tile, prompt, model), tile.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [{'ring': contour.reshape(-1, 2).astype(np.float64) + (x0, y0), 'source': model,
             'confidence': confidence}
            for contour in contours if cv2.contourArea(contour) >= 200]


def segment_sam_text(image, roi, prompt, client=None):
    """SAM 3 with a text prompt on the ROI's bounding area."""
    import sam_detection
    if client is None:
        client = sam_detection.shared_client()
    x, y, w, h = (cv2.boundingRect((roi > 0).astype(np.uint8)) if roi is not None and (roi > 0).any()
                  else (0, 0, image.shape[1], image.shape[0]))
    crop = np.ascontiguousarray(image[y:y + h, x:x + w])
    polygons = sam_detection.extract_polygons(sam_detection.call_workflow(client, crop, [prompt]))
    return [{'ring': polygon.ring.astype(np.float64) + (x, y), 'source': 'sam3 text',
             'confidence': polygon.confidence} for polygon in polygons]
