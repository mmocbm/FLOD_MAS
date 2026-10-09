"""The known shape of a glue line, used as a condition on what is reported.

On these products the glue line is always the same wave: two humps on one side of
the straight line joining its ends, with a dip between them -- a "W". Seen from
the other side, or on the mirrored panel, it is the same W flipped.

A template stores that wave with its ends at (0, 0) and (1, 0). Detection
evidence is rarely the whole line: an edge fades, mesh covers a stretch, a fold
looks like glue for a while. Here each piece of evidence proposes where the whole
W would have to lie for that piece to be part of it; a proposal is kept only if
the rest of the evidence agrees along enough of its length. What comes out is the
complete line, and pieces that belong to no W are left out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

TEMPLATE_POINTS = 201
SHAPES_FILE = Path(__file__).resolve().with_name('glue_shapes.json')


@dataclass(frozen=True)
class Template:
    """One wave shape. ``points`` run from (0, 0) to (1, 0), evenly spaced along
    the curve; ``chord_px`` is the end-to-end distance it was learned at."""

    name: str
    points: np.ndarray
    chord_px: float

    @property
    def arc(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]

    def at(self, arc_positions: np.ndarray, mirror: float = 1.0) -> np.ndarray:
        """Template points at the given distances along the curve."""
        arc = self.arc
        return np.column_stack([np.interp(arc_positions, arc, self.points[:, 0]),
                                mirror * np.interp(arc_positions, arc, self.points[:, 1])])


@dataclass
class Placement:
    """A template laid into the image: ``path`` is the whole line in pixels."""

    template: Template
    mirror: float
    matrix: np.ndarray          # 2x3, template coordinates -> image pixels
    path: np.ndarray            # (N, 2) evenly spaced along the line
    seen: np.ndarray            # (N,) bool: evidence found beside this sample
    residual: float             # RMS distance of that evidence from the path, px

    @property
    def seen_share(self) -> float:
        return float(self.seen.mean())

    @property
    def scale(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))


def load_templates(names=None, path=SHAPES_FILE) -> list[Template]:
    """Templates from the shapes file; all of them when ``names`` is empty."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    wanted = set(names or [])
    return [Template(item['name'], np.asarray(item['points'], np.float64), float(item['chord_px']))
            for item in data['templates'] if not wanted or item['name'] in wanted]


def save_templates(templates, path=SHAPES_FILE) -> None:
    payload = {'templates': [
        {'name': item.name, 'chord_px': round(float(item.chord_px), 1),
         'points': np.round(item.points, 5).tolist()} for item in templates]}
    Path(path).write_text(json.dumps(payload) + '\n', encoding='utf-8')


def resample(points: np.ndarray, count: int | None = None, step: float | None = None) -> np.ndarray:
    """Points evenly spaced along a polyline, by number or by spacing."""
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if count is None:
        count = max(2, int(round(distance[-1] / step)) + 1)
    targets = np.linspace(0.0, distance[-1], count)
    return np.column_stack([np.interp(targets, distance, points[:, 0]),
                            np.interp(targets, distance, points[:, 1])])


def to_chord_frame(points: np.ndarray) -> tuple[np.ndarray, float]:
    """A line re-expressed with its ends at (0, 0) and (1, 0), humps upward."""
    line = resample(np.asarray(points, np.float64), TEMPLATE_POINTS)
    axis = line[-1] - line[0]
    chord = float(np.linalg.norm(axis))
    axis /= chord
    normal = np.array([-axis[1], axis[0]])
    local = np.column_stack([(line - line[0]) @ axis, (line - line[0]) @ normal]) / chord
    if local[:, 1].mean() < 0:      # the same W seen from the other side
        local[:, 1] *= -1.0
    return local, chord


def learn_template(name: str, lines: list[np.ndarray]) -> tuple[Template, float]:
    """The average wave of several whole lines, and how far they stray from it
    (RMS, as a share of the chord)."""
    frames = [to_chord_frame(line) for line in lines]
    shapes = [frames[0][0]]
    for local, _ in frames[1:]:
        # A line may have been traced from either end.
        flipped = np.column_stack([1.0 - local[::-1, 0], local[::-1, 1]])
        first = shapes[0]
        shapes.append(local if np.abs(local - first).sum() <= np.abs(flipped - first).sum()
                      else flipped)
    mean = resample(np.mean(shapes, axis=0), TEMPLATE_POINTS)
    mean = (mean - mean[0]) / np.linalg.norm(mean[-1] - mean[0])
    spread = float(np.sqrt(np.mean([(shape[:, 1] - mean[:, 1]) ** 2 for shape in shapes])))
    return Template(name, mean, float(np.mean([chord for _, chord in frames]))), spread


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares rotation, uniform scale and shift taking source onto target."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    a, b = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(b.T @ a)
    sign = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ np.diag([1.0, sign]) @ vt
    scale = float((singular * [1.0, sign]).sum() / max((a ** 2).sum(), 1e-12))
    matrix = np.column_stack([scale * rotation, target_mean - scale * rotation @ source_mean])
    residual = float(np.sqrt(np.mean(np.sum((source @ matrix[:, :2].T + matrix[:, 2] - target) ** 2,
                                            axis=1))))
    return matrix, residual


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def propose(piece: np.ndarray, template: Template, scale_range, keep: int = 3,
            offset_step: float = 0.02) -> list[tuple[float, float, np.ndarray]]:
    """Where the whole template would lie if ``piece`` were part of it.

    The piece may be any stretch of the line, traced from either end, and the
    line may be mirrored, so every such reading is tried. Returns the best few
    as ``(residual_px, mirror, matrix)``, clearly different placements only.
    """
    piece = resample(np.asarray(piece, np.float64), step=8.0)
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
    total = template.arc[-1]
    low, high = (template.chord_px * factor for factor in scale_range)
    def match(guess, start, mirror, points):
        """Fit with the piece read at this size and starting this far along."""
        span = along[-1] / guess
        if not (low <= guess <= high and 0.0 <= start and start + span <= total * 1.001):
            return None
        matrix, residual = _similarity(template.at(start + along / guess, mirror), points)
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        # The fitted size must agree with the one assumed for the
        # correspondences, or the match is not the one tested.
        if not (low <= scale <= high and abs(scale / guess - 1.0) < 0.06):
            return None
        return residual, matrix

    coarse = []
    for mirror in (1.0, -1.0):
        for points in (piece, piece[::-1]):
            for guess in np.linspace(low, high, 7):
                span = along[-1] / guess            # the piece's length in template units
                for start in np.arange(0.0, max(total - span, 0.0) + 1e-9, offset_step * total):
                    result = match(guess, start, mirror, points)
                    if result is not None:
                        coarse.append((result[0], guess, start, mirror, points))
    coarse.sort(key=lambda item: item[0])
    # A smooth wave lets a piece slide and stretch a little with almost no
    # change in fit, so the best coarse readings are settled exactly.
    found = []
    for _, guess, start, mirror, points in coarse[:12]:
        def cost(values, mirror=mirror, points=points):
            result = match(values[0], values[1], mirror, points)
            return 1e6 if result is None else result[0]
        best = minimize(cost, [guess, start], method='Nelder-Mead',
                        options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 120})
        result = match(best.x[0], best.x[1], mirror, points)
        if result is not None:
            found.append((result[0], mirror, result[1]))
    found.sort(key=lambda item: item[0])
    chosen: list[tuple[float, float, np.ndarray]] = []
    for residual, mirror, matrix in found:
        ends = _apply(matrix, np.array([[0.0, 0.0], [1.0, 0.0]]))
        if all(np.abs(ends - _apply(other, np.array([[0.0, 0.0], [1.0, 0.0]]))).max() > 25.0
               or mirror != other_mirror for _, other_mirror, other in chosen):
            chosen.append((residual, mirror, matrix))
        if len(chosen) >= keep:
            break
    return chosen


def place(template: Template, mirror: float, matrix: np.ndarray, tree: cKDTree,
          evidence: np.ndarray, tolerance: float, step: float = 4.0,
          flex_px: float = 0.0) -> Placement:
    """Settle a proposed placement onto the evidence and say how much of it is seen.

    The placement is refitted to the evidence lying beside it a few times. With
    ``flex_px`` the line may then also bend sideways, smoothly and by no more than
    that, to follow fabric that was laid down slightly stretched.
    """
    count = max(8, int(round(template.arc[-1] * np.sqrt(abs(np.linalg.det(matrix[:, :2]))) / step)))
    shape = template.at(np.linspace(0.0, template.arc[-1], count), mirror)
    for _ in range(4):
        path = _apply(matrix, shape)
        distance, index = tree.query(path, distance_upper_bound=2.0 * tolerance)
        near = np.isfinite(distance)
        if near.sum() < 8:
            break
        matrix, _ = _similarity(shape[near], evidence[index[near]])
    path = _apply(matrix, shape)
    if flex_px > 0:
        distance, index = tree.query(path, distance_upper_bound=2.0 * tolerance)
        near = np.isfinite(distance)
        if near.sum() >= 12:
            tangent = np.gradient(path, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            offset = np.einsum('ij,ij->i', evidence[index[near]] - path[near], normal[near])
            position = np.linspace(-1.0, 1.0, count)
            # A low-order curve with a penalty keeps the bend smooth and stops it
            # swinging away where there is no evidence to hold it.
            basis = np.polynomial.legendre.legvander(position, 5)
            ridge = 0.5 * near.sum() / count * np.eye(basis.shape[1])
            weights = np.linalg.solve(basis[near].T @ basis[near] + ridge, basis[near].T @ offset)
            path = path + normal * np.clip(basis @ weights, -flex_px, flex_px)[:, None]
    distance, _ = tree.query(path, distance_upper_bound=tolerance)
    seen = np.isfinite(distance)
    residual = float(np.sqrt(np.mean(distance[seen] ** 2))) if seen.any() else float('inf')
    return Placement(template, mirror, matrix, path, seen, residual)


def find_lines(pieces: list[np.ndarray], evidence: np.ndarray, templates: list[Template],
               scale_range=(0.75, 1.3), tolerance: float = 7.0, min_seen_share: float = 0.4,
               min_piece_px: float = 300.0, flex_px: float = 12.0,
               inside=None) -> list[Placement]:
    """Whole glue lines: template placements that the evidence supports.

    ``pieces`` are ordered centreline stretches, ``evidence`` every bead centre
    point found. ``inside`` is an optional callable giving, for (N, 2) points, the
    share lying on fabric; a line must lie on fabric.
    """
    if len(evidence) == 0 or not templates:
        return []
    tree = cKDTree(evidence)
    candidates: list[Placement] = []
    for piece in pieces:
        length = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
        if length < min_piece_px:
            continue
        for template in templates:
            for _, mirror, matrix in propose(piece, template, scale_range, keep=6):
                placement = place(template, mirror, matrix, tree, evidence, tolerance,
                                  flex_px=flex_px)
                low, high = (template.chord_px * factor for factor in scale_range)
                if not (low <= placement.scale <= high
                        and placement.seen_share >= min_seen_share
                        and (inside is None or inside(placement.path) >= 0.85)):
                    continue
                # The piece that suggested this line must itself lie along it. A
                # bead that runs with the line for a while and then leaves it is
                # saying the line is somewhere else.
                gap, _ = cKDTree(placement.path).query(piece)
                if np.mean(gap <= 2.0 * tolerance) >= 0.9:
                    candidates.append(placement)
    # Best supported first; a line already taken cannot be claimed again.
    candidates.sort(key=lambda item: -float(item.seen.sum()))
    accepted: list[Placement] = []
    for candidate in candidates:
        clash = False
        for other in accepted:
            distance, _ = cKDTree(other.path).query(candidate.path, distance_upper_bound=30.0)
            if np.isfinite(distance).mean() > 0.3:
                clash = True
                break
        if not clash:
            accepted.append(candidate)
    return _agreeing_direction(accepted)


def _agreeing_direction(lines: list[Placement], limit_degrees: float = 35.0) -> list[Placement]:
    """Drop a line lying across the others.

    Panels are laid side by side, so their glue lines run the same way. A line
    fitted across them has been assembled from weave or moire, not from a bead.
    With fewer than three lines there is no majority to compare with.
    """
    if len(lines) < 3:
        return lines
    ends = np.array([line.path[-1] - line.path[0] for line in lines])
    angle = np.degrees(np.arctan2(ends[:, 1], ends[:, 0])) % 180.0
    # Angles wrap at 180, so the middle direction is found on the doubled angle.
    doubled = np.radians(2.0 * angle)
    middle = np.degrees(np.arctan2(np.median(np.sin(doubled)), np.median(np.cos(doubled)))) / 2.0
    apart = np.abs((angle - middle + 90.0) % 180.0 - 90.0)
    return [line for line, gap in zip(lines, apart) if gap <= limit_degrees]
