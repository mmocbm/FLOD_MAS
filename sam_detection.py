"""SAM3 object detection through the Roboflow serverless workflow.

Ported from the standalone ``polygon_preview`` tool so the inspection pipeline
reuses its prompt handling, response parsing and drawing. The workflow's SAM3
step is configured with ``output_format="polygons"``, so each prediction carries
a ``points`` list of ``{"x": .., "y": ..}`` vertices in source-image pixels
instead of an ``rle_mask``. Each disjoint mask region arrives as its own
prediction, so one object can produce several.

Detection is fail-soft by design: :func:`detect_crop` never raises for a network,
key or response problem. It returns the error on the result instead, because the
inspection must still save its raw crops and report a warning rather than lose
the capture.
"""

from __future__ import annotations

import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

import plane_scale
import strip_analysis

try:
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.http.entities import InferenceConfiguration
except ImportError:  # Reported by detect_crop; the app still starts without it.
    InferenceHTTPClient = None
    InferenceConfiguration = None

WORKSPACE_NAME = "obhashcolab-1"
WORKFLOW_ID = "sam3-with-prompts"
API_URL = "https://serverless.roboflow.com"
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"
_ENV_FILE = Path(__file__).resolve().with_name(".env")

# supervision's default palette, in BGR (OpenCV channel order).
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
)

Point = tuple[float, float]

_client: Any = None


class PolygonError(RuntimeError):
    """The workflow call or its response could not be used."""


@dataclass(frozen=True)
class Polygon:
    """One predicted region: a closed ring plus what it was labelled as."""

    class_name: str
    confidence: float
    ring: np.ndarray  # (N, 2) int32, x/y in source-image pixels


@dataclass(frozen=True)
class CropDetection:
    """Outcome of one crop's workflow call."""

    polygons: list[Polygon]
    upload_bytes: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class StripMeasurement:
    """A strip measured from one of the detected polygons."""

    polygon_index: int
    analysis: strip_analysis.StripAnalysis


# --- prompts and credentials -------------------------------------------------


def parse_prompts(text: str) -> list[str]:
    """Split a comma-separated prompt string into class names.

    Only commas separate, so a class name may contain spaces. Newlines separate
    as well, and empty chunks are dropped, so trailing or doubled commas are
    harmless.
    """
    names: list[str] = []
    for chunk in text.replace("\n", ",").split(","):
        name = " ".join(chunk.split())
        if name:
            names.append(name)
    return names


def load_api_key() -> str:
    """Read the key from the environment, falling back to a sibling .env."""
    import os

    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if key:
        return key
    if _ENV_FILE.is_file():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == API_KEY_ENV_VAR:
                value = value.strip().strip("'\"")
                if value:
                    return value
    raise PolygonError(
        f"No API key. Set {API_KEY_ENV_VAR} in the environment or add it to {_ENV_FILE}."
    )


def build_client(api_key: str):
    """SDK client that sends the key as a bearer header, never in the URL."""
    if InferenceHTTPClient is None or InferenceConfiguration is None:
        raise PolygonError(
            "The inference_sdk package is not installed. Run "
            "pip install -r requirements.txt"
        )
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key).configure(
        InferenceConfiguration(api_key_transport="header")
    )


def shared_client():
    """Build the client once per process; a missing key retries next time."""
    global _client
    if _client is None:
        _client = build_client(load_api_key())
    return _client


# --- the API call ------------------------------------------------------------


def call_workflow(client, image: str, prompts: Sequence[str]) -> list[dict[str, Any]]:
    """Run the workflow and return the raw prediction dicts.

    ``image`` is a local path or an https URL; the SDK rejects ``pathlib.Path``,
    so it must be a string.
    """
    response = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image},
        parameters={"prompts": list(prompts)},
        use_cache=False,
    )

    if not isinstance(response, list) or not response:
        raise PolygonError(f"Unexpected workflow response: {type(response).__name__}")

    entry = response[0]
    if not isinstance(entry, dict):
        raise PolygonError(f"Unexpected first output entry: {type(entry).__name__}")

    sam = entry.get("sam")
    if not isinstance(sam, dict):
        raise PolygonError(f"No 'sam' output in response; got keys {sorted(entry)}")

    predictions = sam.get("predictions")
    if predictions is None:
        return []
    if not isinstance(predictions, list):
        raise PolygonError(f"'predictions' was {type(predictions).__name__}, expected list")
    return [p for p in predictions if isinstance(p, dict)]


def extract_polygons(predictions: Sequence[dict[str, Any]]) -> list[Polygon]:
    """Pull each prediction's ``points`` out as a closed ring of int32 pixels.

    Rings with fewer than 3 vertices are dropped: they enclose no area and
    cannot be drawn. Malformed points are skipped rather than raising, so one
    bad prediction cannot lose the whole result.
    """
    polygons: list[Polygon] = []
    for prediction in predictions:
        raw = prediction.get("points")
        if not isinstance(raw, list):
            continue

        vertices: list[Point] = []
        for point in raw:
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[0], point[1]
            else:
                continue
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                vertices.append((x, y))

        if len(vertices) < 3:
            continue  # a 1- or 2-point "polygon" has no area

        confidence = prediction.get("confidence")
        polygons.append(
            Polygon(
                class_name=str(prediction.get("class") or "?"),
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                ring=np.round(np.asarray(vertices, dtype=np.float64)).astype(np.int32),
            )
        )
    return polygons


def _call_with_timeout(function: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``function`` on a daemon thread and give up after ``timeout_seconds``.

    The SDK exposes no portable request timeout, so the deadline is enforced
    here. An abandoned call keeps running on its own thread until the process
    exits; it can no longer affect the inspection.
    """
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = function()
        except BaseException as error:  # noqa: BLE001 - handed back to the caller
            outcome["error"] = error

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise PolygonError(
            f"The detection request did not finish within {timeout_seconds:g} seconds"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """Encode a crop as JPEG bytes, keeping its resolution and shrinking its size.

    ``quality`` is the only lever here: the pixel dimensions are untouched, so
    detection still sees full detail.
    """
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise PolygonError("The crop could not be encoded as JPEG")
    return buffer.tobytes()


def _write_temp_jpeg(image: np.ndarray, quality: int) -> tuple[Path, int]:
    """Spill the encoded crop to disk; the SDK uploads from a filesystem path.

    The handle is closed before returning because Windows will not let the SDK
    re-open a file that is still held open.
    """
    payload = encode_jpeg(image, quality)
    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        handle.write(payload)
    finally:
        handle.close()
    return Path(handle.name), len(payload)


def detect_crop(
    crop: np.ndarray,
    prompt: str,
    jpeg_quality: int = 75,
    timeout_seconds: float = 30.0,
) -> CropDetection:
    """Detect ``prompt`` in one crop. Never raises; the error is on the result."""
    try:
        client = shared_client()
        path, byte_count = _write_temp_jpeg(crop, jpeg_quality)
    except PolygonError as error:
        return CropDetection([], 0, str(error))
    except Exception as error:  # noqa: BLE001 - reported as a warning
        return CropDetection([], 0, f"{type(error).__name__}: {error}")

    try:
        predictions = _call_with_timeout(
            lambda: call_workflow(client, str(path), [prompt]), timeout_seconds
        )
        polygons = extract_polygons(predictions)
    except PolygonError as error:
        return CropDetection([], byte_count, str(error))
    except Exception as error:  # noqa: BLE001 - surface anything the SDK raises
        return CropDetection([], byte_count, f"{type(error).__name__}: {error}")
    finally:
        path.unlink(missing_ok=True)

    return CropDetection(polygons, byte_count)


# --- drawing -----------------------------------------------------------------


def polygon_area(ring: np.ndarray) -> float:
    """Enclosed area of a ring, in pixels squared, by the shoelace formula."""
    x = ring[:, 0].astype(np.float64)
    y = ring[:, 1].astype(np.float64)
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def color_for(label: str, assigned: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Stable colour per class name, so runs are comparable."""
    if label not in assigned:
        assigned[label] = _PALETTE[len(assigned) % len(_PALETTE)]
    return assigned[label]


def draw_polygons(image: np.ndarray, polygons: Sequence[Polygon]) -> np.ndarray:
    """Return a copy of ``image`` with each ring outlined, filled and labelled."""
    canvas = image.copy()
    overlay = image.copy()
    colors: dict[str, tuple[int, int, int]] = {}

    for polygon in polygons:
        color = color_for(polygon.class_name, colors)
        cv2.fillPoly(overlay, [polygon.ring], color)
        cv2.polylines(canvas, [polygon.ring], isClosed=True, color=color, thickness=2)

    # One blend for all fills, so overlapping masks do not stack opaquely.
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)

    for polygon in polygons:
        color = color_for(polygon.class_name, colors)
        label = f"{polygon.class_name} {polygon.confidence:.2f}"
        x, y = int(polygon.ring[:, 0].min()), int(polygon.ring[:, 1].min())
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, text_h + baseline + 2)
        cv2.rectangle(canvas, (x, y - text_h - baseline - 2), (x + text_w + 4, y), color, -1)
        cv2.putText(canvas, label, (x + 2, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# --- strip measurement -------------------------------------------------------


def analyze_detection(
    polygons: Sequence[Polygon],
    image_shape: Sequence[int],
    segment_count: int = 10,
    scale: "plane_scale.PlaneScale | None" = None,
) -> StripMeasurement | None:
    """Measure the adhesive strip traced by the largest polygon.

    The workflow can return several disjoint regions, and the strip is the
    biggest of them; anything smaller is treated as a stray match. Returns
    ``None`` when there is nothing measurable, including when the geometry is
    degenerate -- a measurement is never allowed to fail an inspection.

    ``scale`` restates the result in millimetres; without it the measurement
    stays in crop pixels.
    """
    if not polygons or segment_count < 1:
        return None

    index = max(range(len(polygons)), key=lambda i: polygon_area(polygons[i].ring))
    try:
        analysis = strip_analysis.analyze_ring(
            polygons[index].ring, image_shape, segment_count
        )
        if analysis is not None and scale is not None:
            analysis = strip_analysis.to_metric(analysis, scale)
    except Exception:  # noqa: BLE001 - reported as "no measurement", not an error
        return None
    if analysis is None:
        return None
    return StripMeasurement(polygon_index=index, analysis=analysis)


def draw_analysis(
    image: np.ndarray,
    polygons: Sequence[Polygon],
    measurement: StripMeasurement | None,
) -> np.ndarray:
    """Outline the polygons and, when one was measured, draw the strip on top."""
    canvas = draw_polygons(image, polygons)
    if measurement is not None:
        strip_analysis.draw_strip_analysis(canvas, measurement.analysis)
    return canvas


def _rounded(value: float | None) -> float | None:
    """Round a possibly-absent measurement, keeping absence as ``null``."""
    return None if value is None else round(float(value), 4)


def strip_record(measurement: StripMeasurement | None) -> dict[str, Any] | None:
    """Serialise a strip measurement for the result JSON, or ``None``.

    Pixel figures are kept alongside the millimetre ones: they are the
    measurement the scale was applied to, so they make a millimetre value
    auditable and let it be recomputed if the scale turns out to be wrong.
    """
    if measurement is None:
        return None
    analysis = measurement.analysis
    return {
        "polygon_index": int(measurement.polygon_index),
        "metric": analysis.metric,
        "total_length_px": _rounded(analysis.total_length_px),
        "average_width_px": _rounded(analysis.average_width_px),
        "minimum_width_px": _rounded(analysis.minimum_width_px),
        "maximum_width_px": _rounded(analysis.maximum_width_px),
        "total_length_mm": _rounded(analysis.total_length_mm),
        "average_width_mm": _rounded(analysis.average_width_mm),
        "minimum_width_mm": _rounded(analysis.minimum_width_mm),
        "maximum_width_mm": _rounded(analysis.maximum_width_mm),
        "segments": [
            {
                "index": int(segment.index),
                "length_px": _rounded(segment.length_px),
                "average_width_px": _rounded(segment.average_width_px),
                "minimum_width_px": _rounded(segment.minimum_width_px),
                "maximum_width_px": _rounded(segment.maximum_width_px),
                "length_mm": _rounded(segment.length_mm),
                "average_width_mm": _rounded(segment.average_width_mm),
                "minimum_width_mm": _rounded(segment.minimum_width_mm),
                "maximum_width_mm": _rounded(segment.maximum_width_mm),
                "samples": int(segment.samples),
                "midpoint": {
                    "x": round(float(segment.midpoint[0]), 3),
                    "y": round(float(segment.midpoint[1]), 3),
                },
            }
            for segment in analysis.segments
        ],
    }


# --- record ------------------------------------------------------------------


def detection_record(
    camera_number: int,
    crop_index: int,
    timestamp: str,
    size_profile: str,
    prompt: str,
    image_size: Sequence[int],
    detection: CropDetection,
    jpeg_quality: int,
    overlay_saved: bool,
    measurement: StripMeasurement | None = None,
) -> dict[str, Any]:
    """Build the JSON body written beside each detected crop.

    A record is written even when the call failed, so a dead API stays
    distinguishable from a frame in which nothing was found.
    """
    return {
        "camera": int(camera_number),
        "crop_index": int(crop_index),
        "timestamp": timestamp,
        "size_profile": size_profile,
        "prompt": prompt,
        "workflow": {"workspace": WORKSPACE_NAME, "id": WORKFLOW_ID},
        "image_size": [int(image_size[0]), int(image_size[1])],
        "upload": {
            "format": "jpeg",
            "quality": int(jpeg_quality),
            "bytes": int(detection.upload_bytes),
        },
        "overlay_saved": bool(overlay_saved),
        "polygons": [
            {
                "class": polygon.class_name,
                "confidence": round(float(polygon.confidence), 6),
                "area_px": round(polygon_area(polygon.ring), 3),
                "points": [
                    {"x": int(x), "y": int(y)} for x, y in polygon.ring
                ],
            }
            for polygon in detection.polygons
        ],
        "strip": strip_record(measurement),
        "error": detection.error,
    }
