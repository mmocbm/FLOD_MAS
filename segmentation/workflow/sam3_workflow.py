"""Client for the Roboflow workflow "SAM3 with Prompts".

Workflow (state as of 2026-09-23, read from its ``lastVersionConfig``):

    inputs:
      image   : InferenceImage
      prompts : WorkflowParameter, default ["person", "car"]   -> list[str]
    steps:
      sam                = roboflow_core/sam3@v3
                           images=$inputs.image, class_names=$inputs.prompts
      mask_visualization = roboflow_core/mask_visualization@v1
    outputs:
      sam                -> $steps.sam.predictions
      mask_visualization -> $steps.mask_visualization.image  (base64 PNG)

``prompts`` is the list of class names handed to the SAM3 step; it genuinely
drives the result (verified: ["dog"] yields dog detections, ["bicycle"] yields
none). Pass ``None`` to let the workflow apply its own default.

Typical use::

    from sam3_workflow import run_sam3_workflow

    result = run_sam3_workflow("photo.jpg", prompts=["glove", "defect"])
    for d in result.detections:
        print(d.class_name, round(d.confidence, 3))
    print("overlay:", result.mask_visualization_path)
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# --- Workflow identity -------------------------------------------------------

WORKSPACE_NAME = "obhashcolab-1"
WORKFLOW_ID = "sam3-with-prompts"
API_URL = "https://serverless.roboflow.com"

# Output names as declared by the workflow. Used only to prefer a known field
# when several candidates are present; parsing never *requires* them.
OUTPUT_PREDICTIONS = "sam"
OUTPUT_MASK_VISUALIZATION = "mask_visualization"

API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# A segmentation outline: an ordered ring of (x, y) pixel coordinates.
Point = tuple[float, float]
Polygon = tuple[Point, ...]


# --- Errors ------------------------------------------------------------------


class Sam3Error(Exception):
    """Base class for every failure raised by this module."""


class Sam3ConfigError(Sam3Error):
    """The client is misconfigured (missing API key, bad timeout, ...)."""


class Sam3InputError(Sam3Error):
    """The caller supplied an unusable image path, URL, or prompt list."""


class Sam3AuthError(Sam3Error):
    """Roboflow rejected the API key (HTTP 401/403)."""


class Sam3TimeoutError(Sam3Error):
    """The workflow did not finish inside the caller's deadline."""


class Sam3ResponseError(Sam3Error):
    """The workflow replied with a payload this module cannot interpret."""


# --- Result types ------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One predicted instance.

    ``rle_mask`` is populated only when ``include_masks=True``; its ``counts``
    field is multi-kilobyte and is dropped by default to keep result objects
    small.

    ``polygons`` is populated only when ``include_polygons=True``, sorted by
    area descending so ``polygons[0]`` (``main_polygon``) is the largest.
    """

    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    detection_id: str | None = None
    rle_mask: Mapping[str, Any] | None = None
    polygons: tuple[Polygon, ...] | None = None

    @property
    def main_polygon(self) -> Polygon | None:
        """The largest polygon, or ``None`` when polygons were not requested.

        ``polygons`` is sorted by area descending, so this is ``polygons[0]``.
        """
        return self.polygons[0] if self.polygons else None

    @property
    def polygon_points(self) -> list[list[float]]:
        """The largest polygon as plain ``[[x, y], ...]`` lists, JSON-ready."""
        main = self.main_polygon
        return [[px, py] for px, py in main] if main else []


@dataclass(frozen=True)
class Sam3Result:
    """Parsed result for a single input image."""

    detections: tuple[Detection, ...] = ()
    image_width: int | None = None
    image_height: int | None = None
    mask_visualization_path: Path | None = None
    output_keys: tuple[str, ...] = ()
    prompt_labels: tuple[str, ...] = ()

    def by_class(self) -> dict[str, list[Detection]]:
        """Group detections by class name."""
        grouped: dict[str, list[Detection]] = {}
        for det in self.detections:
            grouped.setdefault(det.class_name, []).append(det)
        return grouped

    def counts_by_class(self) -> dict[str, int]:
        """Number of detections per class name."""
        return {name: len(dets) for name, dets in self.by_class().items()}


# --- Configuration -----------------------------------------------------------


def load_api_key(env_var: str = API_KEY_ENV_VAR) -> str:
    """Return the Roboflow API key from the environment, falling back to .env.

    The key is never logged and never written anywhere.
    """
    key = os.environ.get(env_var, "").strip()
    if key:
        return key

    if _ENV_FILE.is_file():
        for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == env_var:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value

    raise Sam3ConfigError(
        f"{env_var} is not set. Export it, or put it in {_ENV_FILE.name} "
        f"next to this module (see .env.example)."
    )


def _build_client(api_key: str) -> Any:
    """Create the SDK client with header-based auth (inference v1.5.0+)."""
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.http.entities import InferenceConfiguration

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    return client.configure(InferenceConfiguration(api_key_transport="header"))


# --- Input handling ----------------------------------------------------------


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _normalize_image(image: str | os.PathLike[str] | Any) -> Any:
    """Return a value the SDK accepts as a workflow image input.

    URLs and non-path objects (numpy arrays, PIL images) pass through. Local
    paths are validated and returned as ``Path``.
    """
    if isinstance(image, (str, os.PathLike)):
        text = os.fspath(image)
        if _is_url(text):
            if text.startswith("http://"):
                # Roboflow rejects plain http image inputs.
                raise Sam3InputError(
                    f"Image URLs must use https://, got: {text!r}"
                )
            return text
        path = Path(text)
        if not path.is_file():
            raise Sam3InputError(f"Image file not found: {path}")
        # The SDK accepts a local path only as str; handing it a pathlib.Path
        # raises InvalidInputFormatError("Unknown type of input (WindowsPath)").
        return str(path)
    return image


def _normalize_prompts(prompts: Sequence[str] | None) -> list[str] | None:
    """Validate the ``prompts`` parameter (a list of class-name strings)."""
    if prompts is None:
        return None
    if isinstance(prompts, str):
        raise Sam3InputError(
            "prompts must be a list of class names, e.g. ['person', 'car'] "
            "— not a bare string."
        )
    cleaned = [str(p).strip() for p in prompts]
    if not cleaned or any(not p for p in cleaned):
        raise Sam3InputError(f"prompts must be non-empty strings, got: {prompts!r}")
    return cleaned


# --- Retry policy ------------------------------------------------------------


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code",):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after_of(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _is_sdk_client_error(exc: BaseException) -> bool:
    """True for the SDK's request-shaped errors (bad input, bad key, ...).

    ``RetryError`` is deliberately excluded: it derives from ``Exception``, not
    ``HTTPClientError``, and signals a transient failure worth retrying.
    """
    try:
        from inference_sdk.http.errors import HTTPClientError
    except ImportError:  # SDK absent; nothing to classify
        return False
    return isinstance(exc, HTTPClientError)


def _is_retryable(exc: BaseException) -> bool:
    """Timeouts, connection failures, 429 and 5xx are worth another attempt."""
    if isinstance(exc, Sam3TimeoutError):
        return True
    if isinstance(exc, (Sam3AuthError, Sam3InputError, Sam3ConfigError)):
        return False

    status = _status_of(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600

    name = type(exc).__name__
    if name in {
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError",
        "ChunkedEncodingError",
        "RetryError",
    }:
        return True

    # requests raises these from urllib3 / OSError when a connection dies.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    # Anything else the SDK raised is a caller-side problem: retrying would
    # just repeat the same failure and burn credits.
    return not _is_sdk_client_error(exc)


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with equal jitter (attempt is zero-based)."""
    window = min(cap, base * (2.0**attempt))
    return window / 2.0 + random.uniform(0.0, window / 2.0)


def _translate_error(exc: BaseException) -> Sam3Error:
    """Map a transport/SDK exception onto this module's error taxonomy."""
    if isinstance(exc, Sam3Error):
        return exc

    status = _status_of(exc)
    detail = getattr(exc, "api_message", None) or str(exc)

    if status in (401, 403):
        return Sam3AuthError(f"Roboflow rejected the API key (HTTP {status}): {detail}")
    if status is not None:
        return Sam3Error(f"Roboflow request failed (HTTP {status}): {detail}")
    if isinstance(exc, (TimeoutError,)):
        return Sam3TimeoutError(f"Workflow call timed out: {detail}")
    return Sam3Error(f"{type(exc).__name__}: {detail}")


# --- Transport ---------------------------------------------------------------


class _TimeoutGuard:
    """Enforce a wall-clock deadline around a blocking SDK call.

    The inference SDK exposes no request-timeout knob (verified against
    inference_sdk.config — only WebRTC timings are tunable), so the deadline is
    enforced by running the call on a worker thread and abandoning it when the
    budget expires.

    The worker is a plain daemon thread rather than a
    ``concurrent.futures.ThreadPoolExecutor``: executor threads are non-daemon
    since Python 3.9 and are joined at interpreter exit, so a wedged HTTP call
    would stall shutdown even with ``shutdown(wait=False)``. A daemon thread is
    simply reaped with the process.
    """

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        import threading

        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self._timeout)

        if worker.is_alive():
            raise Sam3TimeoutError(
                f"Workflow call exceeded the {self._timeout:g}s deadline"
            )
        if "error" in box:
            raise box["error"]
        return box.get("value")


def _invoke_workflow(
    client: Any,
    image: Any,
    prompts: list[str] | None,
    timeout: float,
) -> Any:
    """Run the workflow once and return the raw response payload."""
    parameters: dict[str, Any] = {}
    if prompts is not None:
        # Omitted entirely when None, so the workflow applies its own default.
        parameters["prompts"] = prompts

    guard = _TimeoutGuard(timeout)
    return guard.run(
        client.run_workflow,
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image},
        parameters=parameters,
        use_cache=False,
    )


# --- Response parsing --------------------------------------------------------


# Leading base64 magic for the image formats a workflow can return. Only
# distinctive prefixes are listed; anything unmatched is treated as not-an-image.
_B64_IMAGE_MAGIC: tuple[tuple[str, str], ...] = (
    ("/9j/", ".jpg"),  # JPEG  -> FF D8 FF
    ("iVBORw0KGgo", ".png"),  # PNG   -> 89 50 4E 47
    ("R0lGOD", ".gif"),  # GIF   -> 47 49 46
    ("UklGR", ".webp"),  # WebP  -> 52 49 46 46
)


def _image_suffix_for(b64_value: str) -> str | None:
    """Return the file suffix when base64 text starts with a known image magic."""
    for magic, suffix in _B64_IMAGE_MAGIC:
        if b64_value.startswith(magic):
            return suffix
    return None


def _extract_image_b64(value: Any) -> tuple[str, str] | None:
    """Return ``(base64_data, suffix)`` when ``value`` is an image output.

    The SDK hands image outputs back as a bare base64 string, whereas the raw
    REST endpoint wraps them as ``{"type": "base64", "value": ...}``. Both
    shapes are accepted, and the format is sniffed from the data rather than
    assumed.
    """
    candidates: list[str] = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, Mapping) and value.get("type") in {"base64", "url"}:
        inner = value.get("value")
        if isinstance(inner, str):
            candidates.append(inner)

    for candidate in candidates:
        suffix = _image_suffix_for(candidate)
        if suffix is not None:
            return candidate, suffix
    return None


def _looks_like_detection(value: Any) -> bool:
    return isinstance(value, Mapping) and "class" in value and "confidence" in value


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, falling back to ``default`` for nulls and garbage."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _require_supervision() -> Any:
    """Import ``supervision`` on demand, with an actionable error if absent."""
    try:
        import supervision
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise Sam3ConfigError(
            "Polygon extraction requires the 'supervision' package. Install it "
            "with `pip install supervision` — it normally arrives as a "
            "dependency of inference-sdk."
        ) from exc
    return supervision


def polygon_area(polygon: Polygon) -> float:
    """Shoelace area of a closed ring, in pixels.

    Useful for filtering out the small stray rings SAM sometimes emits:

        rings = [p for p in det.polygons if polygon_area(p) > 50]
    """
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _polygons_from_rle(
    rle: Mapping[str, Any] | None, simplify: float
) -> tuple[Polygon, ...]:
    """Decode a COCO RLE mask into its polygon outlines, largest first.

    The workflow returns masks as run-length encoding, not outlines. The mask is
    decoded, contoured, and released immediately: it is only needed transiently
    to derive the rings, and holding a full-frame boolean mask per detection
    would dwarf the rest of the result.
    """
    if not isinstance(rle, Mapping):
        return ()

    counts = rle.get("counts")
    size = rle.get("size")
    if not isinstance(counts, (str, bytes, list)):
        return ()
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        return ()

    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return ()
    if height <= 0 or width <= 0:
        return ()

    supervision = _require_supervision()
    try:
        mask = supervision.rle_to_mask(counts, (width, height))
        contours = supervision.mask_to_polygons(mask)
    except Exception as exc:  # noqa: BLE001 - one bad mask must not sink the call
        logger.warning("could not derive polygons from rle_mask: %s", exc)
        return ()

    polygons: list[Polygon] = []
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue  # a ring of <3 points encloses no area
        if simplify > 0:
            try:
                contour = supervision.approximate_polygon(
                    contour, percentage=simplify
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("polygon simplification failed: %s", exc)
            if len(contour) < 3:
                continue
        polygons.append(tuple((float(px), float(py)) for px, py in contour))

    # Largest first, so polygons[0] is the primary outline.
    polygons.sort(key=polygon_area, reverse=True)
    return tuple(polygons)


def _extract_detections(
    value: Any,
    include_masks: bool,
    include_polygons: bool = False,
    polygon_simplify: float = 0.0,
) -> list[Detection]:
    """Pull detections out of a prediction list or a response object."""
    candidates: Iterable[Any]
    if isinstance(value, Mapping):
        inner = value.get("predictions")
        candidates = inner if isinstance(inner, list) else []
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []

    detections: list[Detection] = []
    for item in candidates:
        if not _looks_like_detection(item):
            continue
        confidence = _as_float(item.get("confidence"), default=float("nan"))
        if not math.isfinite(confidence):
            continue

        raw_mask = item.get("rle_mask")
        mask = raw_mask if isinstance(raw_mask, Mapping) else None

        detections.append(
            Detection(
                class_name=str(item.get("class", "")),
                confidence=confidence,
                x=_as_float(item.get("x")),
                y=_as_float(item.get("y")),
                width=_as_float(item.get("width")),
                height=_as_float(item.get("height")),
                detection_id=item.get("detection_id"),
                rle_mask=mask if include_masks else None,
                polygons=(
                    _polygons_from_rle(mask, polygon_simplify)
                    if include_polygons
                    else None
                ),
            )
        )
    return detections


def _extract_image_size(entry: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Find the source image dimensions anywhere in the output entry."""
    for value in entry.values():
        if not isinstance(value, Mapping):
            continue
        size = value.get("image")
        if isinstance(size, Mapping):
            width = _as_float(size.get("width"), default=float("nan"))
            height = _as_float(size.get("height"), default=float("nan"))
            if math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0:
                return int(width), int(height)
    return None, None


def _write_base64_image(b64_value: str, destination: Path) -> Path:
    """Decode a base64 image straight to disk, never through a log or a str."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = base64.b64decode(b64_value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise Sam3ResponseError(f"Workflow returned undecodable base64: {exc}") from exc
    destination.write_bytes(raw)
    return destination


@dataclass(frozen=True)
class _ParseOptions:
    """Internal knobs threaded through the response parser."""

    include_masks: bool = False
    include_polygons: bool = False
    polygon_simplify: float = 0.0
    output_dir: Path | None = None
    stem: str = "output"
    prompt_labels: tuple[str, ...] = ()


def _parse_single_entry(entry: Mapping[str, Any], options: _ParseOptions) -> Sam3Result:
    detections: list[Detection] = []
    image_path: Path | None = None
    image_width, image_height = _extract_image_size(entry)

    # Prefer the workflow's declared prediction output; otherwise take whatever
    # looks like predictions.
    ordered_keys = sorted(
        entry, key=lambda k: (k != OUTPUT_PREDICTIONS, k != OUTPUT_MASK_VISUALIZATION, k)
    )

    for key in ordered_keys:
        value = entry[key]

        if image_path is None and options.output_dir is not None:
            image = _extract_image_b64(value)
            if image is not None:
                data, suffix = image
                image_path = _write_base64_image(
                    data, options.output_dir / f"{options.stem}_{key}{suffix}"
                )
                continue

        if not detections:
            found = _extract_detections(
                value,
                options.include_masks,
                options.include_polygons,
                options.polygon_simplify,
            )
            if found:
                detections = found

    return Sam3Result(
        detections=tuple(detections),
        image_width=image_width,
        image_height=image_height,
        mask_visualization_path=image_path,
        output_keys=tuple(entry.keys()),
        prompt_labels=options.prompt_labels,
    )


def _parse_response(payload: Any, options: _ParseOptions) -> Sam3Result:
    """Defensively parse a workflow response into a :class:`Sam3Result`."""
    if isinstance(payload, Mapping):
        entries = payload.get("outputs")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise Sam3ResponseError(
            f"Expected a dict or list from the workflow, got {type(payload).__name__}"
        )

    if not isinstance(entries, list):
        raise Sam3ResponseError(
            f"Workflow response has no usable 'outputs' list: {type(entries).__name__}"
        )
    if not entries:
        # One entry per input image; none in means nothing ran.
        return Sam3Result(prompt_labels=options.prompt_labels)

    first = entries[0]
    if not isinstance(first, Mapping):
        raise Sam3ResponseError(
            f"Workflow output entry is {type(first).__name__}, expected a mapping"
        )

    return _parse_single_entry(first, options)


# --- Public API --------------------------------------------------------------


def run_sam3_workflow(
    image: str | os.PathLike[str] | Any,
    *,
    prompts: Sequence[str] | None = None,
    include_masks: bool = False,
    include_polygons: bool = False,
    polygon_simplify: float = 0.0,
    output_dir: str | os.PathLike[str] | None = "outputs",
    timeout: float = 120.0,
    max_retries: int = 3,
    api_key: str | None = None,
    client: Any = None,
) -> Sam3Result:
    """Run the "SAM3 with Prompts" workflow on one image.

    Args:
        image: An https URL, a local file path, or anything the inference SDK
            accepts as an image (numpy array, PIL image).
        prompts: Class names to segment, e.g. ``["person", "car"]``. ``None``
            (default) lets the workflow use its own default
            (``["person", "car"]``).
        include_masks: Keep the raw ``rle_mask`` on each detection. Off by
            default because the RLE ``counts`` strings are multi-kilobyte.
        include_polygons: Derive segmentation outlines from each mask and
            attach them as ``Detection.polygons``. Requires the ``supervision``
            package (installed with inference-sdk). Off by default because
            decoding a full-frame mask per detection is the expensive part of
            the call.
        polygon_simplify: How aggressively to simplify outlines, ``0.0`` (keep
            every contour point) to ``~0.99`` (a few points). SAM outlines are
            typically 500-900 points; ``0.5`` roughly halves them. Ignored
            unless ``include_polygons`` is set.
        output_dir: Where to write the workflow's base64 image output(s).
            ``None`` disables writing and skips decoding entirely.
        timeout: Per-attempt wall-clock deadline in seconds.
        max_retries: Number of *additional* attempts after the first failure.
        api_key: Override the ``ROBOFLOW_API_KEY`` environment variable.
        client: Pre-built SDK client, to reuse one across calls in a batch.

    Returns:
        A :class:`Sam3Result` for the first (and only) input image.

    Raises:
        Sam3ConfigError: Missing API key or an invalid argument.
        Sam3InputError: Unusable image path/URL, or malformed ``prompts``.
        Sam3AuthError: Roboflow rejected the key.
        Sam3TimeoutError: Every attempt exceeded ``timeout``.
        Sam3ResponseError: The reply could not be interpreted.
        Sam3Error: Any other transport failure, after retries are exhausted.
    """
    if timeout <= 0:
        raise Sam3ConfigError(f"timeout must be positive, got {timeout!r}")
    if max_retries < 0:
        raise Sam3ConfigError(f"max_retries must be >= 0, got {max_retries!r}")

    normalized_image = _normalize_image(image)
    normalized_prompts = _normalize_prompts(prompts)

    if client is None:
        client = _build_client(api_key or load_api_key())

    if not 0.0 <= polygon_simplify < 1.0:
        raise Sam3ConfigError(
            f"polygon_simplify must be in [0, 1), got {polygon_simplify!r}"
        )

    options = _ParseOptions(
        include_masks=include_masks,
        include_polygons=include_polygons,
        polygon_simplify=polygon_simplify,
        output_dir=Path(output_dir) if output_dir is not None else None,
        stem=_stem_for(normalized_image),
        prompt_labels=tuple(normalized_prompts or ()),
    )

    attempts = max_retries + 1
    last_error: BaseException | None = None

    for attempt in range(attempts):
        try:
            payload = _invoke_workflow(
                client, normalized_image, normalized_prompts, timeout
            )
        except BaseException as exc:  # noqa: BLE001 - classified below
            error = _translate_error(exc)
            if attempt + 1 >= attempts or not _is_retryable(error):
                raise error from exc

            delay = _backoff_delay(attempt, base=1.0, cap=30.0)
            retry_after = _retry_after_of(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            logger.warning(
                "workflow attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                attempts,
                error,
                delay,
            )
            last_error = error
            time.sleep(delay)
            continue

        return _parse_response(payload, options)

    # Unreachable: the loop either returns or raises.
    raise _translate_error(last_error or RuntimeError("workflow failed"))


def _stem_for(image: Any) -> str:
    """A short, filesystem-safe stem for naming written outputs."""
    if isinstance(image, (str, os.PathLike)):
        text = os.fspath(image)
        if _is_url(text):
            name = text.rsplit("/", 1)[-1].split("?")[0]
            return Path(name).stem or "output"
        return Path(text).stem or "output"
    return "output"


# --- Convenience -------------------------------------------------------------


def count_detections(image: str | os.PathLike[str], **kwargs: Any) -> dict[str, int]:
    """Run the workflow and return ``{class_name: count}``."""
    return run_sam3_workflow(image, **kwargs).counts_by_class()
