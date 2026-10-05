"""Client for the Roboflow workflow "custom-workflow" — one image in, JSON out.

The workflow matches twelve boxed panel exemplars against the target image,
segments the adhesive strip inside each matched panel, stitches the strips back
into the full-frame coordinate space, and draws them. Its declared outputs
(verified against a live call on 2026-10-02) are:

    output_image    base64 JPEG of the input with the strips drawn
    bounding_boxes  [[x1, y1, x2, y2], ...] — the matched panel boxes
    predictions     {"image": {"width", "height"}, "predictions": [...]}
                    each prediction carrying a ``points`` ring of strip vertices

A run that matches no panels comes back with ``output_image: None``,
``bounding_boxes: []`` and ``predictions: None``. That is a legitimate empty
answer, not an error, so it parses to an empty :class:`PanelResult`.

Transport, authentication and retry handling are shared with the sibling
:mod:`sam3_workflow` module rather than duplicated; the leading-underscore
names imported from it are that module's internals, reused here deliberately.

Typical use::

    from adhesive_strip_workflow import run_workflow

    result = run_workflow("frame.jpg")
    print(result.count, "strip(s)")
    for strip in result.strips:
        print(strip.class_name, round(strip.confidence, 3), len(strip.points), "points")
    print("annotated:", result.output_image_path)

    payload = run_to_json("frame.jpg", json_path="outputs/frame.json")

Command line::

    .venv\\Scripts\\python.exe adhesive_strip_workflow.py frame.jpg
    .venv\\Scripts\\python.exe adhesive_strip_workflow.py frame.jpg --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# --- Workflow identity -------------------------------------------------------

WORKSPACE_NAME = "obhashcolab-1"
WORKFLOW_ID = "custom-workflow"

# Input / output names exactly as the workflow declares them.
INPUT_IMAGE = "image"
OUTPUT_IMAGE = "output_image"
OUTPUT_BOXES = "bounding_boxes"
OUTPUT_PREDICTIONS = "predictions"

# See the module docstring: the transport layer is shared with sam3_workflow.
from .sam3_workflow import (  # noqa: E402 - kept below the identity constants
    API_URL,
    Sam3AuthError,
    Sam3ConfigError,
    Sam3Error,
    Sam3InputError,
    Sam3ResponseError,
    Sam3TimeoutError,
    _as_float,
    _backoff_delay,
    _build_client,
    _extract_image_b64,
    _is_retryable,
    _normalize_image,
    _retry_after_of,
    _stem_for,
    _TimeoutGuard,
    _translate_error,
    _write_base64_image,
    load_api_key,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TIMEOUT = 180.0  # panel matching plus two SAM3 passes; 3 min is generous
DEFAULT_MAX_RETRIES = 3

# The error family lives in sam3_workflow but describes this call just as well,
# so it is re-exported under a name that reads correctly from here.
PanelError = Sam3Error

# The Sam3* error classes and API_URL are re-exported so a caller can catch the
# whole failure family, and name the endpoint, without a second import.
__all__ = [
    "API_URL",
    "WORKSPACE_NAME",
    "WORKFLOW_ID",
    "PanelError",
    "PanelResult",
    "Sam3AuthError",
    "Sam3ConfigError",
    "Sam3Error",
    "Sam3InputError",
    "Sam3ResponseError",
    "Sam3TimeoutError",
    "Strip",
    "build_client",
    "load_api_key",
    "run_to_json",
    "run_workflow",
    "write_json",
]


# --- Result types ------------------------------------------------------------


@dataclass(frozen=True)
class Strip:
    """One segmented adhesive strip.

    ``box`` is the API's centre form ``(x, y, width, height)``; ``points`` is the
    outline ring in source-image pixels. As with the sibling SAM3 module, one
    physically continuous strip can arrive as several predictions.
    """

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    detection_id: str | None = None

    @property
    def polygon(self) -> list[list[float]]:
        """The ring as plain ``[[x, y], ...]`` lists, JSON-ready."""
        return [[x, y] for x, y in self.points]


@dataclass(frozen=True)
class PanelResult:
    """Parsed result for a single input image.

    ``raw`` holds the workflow's own output entry, so nothing the API returned
    is lost to parsing; :meth:`to_json` republishes it verbatim.
    """

    strips: tuple[Strip, ...] = ()
    panel_boxes: tuple[tuple[int, int, int, int], ...] = ()
    image_width: int | None = None
    image_height: int | None = None
    output_image_path: Path | None = None
    output_image_base64: str | None = None
    prompt_labels: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """How many strips were segmented."""
        return len(self.strips)

    @property
    def is_empty(self) -> bool:
        """True when no panel was matched, or no strip survived filtering."""
        return not self.strips and not self.panel_boxes

    def to_json(self, *, include_image_base64: bool = True) -> dict[str, Any]:
        """The workflow's response as JSON-serializable data.

        Key-for-key the same shape the REST endpoint returns, so anything
        already written against that keeps working. The image is always the
        ``{"type": ..., "value": ...}`` wrapper the REST
        endpoint uses, whichever shape the SDK handed back.

        Args:
            include_image_base64: When True (default) ``output_image`` carries
                the base64 JPEG. When False it carries the saved file's path
                instead — the same JSON without a ~190 KB string in it. A run
                with no matched panels reports ``None`` either way.
        """
        entry = self.raw
        if include_image_base64 and self.output_image_base64 is not None:
            image: Any = {"type": "base64", "value": self.output_image_base64}
        elif self.output_image_path is not None:
            image = {"type": "file", "value": str(self.output_image_path)}
        else:
            image = None
        return {
            OUTPUT_IMAGE: image,
            OUTPUT_BOXES: [list(box) for box in self.panel_boxes],
            OUTPUT_PREDICTIONS: entry.get(OUTPUT_PREDICTIONS),
        }


# --- Configuration -----------------------------------------------------------


def build_client(api_key: str) -> Any:
    """SDK client that sends the key as a bearer header, never in the URL."""
    return _build_client(api_key)


# --- The API call ------------------------------------------------------------


def _invoke_workflow(client: Any, image: Any, timeout: float) -> Any:
    """Run the workflow once on one image and return the raw payload."""
    guard = _TimeoutGuard(timeout)
    return guard.run(
        client.run_workflow,
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={INPUT_IMAGE: image},
        use_cache=False,
    )


# --- Response parsing --------------------------------------------------------


def _parse_panel_boxes(value: Any) -> tuple[tuple[int, int, int, int], ...]:
    """Read the ``bounding_boxes`` output as a tuple of ``(x1, y1, x2, y2)``.

    The workflow emits plain 4-number lists. A box given in the centre form
    (``x``/``y``/``width``/``height``) is accepted too, so the parser survives a
    change of shape on the workflow side.
    """
    if not isinstance(value, list):
        return ()

    boxes: list[tuple[int, int, int, int]] = []
    for item in value:
        numbers: Sequence[Any] | None = None
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            numbers = item[:4]
        elif isinstance(item, Mapping) and "width" in item and "height" in item:
            x = _as_float(item.get("x"))
            y = _as_float(item.get("y"))
            width = _as_float(item.get("width"))
            height = _as_float(item.get("height"))
            numbers = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        if numbers is None:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(n))) for n in numbers)
        except (TypeError, ValueError):
            continue
        boxes.append((x1, y1, x2, y2))
    return tuple(boxes)


def _parse_point_ring(raw: Any) -> tuple[tuple[float, float], ...]:
    """Pull a polygon's vertices out of ``points``, skipping malformed ones."""
    if not isinstance(raw, list):
        return ()
    vertices: list[tuple[float, float]] = []
    for point in raw:
        if isinstance(point, Mapping):
            x, y = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        vertices.append((x, y))
    return tuple(vertices)


def _parse_strips(value: Any) -> tuple[Strip, ...]:
    """Read the ``predictions`` output into typed :class:`Strip` objects."""
    if not isinstance(value, Mapping):
        return ()
    predictions = value.get("predictions")
    if not isinstance(predictions, list):
        return ()

    strips: list[Strip] = []
    for item in predictions:
        if not isinstance(item, Mapping):
            continue
        confidence = _as_float(item.get("confidence"), default=float("nan"))
        if confidence != confidence:  # NaN: no usable confidence, skip the row
            continue
        points = _parse_point_ring(item.get("points"))
        if len(points) < 3:
            continue  # a ring of fewer than 3 vertices encloses no area
        strips.append(
            Strip(
                class_name=str(item.get("class", "")),
                confidence=confidence,
                box=(
                    _as_float(item.get("x")),
                    _as_float(item.get("y")),
                    _as_float(item.get("width")),
                    _as_float(item.get("height")),
                ),
                points=points,
                detection_id=item.get("detection_id"),
            )
        )
    return tuple(strips)


def _parse_image_size(value: Any) -> tuple[int | None, int | None]:
    """Read ``predictions.image.{width,height}`` — the source image's size."""
    if not isinstance(value, Mapping):
        return None, None
    size = value.get("image")
    if not isinstance(size, Mapping):
        return None, None
    try:
        width, height = int(size["width"]), int(size["height"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _parse_response(payload: Any, output_dir: Path | None, stem: str) -> PanelResult:
    """Defensively parse a workflow response into a :class:`PanelResult`."""
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
        return PanelResult()

    entry = entries[0]
    if not isinstance(entry, Mapping):
        raise Sam3ResponseError(
            f"Workflow output entry is {type(entry).__name__}, expected a mapping"
        )

    # Decode the annotated image first, so a later parse hiccup cannot cost the
    # caller the only copy of the drawing. The base64 is kept even when no file
    # is wanted, so to_json() can still emit it.
    image_path: Path | None = None
    image_b64: str | None = None
    image = _extract_image_b64(entry.get(OUTPUT_IMAGE))
    if image is not None:
        data, suffix = image
        image_b64 = data
        if output_dir is not None:
            image_path = _write_base64_image(data, output_dir / f"{stem}_adhesive{suffix}")

    width, height = _parse_image_size(entry.get(OUTPUT_PREDICTIONS))
    return PanelResult(
        strips=_parse_strips(entry.get(OUTPUT_PREDICTIONS)),
        panel_boxes=_parse_panel_boxes(entry.get(OUTPUT_BOXES)),
        image_width=width,
        image_height=height,
        output_image_path=image_path,
        output_image_base64=image_b64,
        raw=entry,
    )


# --- Public API --------------------------------------------------------------


def run_workflow(
    image: str | os.PathLike[str] | Any,
    *,
    output_dir: str | os.PathLike[str] | None = DEFAULT_OUTPUT_DIR,
    save_name: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: str | None = None,
    client: Any = None,
) -> PanelResult:
    """Run the "custom-workflow" panel/adhesive-strip workflow on one image.

    Args:
        image: An https URL, a local file path, or anything the inference SDK
            accepts as an image (numpy array, PIL image).
        output_dir: Where to write the workflow's base64 annotated image.
            ``None`` disables writing and skips decoding it.
        save_name: Filename stem for that image; defaults to the input's own
            stem, giving ``outputs/<stem>_adhesive.jpg``.
        timeout: Per-attempt wall-clock deadline in seconds.
        max_retries: Number of *additional* attempts after the first failure.
        api_key: Override the ``ROBOFLOW_API_KEY`` environment variable.
        client: Pre-built SDK client, to reuse one across calls in a batch.

    Returns:
        A :class:`PanelResult`. An image with no matching panels yields an empty
        result rather than raising — check :attr:`PanelResult.is_empty`.

    Raises:
        Sam3ConfigError: Missing API key or an invalid argument.
        Sam3InputError: Unusable image path or URL.
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
    stem = save_name or _stem_for(normalized_image)
    destination = Path(output_dir) if output_dir is not None else None

    if client is None:
        client = build_client(api_key or load_api_key())

    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            payload = _invoke_workflow(client, normalized_image, timeout)
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
            time.sleep(delay)
            continue

        return _parse_response(payload, destination, stem)

    # Unreachable: the loop either returns or raises.
    raise Sam3Error("workflow failed with no result and no error")


def run_to_json(
    image: str | os.PathLike[str] | Any,
    *,
    json_path: str | os.PathLike[str] | None = None,
    include_image_base64: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the workflow and return its JSON, optionally writing it to a file.

    Any remaining keyword arguments go to :func:`run_workflow` unchanged.
    """
    result = run_workflow(image, **kwargs)
    payload = result.to_json(include_image_base64=include_image_base64)
    if json_path is not None:
        write_json(json_path, payload)
    return payload


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return destination


# --- Command line ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the panel/adhesive-strip Roboflow workflow on one image."
    )
    parser.add_argument("image", help="image path or https URL")
    parser.add_argument("--json", dest="json_path", help="also write the JSON here")
    parser.add_argument(
        "--no-image-base64",
        action="store_true",
        help="put the saved image's path in the JSON instead of the base64 blob",
    )
    parser.add_argument("--outputs", default=DEFAULT_OUTPUT_DIR, help="where to write the image")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = run_workflow(
            args.image,
            output_dir=args.outputs,
            timeout=args.timeout,
            max_retries=args.retries,
        )
    except Sam3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.is_empty:
        print("No panels matched — the workflow returned an empty result.")
    else:
        print(f"{len(result.panel_boxes)} panel(s), {result.count} strip(s)")
        for index, strip in enumerate(result.strips, start=1):
            x, y, width, height = strip.box
            print(
                f"  {index:>2}  conf {strip.confidence:.3f}  "
                f"{len(strip.points):>4} pts  centre ({x:.0f},{y:.0f}) {width:.0f}x{height:.0f}"
            )
    if result.output_image_path is not None:
        print(f"annotated image: {result.output_image_path}")

    payload = result.to_json(include_image_base64=not args.no_image_base64)
    if args.json_path:
        print(f"json: {write_json(args.json_path, payload)}")
    else:
        # Piping the ~190 KB blob to a terminal helps nobody; the file is the
        # real deliverable, and --json is how you get the whole payload.
        print(json.dumps(_elide_image(payload)))
    return 0


def _elide_image(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the payload with the base64 blob replaced by its length."""
    trimmed = dict(payload)
    image = trimmed.get(OUTPUT_IMAGE)
    if isinstance(image, Mapping) and image.get("type") == "base64":
        value = str(image.get("value", ""))
        trimmed[OUTPUT_IMAGE] = {"type": "base64", "value": f"<{len(value)} chars elided>"}
    return trimmed


if __name__ == "__main__":
    raise SystemExit(main())
