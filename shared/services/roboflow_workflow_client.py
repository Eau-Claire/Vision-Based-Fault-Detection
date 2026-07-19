"""
Roboflow Workflow client for the hosted EVN object-detection workflow.

The workflow definition was read from Roboflow MCP:
- input: image (InferenceImage)
- parameters: none declared on the wrapper workflow
- output: predictions (JsonField)
"""

import base64
import binascii
import os
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests

from shared.schemas.analysis_result import BoundingBox, Detection, DetectionResult
from shared.services.class_mapping import map_class_to_category
from shared.utils.bbox import normalize_bbox
from shared.utils.logging import get_logger

logger = get_logger("roboflow_workflow")

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE_NAME = "les-workspace-ijdwd"
ROBOFLOW_WORKFLOW_ID = (
    "evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic"
)
ROBOFLOW_IMAGE_INPUT_NAME = "image"
ROBOFLOW_DECLARED_PARAMETERS: Dict[str, Any] = {}
ROBOFLOW_OUTPUT_NAMES = ("predictions",)


class RoboflowWorkflowError(Exception):
    """Base exception for Roboflow Workflow integration failures."""


class RoboflowConfigurationError(RoboflowWorkflowError):
    """Raised when required Roboflow configuration is missing or invalid."""


class RoboflowWorkflowTimeoutError(RoboflowWorkflowError):
    """Raised when the Roboflow workflow call exceeds the configured timeout."""


class RoboflowWorkflowResponseError(RoboflowWorkflowError):
    """Raised when Roboflow returns an unexpected response shape."""


@dataclass(frozen=True)
class RoboflowWorkflowRun:
    """Parsed response for one input image."""

    outputs: Mapping[str, Any]
    detections: Sequence[Detection]
    image_width: int = 0
    image_height: int = 0

    def as_detection_result(self) -> DetectionResult:
        return DetectionResult(
            detections=list(self.detections),
            image_width=self.image_width,
            image_height=self.image_height,
            frame_count=1,
        )


def run_evn_object_detection_workflow(
    image: Any,
    *,
    api_key: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[str] = None,
    api_url: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    timeout_seconds: int = 30,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    client: Any = None,
) -> List[RoboflowWorkflowRun]:
    """Run the EVN Roboflow Workflow for one image or a batch.

    Args:
        image: Image input accepted by inference-sdk: HTTPS URL, local path,
            base64 string, PIL image, numpy array, or a list for batching.
        api_key: Roboflow API key. Defaults to ROBOFLOW_API_KEY.
        parameters: Runtime parameters. The current wrapper workflow declares
            none, so unknown keys are rejected.
        output_dir: Optional directory for decoded image-shaped outputs.
        api_url: Roboflow API URL. Defaults to ROBOFLOW_API_URL.
        workspace_name: Roboflow workspace slug. Defaults to ROBOFLOW_WORKSPACE_NAME.
        workflow_id: Roboflow Workflow slug. Defaults to ROBOFLOW_WORKFLOW_ID.
        timeout_seconds: Per-attempt timeout.
        max_retries: Number of retries after the first failed attempt.
        retry_base_delay: Exponential backoff base delay in seconds.
        client: Optional prebuilt SDK-compatible client for tests.

    Returns:
        One parsed result per input image.
    """
    resolved_api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
    if client is None and not resolved_api_key:
        raise RoboflowConfigurationError("ROBOFLOW_API_KEY must be configured")

    request_parameters = _build_parameters(parameters)
    resolved_api_url = api_url or os.getenv("ROBOFLOW_API_URL", ROBOFLOW_API_URL)
    resolved_workspace_name = workspace_name or os.getenv(
        "ROBOFLOW_WORKSPACE_NAME", ROBOFLOW_WORKSPACE_NAME
    )
    resolved_workflow_id = workflow_id or os.getenv(
        "ROBOFLOW_WORKFLOW_ID", ROBOFLOW_WORKFLOW_ID
    )
    sdk_client = client or _make_client(resolved_api_key, api_url=resolved_api_url)

    raw_result = _run_with_retries(
        sdk_client=sdk_client,
        image=image,
        parameters=request_parameters,
        workspace_name=resolved_workspace_name,
        workflow_id=resolved_workflow_id,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    return parse_workflow_response(raw_result, output_dir=output_dir)


def parse_workflow_response(
    raw_result: Any,
    *,
    output_dir: Optional[str] = None,
) -> List[RoboflowWorkflowRun]:
    """Parse workflow responses from serverless or local inference runtimes."""
    entries = _workflow_response_entries(raw_result)

    parsed: List[RoboflowWorkflowRun] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RoboflowWorkflowResponseError(
                f"Expected response entry {index} to be an object"
            )

        outputs = _extract_declared_outputs(entry, index=index)
        safe_outputs = {
            name: _decode_image_output(value, output_dir, f"{index}_{name}")
            for name, value in outputs.items()
        }
        detections, image_width, image_height = _extract_detections(outputs)
        parsed.append(
            RoboflowWorkflowRun(
                outputs=safe_outputs,
                detections=detections,
                image_width=image_width,
                image_height=image_height,
            )
        )

    return parsed


def _workflow_response_entries(raw_result: Any) -> List[Any]:
    if isinstance(raw_result, list):
        return raw_result

    if isinstance(raw_result, Mapping):
        outputs = raw_result.get("outputs")
        if isinstance(outputs, list):
            return outputs
        if any(name in raw_result for name in ROBOFLOW_OUTPUT_NAMES):
            return [raw_result]

    raise RoboflowWorkflowResponseError(
        f"Expected workflow response list or output object, "
        f"got {type(raw_result).__name__}"
    )


def _make_client(api_key: str, api_url: str = ROBOFLOW_API_URL) -> Any:
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        logger.warning(
            "Roboflow inference-sdk import failed; falling back to REST workflow client",
            extra={
                "event": "roboflow_sdk_unavailable",
                "error_type": type(exc).__name__,
            },
        )
        return _RestWorkflowClient(api_key=api_key, api_url=api_url)

    return InferenceHTTPClient(api_url=api_url, api_key=api_key)


class _RestWorkflowClient:
    """Minimal REST client for Roboflow Workflows.

    This keeps production image inference independent from inference-sdk's
    OpenCV import path, which requires libGL in slim Docker images.
    """

    def __init__(self, api_key: str, api_url: str = ROBOFLOW_API_URL):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")

    def run_workflow(
        self,
        *,
        workspace_name: str,
        workflow_id: str,
        images: Mapping[str, Any],
        parameters: Mapping[str, Any],
        use_cache: bool = False,
    ) -> Any:
        del use_cache
        image = images.get(ROBOFLOW_IMAGE_INPUT_NAME)
        if image is None:
            raise RoboflowConfigurationError(
                f"Missing workflow image input: {ROBOFLOW_IMAGE_INPUT_NAME}"
            )

        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "inputs": {ROBOFLOW_IMAGE_INPUT_NAME: _rest_image_input(image)},
        }
        if parameters:
            payload["parameters"] = dict(parameters)

        url = f"{self.api_url}/{workspace_name}/workflows/{workflow_id}"
        response = requests.post(url, json=payload)
        if response.status_code >= 400:
            body = response.text[:500]
            raise RoboflowWorkflowError(
                f"Roboflow REST workflow failed: status={response.status_code}, "
                f"body={body}"
            )
        return response.json()


def _rest_image_input(image: Any) -> Dict[str, str]:
    if isinstance(image, str):
        if image.startswith("https://"):
            return {"type": "url", "value": image}
        if image.startswith("http://"):
            raise RoboflowConfigurationError("Roboflow URL inputs must use https")
        path = Path(image)
        if path.exists():
            return {"type": "base64", "value": _bytes_to_base64(path.read_bytes())}
        return {"type": "base64", "value": image}

    if isinstance(image, Path):
        return {"type": "base64", "value": _bytes_to_base64(image.read_bytes())}

    if isinstance(image, bytes):
        return {"type": "base64", "value": _bytes_to_base64(image)}

    if hasattr(image, "save"):
        buffer = BytesIO()
        image_to_save = image
        if getattr(image_to_save, "mode", "RGB") not in ("RGB", "L"):
            image_to_save = image_to_save.convert("RGB")
        image_to_save.save(buffer, format="JPEG")
        return {"type": "base64", "value": _bytes_to_base64(buffer.getvalue())}

    raise RoboflowConfigurationError(
        f"Unsupported REST workflow image input type: {type(image).__name__}"
    )


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _build_parameters(parameters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not parameters:
        return dict(ROBOFLOW_DECLARED_PARAMETERS)

    unknown = sorted(set(parameters) - set(ROBOFLOW_DECLARED_PARAMETERS))
    if unknown:
        raise RoboflowConfigurationError(
            "Unsupported Roboflow workflow parameter(s): " + ", ".join(unknown)
        )

    merged = dict(ROBOFLOW_DECLARED_PARAMETERS)
    merged.update(parameters)
    return merged


def _run_with_retries(
    *,
    sdk_client: Any,
    image: Any,
    parameters: Mapping[str, Any],
    workspace_name: str,
    workflow_id: str,
    timeout_seconds: int,
    max_retries: int,
    retry_base_delay: float,
) -> Any:
    attempts = max_retries + 1
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            return _run_once_with_timeout(
                sdk_client=sdk_client,
                image=image,
                parameters=parameters,
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                timeout_seconds=timeout_seconds,
            )
        except FutureTimeout as exc:
            last_error = exc
            error: RoboflowWorkflowError = RoboflowWorkflowTimeoutError(
                f"Roboflow workflow timed out after {timeout_seconds}s"
            )
        except Exception as exc:
            last_error = exc
            error = RoboflowWorkflowError(f"Roboflow workflow failed: {exc}")

        if attempt >= attempts:
            raise error from last_error

        sleep_seconds = retry_base_delay * (2 ** (attempt - 1))
        logger.warning(
            f"Roboflow workflow attempt {attempt} failed; retrying in "
            f"{sleep_seconds:.1f}s",
            extra={"event": "roboflow_workflow_retry"},
        )
        time.sleep(sleep_seconds)

    raise RoboflowWorkflowError("Roboflow workflow failed")


def _run_once_with_timeout(
    *,
    sdk_client: Any,
    image: Any,
    parameters: Mapping[str, Any],
    workspace_name: str,
    workflow_id: str,
    timeout_seconds: int,
) -> Any:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        sdk_client.run_workflow,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={ROBOFLOW_IMAGE_INPUT_NAME: image},
        parameters=dict(parameters),
        use_cache=False,
    )
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout:
        future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _extract_declared_outputs(entry: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    outputs = {name: entry[name] for name in ROBOFLOW_OUTPUT_NAMES if name in entry}
    if outputs:
        return outputs

    raise RoboflowWorkflowResponseError(
        f"Response entry {index} did not contain any declared output keys: "
        + ", ".join(ROBOFLOW_OUTPUT_NAMES)
    )


def _extract_detections(outputs: Mapping[str, Any]) -> tuple[List[Detection], int, int]:
    detections: List[Detection] = []
    image_width = 0
    image_height = 0

    for value in outputs.values():
        prediction_items, width, height = _find_prediction_items(value)
        image_width = image_width or width
        image_height = image_height or height

        for item in prediction_items:
            detection = _prediction_to_detection(item, image_width, image_height)
            if detection is not None:
                detections.append(detection)

    return detections, image_width, image_height


def _find_prediction_items(value: Any) -> tuple[List[Mapping[str, Any]], int, int]:
    if isinstance(value, Mapping):
        image = value.get("image")
        width = _int_from_mapping(image, "width")
        height = _int_from_mapping(image, "height")
        predictions = value.get("predictions")
        if isinstance(predictions, list):
            return [p for p in predictions if isinstance(p, Mapping)], width, height
        if _looks_like_prediction(value):
            return [value], width, height

    if isinstance(value, list):
        return [p for p in value if isinstance(p, Mapping)], 0, 0

    return [], 0, 0


def _prediction_to_detection(
    prediction: Mapping[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[Detection]:
    label = prediction.get("class") or prediction.get("class_name")
    if label is None:
        label = prediction.get("label") or prediction.get("name")
    if label is None:
        return None

    confidence = _float_or_none(
        prediction.get("confidence", prediction.get("score"))
    )
    if confidence is None:
        return None

    category = map_class_to_category(str(label))
    if category is None:
        return None

    bbox = _prediction_bbox(prediction, image_width, image_height)
    if bbox is None:
        return None

    return Detection(
        category_code=category,
        confidence=confidence,
        bounding_box=BoundingBox(**bbox),
    )


def _prediction_bbox(
    prediction: Mapping[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, float]]:
    if {"x", "y", "width", "height"}.issubset(prediction.keys()):
        x = _float_or_none(prediction.get("x"))
        y = _float_or_none(prediction.get("y"))
        width = _float_or_none(prediction.get("width"))
        height = _float_or_none(prediction.get("height"))
        if None in (x, y, width, height):
            return None
        if image_width > 0 and image_height > 0 and (width > 1 or height > 1):
            return normalize_bbox(
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
                image_width,
                image_height,
            )
        return {
            "x": max(0.0, min(1.0, x - width / 2)),
            "y": max(0.0, min(1.0, y - height / 2)),
            "width": max(0.0, min(1.0, width)),
            "height": max(0.0, min(1.0, height)),
        }

    if {"x_min", "y_min", "x_max", "y_max"}.issubset(prediction.keys()):
        x1 = _float_or_none(prediction.get("x_min"))
        y1 = _float_or_none(prediction.get("y_min"))
        x2 = _float_or_none(prediction.get("x_max"))
        y2 = _float_or_none(prediction.get("y_max"))
        if None in (x1, y1, x2, y2):
            return None
        return normalize_bbox(x1, y1, x2, y2, image_width, image_height)

    return None


def _decode_image_output(value: Any, output_dir: Optional[str], name: str) -> Any:
    if output_dir is None or not isinstance(value, str):
        return value

    image_bytes = _decode_base64_image(value)
    if image_bytes is None:
        return value

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ext = _image_extension(image_bytes)
    target = output_path / f"roboflow_{name}{ext}"
    target.write_bytes(image_bytes)
    return {"type": "file", "path": str(target)}


def _decode_base64_image(value: str) -> Optional[bytes]:
    payload = value.split(",", 1)[1] if value.startswith("data:image/") else value
    if len(payload) < 128:
        return None
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    if _image_extension(decoded) == ".bin":
        return None
    return decoded


def _image_extension(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _looks_like_prediction(value: Mapping[str, Any]) -> bool:
    has_label = any(key in value for key in ("class", "class_name", "label", "name"))
    has_confidence = any(key in value for key in ("confidence", "score"))
    return has_label and has_confidence


def _int_from_mapping(value: Any, key: str) -> int:
    if not isinstance(value, Mapping):
        return 0
    raw = value.get(key)
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ROBOFLOW_API_URL",
    "ROBOFLOW_WORKSPACE_NAME",
    "ROBOFLOW_WORKFLOW_ID",
    "ROBOFLOW_OUTPUT_NAMES",
    "RoboflowConfigurationError",
    "RoboflowWorkflowError",
    "RoboflowWorkflowResponseError",
    "RoboflowWorkflowRun",
    "RoboflowWorkflowTimeoutError",
    "parse_workflow_response",
    "run_evn_object_detection_workflow",
]
