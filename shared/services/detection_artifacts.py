"""
Detection artifact helpers for video analysis.

Creates full-frame preview images and cropped detection images for frontend
inspection workflows, then attaches their public URLs to Detection objects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import cv2

from shared.schemas.analysis_result import DetectionResult
from shared.utils.logging import get_logger

logger = get_logger("detection_artifacts")

_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


def enrich_video_detection_artifacts(
    video_path: str,
    detection_result: DetectionResult,
    request_id: str,
    artifact_dir: str,
    public_base_url: Optional[str],
    artifact_url_path: str = "/artifacts",
    jpeg_quality: int = 90,
) -> DetectionResult:
    """Generate frame and crop JPEGs for video detections.

    The function is best-effort: if a frame cannot be decoded or an image cannot
    be written, detection metadata remains intact and processing continues.
    """
    if not detection_result.detections:
        return detection_result

    base_url = _resolve_public_base_url(public_base_url)
    if not base_url:
        logger.warning(
            "Detection artifact URLs disabled because public base URL is empty",
            extra={"event": "artifact_url_disabled"},
        )
        return detection_result

    safe_request_id = _safe_segment(request_id)
    output_dir = Path(artifact_dir) / safe_request_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(
            f"Cannot open video for artifact generation: {video_path}",
            extra={"event": "artifact_video_open_failed"},
        )
        return detection_result

    frames_by_index: dict[int, object] = {}
    try:
        for detection_index, detection in enumerate(detection_result.detections):
            if detection.frame_index is None:
                continue

            frame = frames_by_index.get(detection.frame_index)
            if frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, detection.frame_index)
                ok, decoded = cap.read()
                if not ok or decoded is None:
                    logger.warning(
                        f"Cannot decode detection frame {detection.frame_index}",
                        extra={"event": "artifact_frame_decode_failed"},
                    )
                    continue
                frame = decoded
                frames_by_index[detection.frame_index] = frame

            frame_filename = f"frame-{detection.frame_index}.jpg"
            frame_path = output_dir / frame_filename
            if _write_jpeg_once(frame_path, frame, jpeg_quality):
                detection.image_url = _build_artifact_url(
                    base_url,
                    artifact_url_path,
                    safe_request_id,
                    frame_filename,
                )

            crop = _crop_detection(frame, detection.bounding_box)
            if crop is None:
                continue

            detection_id = _safe_segment(detection.id or str(detection_index))
            crop_filename = f"crop-{detection.frame_index}-{detection_id}.jpg"
            crop_path = output_dir / crop_filename
            if _write_jpeg_once(crop_path, crop, jpeg_quality):
                detection.crop_url = _build_artifact_url(
                    base_url,
                    artifact_url_path,
                    safe_request_id,
                    crop_filename,
                )
    finally:
        cap.release()

    return detection_result


def _crop_detection(frame, bounding_box):
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(round(bounding_box.x * w))))
    y1 = max(0, min(h, int(round(bounding_box.y * h))))
    x2 = max(0, min(w, int(round((bounding_box.x + bounding_box.width) * w))))
    y2 = max(0, min(h, int(round((bounding_box.y + bounding_box.height) * h))))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _write_jpeg_once(path: Path, image, jpeg_quality: int) -> bool:
    if path.exists():
        return True
    return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]))


def _resolve_public_base_url(public_base_url: Optional[str]) -> str:
    return (public_base_url or "").strip().rstrip("/")


def _build_artifact_url(
    public_base_url: str,
    artifact_url_path: str,
    request_id: str,
    filename: str,
) -> str:
    path = "/" + artifact_url_path.strip("/")
    return (
        f"{public_base_url}{path}/"
        f"{quote(request_id, safe='')}/{quote(filename, safe='')}"
    )


def _safe_segment(value: str) -> str:
    cleaned = _SAFE_ID_PATTERN.sub("-", str(value).strip())
    return cleaned.strip(".-") or "artifact"
