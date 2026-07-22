"""
EdgeYoloDetector — YOLO11 inference for Raspberry Pi / edge devices.

Implements the shared Detector protocol. Optimized for CPU inference,
low memory usage, and near-real-time processing.
"""

import os
import cv2
import numpy as np
from typing import List

from ultralytics import YOLO

from shared.schemas.analysis_result import (
    BoundingBox,
    Detection,
    DetectionResult,
)
from shared.services.class_mapping import map_class_to_category
from shared.utils.bbox import normalize_bbox
from shared.utils.logging import get_logger

logger = get_logger("edge_detector")


class EdgeYoloDetector:
    """YOLO11 detector optimized for edge/ARM64 deployment.

    Implements the Detector protocol defined in
    shared.schemas.detector_interface.
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        frame_sample_interval: int = 15,
        max_frames_per_video: int = 100,
        dedup_iou_threshold: float = 0.5,
        max_detections_per_class: int = 5,
    ):
        """Initialize the YOLO11 detector.

        Args:
            model_path: Path to the YOLO11 weights file (.pt).
            image_size: Input image size for inference.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            device: Inference device ('cpu').
            frame_sample_interval: Process every Nth frame for video.
            max_frames_per_video: Max frames to process per video.
            dedup_iou_threshold: IoU threshold for deduplication.
            max_detections_per_class: Max detections per class in video.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO11 model not found: {model_path}"
            )

        logger.info(
            f"Loading YOLO11 model from: {model_path}",
            extra={"event": "model_load_start", "model": "YOLO11"},
        )

        self._model = YOLO(model_path)
        self._image_size = image_size
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._device = device
        self._frame_sample_interval = frame_sample_interval
        self._max_frames_per_video = max_frames_per_video
        self._dedup_iou_threshold = dedup_iou_threshold
        self._max_detections_per_class = max_detections_per_class

        # Extract model metadata
        self._model_name = "YOLO11"
        self._model_version = "1.0.0"

        logger.info(
            "YOLO11 model loaded successfully",
            extra={"event": "model_load_complete", "model": "YOLO11"},
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Run YOLO11 inference on a single image.

        Args:
            image: OpenCV BGR image as numpy array.

        Returns:
            DetectionResult with normalized bounding boxes.
        """
        h, w = image.shape[:2]
        detections: List[Detection] = []

        results = self._model(
            image,
            imgsz=self._image_size,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
        )

        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]
                conf = float(box.conf[0])

                # Filter by confidence
                if conf < self._conf_threshold:
                    continue

                # Map class to backend category
                category = map_class_to_category(cls_name)
                if category is None:
                    continue

                bbox = normalize_bbox(x1, y1, x2, y2, w, h)
                detections.append(
                    Detection(
                        category_code=category,
                        confidence=conf,
                        bounding_box=BoundingBox(**bbox),
                    )
                )

        logger.info(
            f"Image inference: {len(detections)} detections "
            f"({w}x{h})",
            extra={"event": "inference_image_complete"},
        )

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            frame_count=1,
        )

    def detect_video(self, video_path: str) -> DetectionResult:
        """Run YOLO11 inference on a video file with frame sampling.

        Args:
            video_path: Path to the video file.

        Returns:
            DetectionResult with deduplicated detections across frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(total_frames / fps, 3) if fps and total_frames else None

        all_detections: List[Detection] = []
        frame_idx = 0
        processed_count = 0

        try:
            while processed_count < self._max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self._frame_sample_interval == 0:
                    timestamp_ms = int((frame_idx / fps) * 1000)

                    results = self._model(
                        frame,
                        imgsz=self._image_size,
                        conf=self._conf_threshold,
                        iou=self._iou_threshold,
                        device=self._device,
                        verbose=False,
                    )

                    if results and len(results) > 0:
                        result = results[0]
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(float, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            cls_name = self._model.names[cls_id]
                            conf = float(box.conf[0])

                            if conf < self._conf_threshold:
                                continue

                            category = map_class_to_category(cls_name)
                            if category is None:
                                continue

                            bbox = normalize_bbox(x1, y1, x2, y2, w, h)
                            all_detections.append(
                                Detection(
                                    category_code=category,
                                    confidence=conf,
                                    bounding_box=BoundingBox(**bbox),
                                    timestamp_ms=timestamp_ms,
                                    frame_index=frame_idx,
                                )
                            )

                    processed_count += 1

                frame_idx += 1
        finally:
            cap.release()

        # Deduplicate nearby detections
        deduped = self._deduplicate(all_detections)

        logger.info(
            f"Video inference: {len(deduped)} detections "
            f"(from {processed_count} frames, {frame_idx} total)",
            extra={"event": "inference_video_complete"},
        )

        return DetectionResult(
            detections=deduped,
            image_width=w,
            image_height=h,
            frame_count=processed_count,
            fps=round(fps, 3),
            duration=duration,
        )

    def _deduplicate(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Deduplicate detections by category and spatial overlap.

        Groups by categoryCode, keeps the highest-confidence detections
        that don't overlap significantly with already-kept ones.
        """
        from shared.utils.bbox import calculate_iou

        # Group by category
        grouped: dict = {}
        for det in detections:
            grouped.setdefault(det.category_code, []).append(det)

        result: List[Detection] = []

        for category, items in grouped.items():
            # Sort by confidence descending
            items.sort(key=lambda d: d.confidence, reverse=True)
            kept: List[Detection] = []

            for det in items:
                if len(kept) >= self._max_detections_per_class:
                    break

                box = det.bounding_box.model_dump()
                is_dup = False
                for existing in kept:
                    existing_box = existing.bounding_box.model_dump()
                    if calculate_iou(box, existing_box) > self._dedup_iou_threshold:
                        is_dup = True
                        break

                if not is_dup:
                    kept.append(det)

            result.extend(kept)

        return result
