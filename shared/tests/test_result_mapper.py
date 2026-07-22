import unittest
from shared.schemas.analysis_result import DetectionResult, Detection, BoundingBox, AnalysisStatus
from shared.services.result_mapper import map_success_result, map_failure_result

class TestResultMapper(unittest.TestCase):
    def test_map_success_with_detections(self):
        det = Detection(
            categoryCode="CI",
            confidence=0.92,
            boundingBox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            timestampMs=1000,
            frameIndex=30
        )
        det_result = DetectionResult(
            detections=[det],
            image_width=1920,
            image_height=1080,
            frame_count=100,
            fps=30,
            duration=132.5
        )
        
        result = map_success_result(
            request_id="req-123",
            media_id="med-456",
            detection_result=det_result,
            model_name="YOLO11",
            model_version="1.0.0",
            processing_time_ms=150,
            device_profile="edge",
            asset_id="asset-001",
            image_url="https://pms.local/media/video.mp4"
        )
        
        self.assertEqual(result.request_id, "req-123")
        self.assertEqual(result.media_id, "med-456")
        self.assertEqual(result.status, AnalysisStatus.COMPLETED)
        self.assertEqual(result.model_name, "YOLO11")
        self.assertEqual(result.processing_time_ms, 150)
        self.assertEqual(len(result.detections), 1)
        mapped_detection = result.detections[0]
        self.assertEqual(mapped_detection.category_code, "CI")
        self.assertEqual(mapped_detection.class_name, "Cracked Insulator")
        self.assertEqual(mapped_detection.timestamp, 1.0)
        self.assertEqual(mapped_detection.frame_index, 30)
        self.assertEqual(mapped_detection.image_url, "https://pms.local/media/video.mp4")
        self.assertIsNone(mapped_detection.crop_url)
        self.assertEqual(mapped_detection.asset_id, "asset-001")
        self.assertEqual(result.video_metadata.duration, 132.5)
        self.assertEqual(result.video_metadata.fps, 30)
        self.assertEqual(result.video_metadata.width, 1920)
        self.assertEqual(result.video_metadata.height, 1080)
        self.assertEqual(result.raw_result["imageWidth"], 1920)
        self.assertEqual(result.raw_result["fps"], 30)
        self.assertEqual(result.raw_result["duration"], 132.5)
        self.assertEqual(result.raw_result["deviceProfile"], "edge")

    def test_map_success_empty_detections(self):
        # Must return Completed even when detections is an empty list
        det_result = DetectionResult(
            detections=[],
            image_width=640,
            image_height=480,
            frame_count=1
        )
        result = map_success_result(
            request_id="req-789",
            media_id=None,
            detection_result=det_result,
            model_name="YOLO11",
            model_version="1.0.0",
            processing_time_ms=50,
            device_profile="edge"
        )
        self.assertEqual(result.status, AnalysisStatus.COMPLETED)
        self.assertEqual(len(result.detections), 0)

    def test_map_failure(self):
        result = map_failure_result(
            request_id="req-failed",
            media_id="med-failed",
            error_code="MODEL_INFERENCE_FAILED",
            error_message="CUDA out of memory"
        )
        self.assertEqual(result.status, AnalysisStatus.FAILED)
        self.assertEqual(result.error_code, "MODEL_INFERENCE_FAILED")
        self.assertEqual(result.error_message, "CUDA out of memory")
        self.assertEqual(len(result.detections), 0)

if __name__ == "__main__":
    unittest.main()
