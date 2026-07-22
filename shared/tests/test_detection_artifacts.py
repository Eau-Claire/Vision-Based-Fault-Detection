import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from shared.schemas.analysis_result import BoundingBox, Detection, DetectionResult
from shared.services.detection_artifacts import enrich_video_detection_artifacts


class TestDetectionArtifacts(unittest.TestCase):
    def test_enrich_video_detection_artifacts_writes_frame_and_crop_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (100, 80),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is unavailable")

            try:
                for i in range(3):
                    frame = np.full((80, 100, 3), 30 + i * 40, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            detection = Detection(
                id="det/1",
                categoryCode="CI",
                confidence=0.9,
                boundingBox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
                frameIndex=1,
            )
            result = DetectionResult(
                detections=[detection],
                image_width=100,
                image_height=80,
                frame_count=3,
                fps=10,
                duration=0.3,
            )

            enriched = enrich_video_detection_artifacts(
                video_path=str(video_path),
                detection_result=result,
                request_id="req/abc",
                artifact_dir=str(temp_path / "artifacts"),
                public_base_url="http://ai.local:8002",
                artifact_url_path="/artifacts",
            )

            enriched_detection = enriched.detections[0]
            self.assertEqual(
                enriched_detection.image_url,
                "http://ai.local:8002/artifacts/req-abc/frame-1.jpg",
            )
            self.assertEqual(
                enriched_detection.crop_url,
                "http://ai.local:8002/artifacts/req-abc/crop-1-det-1.jpg",
            )
            self.assertTrue((temp_path / "artifacts/req-abc/frame-1.jpg").exists())
            self.assertTrue((temp_path / "artifacts/req-abc/crop-1-det-1.jpg").exists())

            crop = cv2.imread(str(temp_path / "artifacts/req-abc/crop-1-det-1.jpg"))
            self.assertIsNotNone(crop)
            self.assertEqual(crop.shape[1], 30)
            self.assertEqual(crop.shape[0], 32)


if __name__ == "__main__":
    unittest.main()
