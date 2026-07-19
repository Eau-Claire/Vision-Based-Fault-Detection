import unittest
from unittest.mock import patch

import numpy as np

from shared.schemas.analysis_result import BoundingBox, Detection, DetectionResult
from server_pc.app.roboflow_detector import ServerRoboflowWorkflowDetector


class FakeRun:
    def as_detection_result(self):
        return DetectionResult(
            detections=[
                Detection(
                    categoryCode="VE",
                    confidence=0.9,
                    boundingBox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
                )
            ],
            image_width=0,
            image_height=0,
            frame_count=1,
        )


class TestServerRoboflowWorkflowDetector(unittest.TestCase):
    @patch("server_pc.app.roboflow_detector.run_evn_object_detection_workflow")
    def test_detect_image_uses_server_workflow_client(self, run_workflow):
        run_workflow.return_value = [FakeRun()]
        detector = ServerRoboflowWorkflowDetector(api_key="test-key", timeout_seconds=7)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.detect_image(image)

        self.assertEqual(detector.model_name, "Roboflow Workflow")
        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)
        self.assertEqual(len(result.detections), 1)
        run_workflow.assert_called_once()
        self.assertEqual(run_workflow.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(run_workflow.call_args.kwargs["timeout_seconds"], 7)


if __name__ == "__main__":
    unittest.main()
