import os
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from shared.services.roboflow_workflow_client import (
    ROBOFLOW_OUTPUT_NAMES,
    RoboflowConfigurationError,
    parse_workflow_response,
    _RestWorkflowClient,
    run_evn_object_detection_workflow,
)


class FakeWorkflowClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run_workflow(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class TestRoboflowWorkflowClient(unittest.TestCase):
    def test_parse_real_shape_predictions_output(self):
        raw = [
            {
                "predictions": {
                    "image": {"width": 1000, "height": 500},
                    "predictions": [
                        {
                            "class": "vegetation",
                            "confidence": 0.91,
                            "x": 250,
                            "y": 100,
                            "width": 100,
                            "height": 50,
                        },
                        {
                            "class": "insulator",
                            "confidence": 0.88,
                            "x": 500,
                            "y": 250,
                            "width": 80,
                            "height": 60,
                        },
                    ],
                }
            }
        ]

        parsed = parse_workflow_response(raw)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(set(parsed[0].outputs.keys()), set(ROBOFLOW_OUTPUT_NAMES))
        self.assertEqual(parsed[0].image_width, 1000)
        self.assertEqual(parsed[0].image_height, 500)
        self.assertEqual(len(parsed[0].detections), 1)
        self.assertEqual(parsed[0].detections[0].category_code, "VE")
        self.assertAlmostEqual(parsed[0].detections[0].bounding_box.x, 0.2)

    def test_rejects_parameters_not_declared_by_workflow(self):
        with self.assertRaises(RoboflowConfigurationError):
            run_evn_object_detection_workflow(
                "https://example.com/image.jpg",
                api_key="test-key",
                parameters={"confidence": 0.4},
                client=FakeWorkflowClient([]),
            )

    def test_runs_sdk_client_with_declared_workflow_identity(self):
        client = FakeWorkflowClient([{"predictions": []}])

        parsed = run_evn_object_detection_workflow(
            "https://example.com/image.jpg",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(client.calls[0]["workspace_name"], "les-workspace-ijdwd")
        self.assertEqual(
            client.calls[0]["workflow_id"],
            "evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic",
        )
        self.assertEqual(client.calls[0]["images"]["image"], "https://example.com/image.jpg")
        self.assertEqual(client.calls[0]["parameters"], {})

    @patch("shared.services.roboflow_workflow_client.requests.post")
    def test_rest_client_posts_base64_workflow_payload(self, post):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"predictions": []}]

        post.return_value = FakeResponse()
        buffer = BytesIO()
        Image.new("RGB", (10, 10), color="white").save(buffer, format="JPEG")
        client = _RestWorkflowClient(api_key="test-key")

        result = client.run_workflow(
            workspace_name="les-workspace-ijdwd",
            workflow_id="workflow-id",
            images={"image": Image.open(BytesIO(buffer.getvalue()))},
            parameters={},
        )

        self.assertEqual(result, [{"predictions": []}])
        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            url,
            "https://serverless.roboflow.com/les-workspace-ijdwd/workflows/workflow-id",
        )
        self.assertEqual(payload["api_key"], "test-key")
        self.assertEqual(payload["inputs"]["image"]["type"], "base64")
        self.assertGreater(len(payload["inputs"]["image"]["value"]), 100)

    @patch("shared.services.roboflow_workflow_client.requests.post")
    def test_rest_client_posts_https_url_workflow_payload(self, post):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"predictions": []}]

        post.return_value = FakeResponse()
        client = _RestWorkflowClient(api_key="test-key")

        client.run_workflow(
            workspace_name="les-workspace-ijdwd",
            workflow_id="workflow-id",
            images={"image": "https://example.com/image.jpg"},
            parameters={},
        )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["inputs"]["image"],
            {"type": "url", "value": "https://example.com/image.jpg"},
        )


class TestRoboflowWorkflowSmoke(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_ROBOFLOW_SMOKE_TEST") == "1",
        "Set RUN_ROBOFLOW_SMOKE_TEST=1 and ROBOFLOW_API_KEY to run live smoke test",
    )
    def test_live_workflow_returns_declared_output_keys(self):
        result = run_evn_object_detection_workflow(
            "https://media.roboflow.com/notebooks/examples/dog.jpeg",
            timeout_seconds=int(os.getenv("ROBOFLOW_TIMEOUT", "30")),
            max_retries=int(os.getenv("ROBOFLOW_MAX_RETRIES", "2")),
        )

        self.assertGreaterEqual(len(result), 1)
        self.assertTrue(set(ROBOFLOW_OUTPUT_NAMES).issubset(result[0].outputs.keys()))


if __name__ == "__main__":
    unittest.main()
