import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server_pc.app.analysis_runner import HarnessAnalysisRunner
from server_pc.app.consumer import create_server_consumer


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = REPO_ROOT / "dataset/train/Broken-Glass/su110kv_vo--37-_jpg.rf.h3PRWvHBOg3cOTG3f0cZ.jpg"


def make_settings(tmpdir):
    return SimpleNamespace(
        callback_base_url="http://backend.local",
        callback_url="http://backend.local/api/internal/ai-analysis/results",
        media_download_timeout=10,
        media_max_size_bytes=1024 * 1024,
        allow_private_ips=True,
        ai_service_key="test-service-key",
        callback_max_retries=0,
        callback_retry_base_delay=0,
        callback_retry_max_delay=0,
        callback_timeout=5,
        restrict_callback_to_base_url=True,
        rabbitmq_host="",
        rabbitmq_port=5672,
        rabbitmq_user="guest",
        rabbitmq_pass="guest",
        server_queue_name="server",
        rabbitmq_exchange="ai.analysis",
        edge_queue_name="edge",
        dead_letter_exchange="dlx",
        dead_letter_queue="dlq",
        rabbitmq_heartbeat=600,
        rabbitmq_prefetch_count=1,
        harness_checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
        harness_workflow_ref="fake://evn-object-detection",
    )


class FakeChannel:
    def __init__(self):
        self.acked = []
        self.nacked = []

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue=False):
        self.nacked.append((delivery_tag, requeue))


class ServerHarnessFlowTests(unittest.TestCase):
    def test_harness_runner_returns_callback_compatible_detection_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(tmpdir) / "checkpoints",
            )

            output = runner.analyze_media(
                file_bytes=FIXTURE_IMAGE.read_bytes(),
                extension=".jpg",
                media_type="Image",
            )

            self.assertEqual(output.model_name, "HarnessRuntime")
            self.assertTrue(output.harness_run_id.startswith("run-"))
            self.assertTrue(Path(output.harness_checkpoint_path).exists())
            self.assertEqual(output.detection_result.frame_count, 1)
            self.assertGreater(output.detection_result.image_width, 0)
            self.assertGreater(output.detection_result.image_height, 0)
            self.assertGreaterEqual(len(output.detection_result.detections), 1)

    @patch("shared.services.callback_service.send_callback")
    @patch("shared.services.media_downloader.download_media")
    def test_rest_api_background_flow_uses_harness_runner_and_callback_contract(
        self,
        download_media,
        send_callback,
    ):
        from server_pc.app import main as server_main

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE.read_bytes(), ".jpg")

            old_settings = server_main.settings
            old_runner = server_main.analysis_runner
            server_main.settings = settings
            server_main.analysis_runner = runner
            try:
                payload = server_main.AnalyzePayload(
                    requestId="req-rest-1",
                    mediaId="media-rest-1",
                    fileUrl="/uploads/image.jpg",
                    mediaType="Image",
                    callbackUrl=settings.callback_url,
                    correlationId="corr-rest-1",
                )
                server_main._run_analysis(payload)
            finally:
                server_main.settings = old_settings
                server_main.analysis_runner = old_runner

            send_callback.assert_called_once()
            result = send_callback.call_args.kwargs["result"]
            self.assertEqual(result.request_id, "req-rest-1")
            self.assertEqual(result.media_id, "media-rest-1")
            self.assertEqual(result.model_name, "HarnessRuntime")
            self.assertIn("harnessRunId", result.raw_result)
            self.assertIn("harnessCheckpointPath", result.raw_result)
            self.assertGreaterEqual(len(result.detections), 1)

    @patch("shared.services.callback_service.send_callback")
    @patch("shared.services.media_downloader.download_media")
    def test_api_analyze_endpoint_schedules_harness_flow(
        self,
        download_media,
        send_callback,
    ):
        from fastapi.testclient import TestClient
        from server_pc.app import main as server_main

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE.read_bytes(), ".jpg")

            old_settings = server_main.settings
            old_runner = server_main.analysis_runner
            old_ready = server_main._ready
            server_main.settings = settings
            server_main.analysis_runner = runner
            server_main._ready = True
            try:
                client = TestClient(server_main.app)
                response = client.post(
                    "/api/analyze",
                    json={
                        "requestId": "req-api-1",
                        "mediaId": "media-api-1",
                        "fileUrl": "/uploads/image.jpg",
                        "mediaType": "Image",
                        "preferredModel": "SERVER",
                        "callbackUrl": settings.callback_url,
                        "correlationId": "corr-api-1",
                    },
                )
            finally:
                server_main.settings = old_settings
                server_main.analysis_runner = old_runner
                server_main._ready = old_ready

            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["requestId"], "req-api-1")
            send_callback.assert_called_once()
            result = send_callback.call_args.kwargs["result"]
            self.assertEqual(result.request_id, "req-api-1")
            self.assertEqual(result.model_name, "HarnessRuntime")
            self.assertIn("harnessRunId", result.raw_result)

    @patch("server_pc.app.consumer.send_callback")
    @patch("server_pc.app.consumer.download_media")
    def test_rabbitmq_consumer_flow_uses_harness_runner_and_callback_contract(
        self,
        download_media,
        send_callback,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE.read_bytes(), ".jpg")
            callback = create_server_consumer(runner, settings)
            channel = FakeChannel()
            method = SimpleNamespace(delivery_tag="delivery-1")
            body = json.dumps(
                {
                    "requestId": "req-rabbit-1",
                    "mediaId": "media-rabbit-1",
                    "fileUrl": "/uploads/image.jpg",
                    "mediaType": "Image",
                    "preferredModel": "SERVER",
                    "callbackUrl": settings.callback_url,
                    "correlationId": "corr-rabbit-1",
                }
            ).encode("utf-8")

            callback(channel, method, None, body)

            self.assertEqual(channel.acked, ["delivery-1"])
            self.assertEqual(channel.nacked, [])
            send_callback.assert_called_once()
            result = send_callback.call_args.kwargs["result"]
            self.assertEqual(result.request_id, "req-rabbit-1")
            self.assertEqual(result.media_id, "media-rabbit-1")
            self.assertEqual(result.model_name, "HarnessRuntime")
            self.assertIn("harnessRunId", result.raw_result)
            self.assertIn("harnessCheckpointPath", result.raw_result)
            self.assertGreaterEqual(len(result.detections), 1)


if __name__ == "__main__":
    unittest.main()
