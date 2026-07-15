import unittest
from unittest.mock import patch, MagicMock
import requests
from shared.schemas.analysis_result import AnalysisResult, AnalysisStatus
from shared.services.callback_service import send_callback, CallbackError

class TestCallback(unittest.TestCase):
    def setUp(self):
        self.result = AnalysisResult(
            requestId="req-123",
            mediaId="med-456",
            status=AnalysisStatus.COMPLETED,
            modelName="YOLO11",
            modelVersion="1.0.0",
            processingTimeMs=100
        )

    @patch("requests.post")
    def test_send_callback_success_first_try(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        success = send_callback(
            result=self.result,
            callback_url="http://test/callback",
            service_key="secret",
            max_retries=3,
            base_delay=0.01
        )
        
        self.assertTrue(success)
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    @patch("time.sleep")
    def test_send_callback_retry_then_success(self, mock_sleep, mock_post):
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        
        mock_success = MagicMock()
        mock_success.status_code = 200
        
        mock_post.side_effect = [mock_fail, mock_fail, mock_success]

        success = send_callback(
            result=self.result,
            callback_url="http://test/callback",
            service_key="secret",
            max_retries=3,
            base_delay=0.01
        )
        
        self.assertTrue(success)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("requests.post")
    def test_send_callback_non_retryable_error(self, mock_post):
        mock_fail = MagicMock()
        mock_fail.status_code = 400
        mock_fail.text = "Bad Request"
        mock_post.return_value = mock_fail

        with self.assertRaises(CallbackError) as ctx:
            send_callback(
                result=self.result,
                callback_url="http://test/callback",
                service_key="secret",
                max_retries=3,
                base_delay=0.01
            )
        self.assertIn("Callback rejected with status 400", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    @patch("time.sleep")
    def test_send_callback_exhaust_retries(self, mock_sleep, mock_post):
        mock_fail = MagicMock()
        mock_fail.status_code = 502
        mock_post.return_value = mock_fail

        with self.assertRaises(CallbackError) as ctx:
            send_callback(
                result=self.result,
                callback_url="http://test/callback",
                service_key="secret",
                max_retries=2,
                base_delay=0.01
            )
        self.assertIn("Callback delivery failed after 3 attempts", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)

if __name__ == "__main__":
    unittest.main()
