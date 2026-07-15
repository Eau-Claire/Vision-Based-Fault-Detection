import unittest
from shared.schemas.analysis_request import PreferredModel, AnalysisRequest, MediaType, AnalysisType

class TestRouting(unittest.TestCase):
    def test_preferred_model_helpers(self):
        # Edge cases
        self.assertTrue(PreferredModel.is_edge("YOLO11"))
        self.assertTrue(PreferredModel.is_edge("EDGE"))
        self.assertTrue(PreferredModel.is_edge("yolo11"))
        self.assertTrue(PreferredModel.is_edge("edge"))
        self.assertFalse(PreferredModel.is_edge("RF-DETR"))
        self.assertFalse(PreferredModel.is_edge("SERVER"))
        self.assertFalse(PreferredModel.is_edge(None))
        self.assertFalse(PreferredModel.is_edge(""))

        # Server cases
        self.assertTrue(PreferredModel.is_server("RF-DETR"))
        self.assertTrue(PreferredModel.is_server("SERVER"))
        self.assertTrue(PreferredModel.is_server("rf-detr"))
        self.assertTrue(PreferredModel.is_server("server"))
        self.assertFalse(PreferredModel.is_server("YOLO11"))
        self.assertFalse(PreferredModel.is_server("EDGE"))
        self.assertFalse(PreferredModel.is_server(None))
        self.assertFalse(PreferredModel.is_server(""))

    def test_routing_validation_in_analysis_request(self):
        payload = {
            "requestId": "req-1",
            "mediaId": "med-1",
            "fileUrl": "http://example.com/image.jpg",
            "mediaType": "Image",
            "analysisType": "General",
            "preferredModel": "YOLO11"
        }
        request = AnalysisRequest(**payload)
        self.assertEqual(request.request_id, "req-1")
        self.assertEqual(request.preferred_model, "YOLO11")
        self.assertTrue(PreferredModel.is_edge(request.preferred_model))
        self.assertFalse(PreferredModel.is_server(request.preferred_model))

if __name__ == "__main__":
    unittest.main()
