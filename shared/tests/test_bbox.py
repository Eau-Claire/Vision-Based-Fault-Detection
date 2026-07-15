import unittest
from shared.utils.bbox import normalize_bbox, clamp_bbox, calculate_iou

class TestBBox(unittest.TestCase):
    def test_normalize_bbox(self):
        # 1000x1000 image, box at (100, 200) to (500, 600)
        box = normalize_bbox(100, 200, 500, 600, 1000, 1000)
        self.assertAlmostEqual(box["x"], 0.1)
        self.assertAlmostEqual(box["y"], 0.2)
        self.assertAlmostEqual(box["width"], 0.4)
        self.assertAlmostEqual(box["height"], 0.4)

    def test_normalize_bbox_invalid_dims(self):
        box = normalize_bbox(100, 200, 500, 600, 0, 1000)
        self.assertEqual(box, {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0})

    def test_clamp_bbox(self):
        box = clamp_bbox(-0.5, 1.5, 2.0, -1.0)
        self.assertEqual(box, {"x": 0.0, "y": 1.0, "width": 1.0, "height": 0.0})

    def test_calculate_iou(self):
        box_a = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
        box_b = {"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3}
        
        # Intersection: x in [0.2, 0.4], y in [0.2, 0.4] -> width=0.2, height=0.2 -> area = 0.04
        # Area A = 0.09, Area B = 0.09
        # Union = 0.09 + 0.09 - 0.04 = 0.14
        # IoU = 0.04 / 0.14 = 0.2857
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 0.04 / 0.14)

        # No overlap
        box_c = {"x": 0.6, "y": 0.6, "width": 0.1, "height": 0.1}
        self.assertEqual(calculate_iou(box_a, box_c), 0.0)

if __name__ == "__main__":
    unittest.main()
