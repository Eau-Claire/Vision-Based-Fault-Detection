import unittest
from shared.services.class_mapping import map_class_to_category

class TestClassMapping(unittest.TestCase):
    def test_map_normal_classes(self):
        self.assertIsNone(map_class_to_category("normal"))
        self.assertIsNone(map_class_to_category("clean"))
        self.assertIsNone(map_class_to_category("clean-insulator"))
        self.assertIsNone(map_class_to_category("insulator"))

    def test_map_defect_classes(self):
        # Cracked insulator defects
        self.assertEqual(map_class_to_category("broken-disc"), "CI")
        self.assertEqual(map_class_to_category("broken_glass"), "CI")
        self.assertEqual(map_class_to_category("cracked-insulator"), "CI")
        self.assertEqual(map_class_to_category("damage"), "CI")
        self.assertEqual(map_class_to_category("pollution-flashover"), "CI")
        self.assertEqual(map_class_to_category("dirt-insulator"), "CI")

        # Damaged Splice defects
        self.assertEqual(map_class_to_category("disconnected"), "DS")
        self.assertEqual(map_class_to_category("misroute"), "DS")
        self.assertEqual(map_class_to_category("splice"), "DS")

        # Vegetation Encroachment
        self.assertEqual(map_class_to_category("vegetation"), "VE")
        self.assertEqual(map_class_to_category("vegetation-encroachment"), "VE")

        # Corrosion
        self.assertEqual(map_class_to_category("corrosion"), "CC")
        self.assertEqual(map_class_to_category("rust"), "CC")

        # Birds / nests
        self.assertEqual(map_class_to_category("bird-nest"), "BW")

        # Sagging
        self.assertEqual(map_class_to_category("sagging"), "SA")

    def test_fuzzy_matching_and_fallback(self):
        # Fuzzy match for conductor corrosion
        self.assertEqual(map_class_to_category("highly-corroded-conductor"), "CC")
        # Unknown label should fall back to "OT" (Other) instead of None
        self.assertEqual(map_class_to_category("random-unknown-defect-type"), "OT")

if __name__ == "__main__":
    unittest.main()
