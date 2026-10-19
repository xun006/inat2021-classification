import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Classifier.data import Images
from Classifier.evaluate_official_val import dataset_audit, load_mapping, mapping_hash


class OfficialValSafetyTests(unittest.TestCase):
    def test_mapping_and_imagefolder_order_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "official_val"
            mapping = {"class_a": 0, "class_b": 1}
            map_path = root / "class_to_idx.json"
            map_path.write_text(json.dumps(mapping), encoding="utf-8")
            for name in mapping:
                (split / name).mkdir(parents=True)
                Image.new("RGB", (8, 8)).save(split / name / "image.jpg")
            loaded = load_mapping(map_path, 2)
            dataset = Images(split, loaded, 8, train=False)
            audit = dataset_audit(dataset, expected_samples=2)
            self.assertEqual(audit["classes"], 2)
            self.assertEqual(audit["min_samples_per_class"], 1)
            self.assertEqual(len(mapping_hash(loaded)), 64)

    def test_sample_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "official_val" / "only_class"
            split.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(split / "image.jpg")
            dataset = Images(split.parent, {"only_class": 0}, 8, train=False)
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                dataset_audit(dataset, expected_samples=2)

    def test_non_contiguous_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.json"
            path.write_text(json.dumps({"a": 0, "b": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_mapping(path, 2)


if __name__ == "__main__":
    unittest.main()
