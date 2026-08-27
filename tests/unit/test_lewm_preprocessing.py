import unittest

try:
    import torch
except ImportError:
    torch = None

from tdwm.training.lewm import LeWMTransform, _preprocess_image_batch


@unittest.skipUnless(torch is not None, "PyTorch is required")
class LeWMImagePreprocessingTest(unittest.TestCase):
    def test_device_batch_path_matches_released_per_sample_transform(self):
        generator = torch.Generator().manual_seed(3072)
        pixels = torch.randint(
            0,
            256,
            (2, 4, 3, 64, 64),
            dtype=torch.uint8,
            generator=generator,
        )
        image = {
            "size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        transform = LeWMTransform(image=image, columns={})
        expected = torch.stack(
            [transform({"pixels": sample.clone()})["pixels"] for sample in pixels]
        )
        mean = torch.tensor(image["mean"]).reshape(1, 1, 3, 1, 1)
        std = torch.tensor(image["std"]).reshape(1, 1, 3, 1, 1)

        actual = _preprocess_image_batch(
            pixels, mean=mean, std=std, size=image["size"]
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_device_path_rejects_already_expanded_float_images(self):
        with self.assertRaisesRegex(TypeError, "expects uint8"):
            _preprocess_image_batch(
                torch.zeros(1, 4, 3, 64, 64),
                mean=torch.zeros(1, 1, 3, 1, 1),
                std=torch.ones(1, 1, 3, 1, 1),
                size=224,
            )

    def test_cube_resolution_avoids_a_redundant_resize(self):
        pixels = torch.full((1, 4, 3, 224, 224), 255, dtype=torch.uint8)
        mean = torch.zeros(1, 1, 3, 1, 1)
        std = torch.ones(1, 1, 3, 1, 1)

        actual = _preprocess_image_batch(
            pixels, mean=mean, std=std, size=224
        )

        self.assertEqual(tuple(actual.shape), (1, 4, 3, 224, 224))
        torch.testing.assert_close(actual, torch.ones_like(actual))


if __name__ == "__main__":
    unittest.main()
