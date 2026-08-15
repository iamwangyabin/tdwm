import unittest
from io import BytesIO

import numpy as np

from tdwm.training.lance_batch import (
    StrideAwareLanceDataset,
    build_stride_batch_plan,
)

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None
    Image = None


class StrideBatchPlanTest(unittest.TestCase):
    def test_only_requests_frames_consumed_by_the_stride(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 0), (0, 5)],
            offsets=[0],
            indices=[0, 1],
            frameskip=5,
            num_steps=4,
        )

        self.assertEqual(plan.global_starts, (0, 5))
        self.assertEqual(plan.unique_frame_rows, (0, 5, 10, 15, 20))
        self.assertEqual(plan.frame_gathers, ((0, 1, 2, 3), (1, 2, 3, 4)))
        self.assertEqual(plan.legacy_row_requests, 40)
        self.assertEqual(plan.image_row_requests, 5)

    def test_episode_offsets_are_applied_before_stride(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 2), (1, 3)],
            offsets=[0, 100],
            indices=[1, 0],
            frameskip=2,
            num_steps=3,
        )

        self.assertEqual(plan.global_starts, (103, 2))
        self.assertEqual(plan.unique_frame_rows, (2, 4, 6, 103, 105, 107))
        self.assertEqual(plan.frame_gathers, ((3, 4, 5), (0, 1, 2)))

    def test_negative_indices_match_dataset_indexing(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 0), (0, 1)],
            offsets=[10],
            indices=[-1],
            frameskip=1,
            num_steps=2,
        )

        self.assertEqual(plan.global_starts, (11,))
        self.assertEqual(plan.unique_frame_rows, (11, 12))

    def test_invalid_parameters_fail_before_access(self):
        with self.assertRaisesRegex(ValueError, "frameskip"):
            build_stride_batch_plan(
                clip_indices=[],
                offsets=[],
                indices=[],
                frameskip=0,
                num_steps=4,
            )
        with self.assertRaisesRegex(IndexError, "out of range"):
            build_stride_batch_plan(
                clip_indices=[(0, 0)],
                offsets=[0],
                indices=[1],
                frameskip=1,
                num_steps=1,
            )


@unittest.skipUnless(torch is not None and Image is not None, "PyTorch is required")
class StrideAwareLanceDatasetTest(unittest.TestCase):
    class FakeLanceDataset:
        def __init__(self):
            self.offsets = np.array([0])
            self.lengths = np.array([10])
            self.frameskip = 2
            self.num_steps = 3
            self.span = 6
            self.clip_indices = [(0, start) for start in range(5)]
            self.column_names = ["pixels", "action", "observation"]
            self.action = np.arange(20, dtype=np.float32).reshape(10, 2)
            self.observation = np.arange(30, dtype=np.float32).reshape(10, 3)
            self.pixels = []
            for value in range(10):
                array = np.full((8, 8, 3), value * 20, dtype=np.uint8)
                buffer = BytesIO()
                Image.fromarray(array).save(buffer, format="JPEG", quality=100)
                self.pixels.append(buffer.getvalue())
            self.requested_rows = None
            self.transform_calls = 0
            self.transform = self._transform

        def __len__(self):
            return len(self.clip_indices)

        def get_row_data(self, rows):
            self.requested_rows = list(rows)
            return {
                "pixels": np.asarray([self.pixels[row] for row in rows], dtype=object),
                "action": self.action[rows],
                "observation": self.observation[rows],
            }

        def get_col_data(self, column):
            if column == "action":
                return self.action
            if column == "observation":
                return self.observation
            raise KeyError(column)

        def get_dim(self, column):
            return self.get_col_data(column).shape[1]

        def _transform(self, sample):
            self.transform_calls += 1
            sample["observation"] = sample["observation"] + 1
            return sample

    def test_batch_preserves_dense_actions_and_strided_observations(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)

        samples = dataset.__getitems__([0, 2])

        self.assertEqual(source.requested_rows, [0, 2, 4, 6])
        self.assertEqual(source.transform_calls, 2)
        self.assertEqual(tuple(samples[0]["pixels"].shape), (3, 3, 8, 8))
        torch.testing.assert_close(
            samples[0]["action"],
            torch.tensor(source.action[0:6]).reshape(3, 4),
        )
        torch.testing.assert_close(
            samples[1]["observation"],
            torch.tensor(source.observation[[2, 4, 6]]) + 1,
        )


if __name__ == "__main__":
    unittest.main()
