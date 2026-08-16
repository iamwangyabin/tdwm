import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np

from tdwm.training.lance_batch import (
    BlockPrefetchBatchDataset,
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
            self.row_requests = 0
            self.transform_calls = 0
            self.transform = self._transform

        def __len__(self):
            return len(self.clip_indices)

        def get_row_data(self, rows):
            self.requested_rows = list(rows)
            self.row_requests += 1
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

    def test_prefetched_block_decodes_once_and_materializes_selected_clips(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)

        prefetched = dataset.prefetch([0, 1, 2, 3])
        first_batch = dataset.materialize_prefetched(prefetched, [0, 1])
        second_batch = dataset.materialize_prefetched(prefetched, [2, 3])

        self.assertEqual(source.row_requests, 1)
        self.assertEqual(source.transform_calls, 4)
        self.assertEqual(tuple(first_batch[0]["pixels"].shape), (3, 3, 8, 8))
        torch.testing.assert_close(
            second_batch[1]["action"],
            torch.tensor(source.action[3:9]).reshape(3, 4),
        )

    def test_block_prefetch_yields_every_source_clip_once(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)
        batches = BlockPrefetchBatchDataset(
            dataset,
            [4, 0, 3, 1, 2],
            batch_size=2,
            block_size=4,
            drop_last=False,
            generator=torch.Generator().manual_seed(7),
            shuffle_batches_within_block=True,
        )

        emitted = list(batches)

        self.assertEqual(len(emitted), 3)
        self.assertEqual(source.row_requests, 2)
        starts = []
        for batch in emitted:
            starts.extend(batch["pixels"][:, 0, 0, 0, 0].tolist())
        self.assertEqual(sorted(starts), [0, 20, 40, 60, 80])

    def test_block_prefetch_partitions_blocks_across_workers(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)
        batches = BlockPrefetchBatchDataset(
            dataset,
            [4, 0, 3, 1, 2],
            batch_size=2,
            block_size=4,
            drop_last=False,
            generator=torch.Generator().manual_seed(7),
            shuffle_batches_within_block=True,
        )

        class Worker:
            def __init__(self, worker_id):
                self.id = worker_id
                self.num_workers = 2
                self.seed = 11 + worker_id

        def collate_in_test(samples):
            return {
                name: torch.stack([sample[name] for sample in samples])
                for name in samples[0]
            }

        with (
            patch("torch.utils.data.get_worker_info", return_value=Worker(0)),
            patch("torch.utils.data.default_collate", side_effect=collate_in_test),
        ):
            first_worker = list(batches)
        with (
            patch("torch.utils.data.get_worker_info", return_value=Worker(1)),
            patch("torch.utils.data.default_collate", side_effect=collate_in_test),
        ):
            second_worker = list(batches)
        emitted = first_worker + second_worker

        self.assertEqual(len(emitted), 3)
        starts = []
        for batch in emitted:
            starts.extend(batch["pixels"][:, 0, 0, 0, 0].tolist())
        self.assertEqual(sorted(starts), [0, 20, 40, 60, 80])


if __name__ == "__main__":
    unittest.main()
