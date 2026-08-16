import unittest

import torch

from tdwm.training.block_sampler import BlockShuffleBatchSampler


class BlockShuffleBatchSamplerTest(unittest.TestCase):
    def test_covers_each_subset_position_once_without_dropping(self):
        sampler = BlockShuffleBatchSampler(
            [17, 2, 10, 3, 11, 1, 12, 4, 13, 5],
            batch_size=2,
            block_size=4,
            drop_last=False,
            generator=torch.Generator().manual_seed(7),
            shuffle_batches_within_block=True,
        )

        batches = list(sampler)

        self.assertEqual(len(sampler), 5)
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(10))
        )

    def test_batch_keeps_adjacent_source_indices_together(self):
        source_indices = [11, 0, 9, 1, 10, 2, 8, 3, 7, 4, 6, 5]
        sampler = BlockShuffleBatchSampler(
            source_indices,
            batch_size=3,
            block_size=6,
            drop_last=True,
            generator=torch.Generator().manual_seed(0),
            shuffle_batches_within_block=False,
        )

        batches = list(sampler)

        self.assertEqual(
            [[source_indices[index] for index in batch] for batch in batches],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )

    def test_validation_mode_is_sorted_and_visits_every_source_clip_once(self):
        source_indices = [11, 0, 9, 1, 10, 2, 8, 3, 7, 4, 6, 5]
        sampler = BlockShuffleBatchSampler(
            source_indices,
            batch_size=3,
            block_size=6,
            drop_last=False,
            shuffle_batches_within_block=False,
            shuffle_blocks=False,
        )

        batches = list(sampler)

        self.assertEqual(
            [[source_indices[index] for index in batch] for batch in batches],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )

    def test_seeded_schedule_is_reproducible(self):
        source_indices = list(reversed(range(16)))
        first = BlockShuffleBatchSampler(
            source_indices,
            batch_size=2,
            block_size=8,
            drop_last=True,
            generator=torch.Generator().manual_seed(42),
            shuffle_batches_within_block=True,
        )
        second = BlockShuffleBatchSampler(
            source_indices,
            batch_size=2,
            block_size=8,
            drop_last=True,
            generator=torch.Generator().manual_seed(42),
            shuffle_batches_within_block=True,
        )

        self.assertEqual(list(first), list(second))

    def test_rejects_partial_blocks_that_would_drop_each_block_tail(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            BlockShuffleBatchSampler(
                [0, 1, 2],
                batch_size=2,
                block_size=3,
                drop_last=True,
                generator=torch.Generator().manual_seed(0),
                shuffle_batches_within_block=False,
            )


if __name__ == "__main__":
    unittest.main()
