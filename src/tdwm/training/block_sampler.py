"""Locality-preserving batch sampling for static clip datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


class BlockShuffleBatchSampler:
    """Emit contiguous source clips while randomizing their larger blocks.

    ``Subset`` indices are commonly a random permutation of the source dataset.
    This sampler returns positions into that subset, grouped by ascending source
    clip index. Lance therefore sees local row ranges for every batch while the
    order of blocks and batches is refreshed each epoch.
    """

    def __init__(
        self,
        source_indices: Sequence[int],
        *,
        batch_size: int,
        block_size: int,
        drop_last: bool,
        generator,
        shuffle_batches_within_block: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if block_size < batch_size:
            raise ValueError("block_size must be at least batch_size.")
        if block_size % batch_size:
            raise ValueError("block_size must be divisible by batch_size.")

        self._batch_size = batch_size
        self._block_size = block_size
        self._drop_last = drop_last
        self._generator = generator
        self._shuffle_batches_within_block = shuffle_batches_within_block
        self._positions = tuple(
            sorted(
                range(len(source_indices)),
                key=lambda position: int(source_indices[position]),
            )
        )

    def __iter__(self) -> Iterator[list[int]]:
        import torch

        block_starts = list(range(0, len(self._positions), self._block_size))
        if len(block_starts) > 1:
            order = torch.randperm(
                len(block_starts), generator=self._generator
            ).tolist()
            block_starts = [block_starts[position] for position in order]

        for block_start in block_starts:
            block = self._positions[block_start : block_start + self._block_size]
            batches = [
                list(block[offset : offset + self._batch_size])
                for offset in range(0, len(block), self._batch_size)
            ]
            if self._drop_last and batches and len(batches[-1]) != self._batch_size:
                batches.pop()
            if self._shuffle_batches_within_block and len(batches) > 1:
                order = torch.randperm(
                    len(batches), generator=self._generator
                ).tolist()
                batches = [batches[position] for position in order]
            yield from batches

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._positions) // self._batch_size
        return (len(self._positions) + self._batch_size - 1) // self._batch_size
