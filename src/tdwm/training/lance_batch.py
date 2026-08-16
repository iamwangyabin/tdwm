from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class StrideBatchPlan:
    """Rows needed to assemble a batch of strided sequence clips."""

    global_starts: tuple[int, ...]
    unique_frame_rows: tuple[int, ...]
    frame_gathers: tuple[tuple[int, ...], ...]
    legacy_row_requests: int

    @property
    def image_row_requests(self) -> int:
        return len(self.unique_frame_rows)


@dataclass(frozen=True)
class PrefetchedStrideBlock:
    """Decoded rows shared by several consecutive LeWM mini-batches."""

    global_starts: tuple[int, ...]
    frame_gathers: tuple[tuple[int, ...], ...]
    row_data: dict[str, Any]
    decoded_images: dict[str, Any]
    dense_actions: Any


def build_stride_batch_plan(
    *,
    clip_indices: Sequence[tuple[int, int]],
    offsets: Sequence[int],
    indices: Sequence[int],
    frameskip: int,
    num_steps: int,
) -> StrideBatchPlan:
    """Map clip indices to only the observation rows consumed by LeWM."""

    if frameskip <= 0:
        raise ValueError("frameskip must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    global_starts: list[int] = []
    sample_rows: list[tuple[int, ...]] = []
    clip_count = len(clip_indices)
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx < 0:
            idx += clip_count
        if idx < 0 or idx >= clip_count:
            raise IndexError(f"Clip index {raw_idx} is out of range.")
        episode, start = clip_indices[idx]
        global_start = int(offsets[episode]) + int(start)
        global_starts.append(global_start)
        sample_rows.append(
            tuple(global_start + step * frameskip for step in range(num_steps))
        )

    unique_frame_rows = tuple(sorted({row for rows in sample_rows for row in rows}))
    row_positions = {row: position for position, row in enumerate(unique_frame_rows)}
    frame_gathers = tuple(
        tuple(row_positions[row] for row in rows) for rows in sample_rows
    )
    return StrideBatchPlan(
        global_starts=tuple(global_starts),
        unique_frame_rows=unique_frame_rows,
        frame_gathers=frame_gathers,
        legacy_row_requests=len(indices) * frameskip * num_steps,
    )


def _decode_images(blobs: Sequence[Any]):
    import torch

    if not blobs:
        return torch.empty(0, dtype=torch.uint8)

    try:
        from torchvision.io import ImageReadMode, decode_jpeg

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            encoded = [
                torch.frombuffer(
                    blob
                    if isinstance(blob, (bytes, bytearray))
                    else bytes(blob),
                    dtype=torch.uint8,
                )
                for blob in blobs
            ]
        return torch.stack(decode_jpeg(encoded, mode=ImageReadMode.RGB))
    except (AttributeError, ImportError, RuntimeError, TypeError):
        from PIL import Image

        decoded = []
        for blob in blobs:
            with Image.open(io.BytesIO(bytes(blob))) as image:
                array = np.array(image.convert("RGB"), copy=True)
            decoded.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(decoded)


def _numeric_tensor(values: Any):
    import torch

    array = np.asarray(values)
    if array.dtype == object or array.dtype.kind in ("S", "U"):
        raise TypeError("The stride-aware Cube loader only accepts numeric columns.")
    tensor = torch.tensor(array)
    if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor


class StrideAwareLanceDataset:
    """Avoid fetching intermediate Lance image rows discarded by ``frameskip``.

    The adapter uses the public dataset row/column accessors and preserves the
    released LeWM sample structure. Dense actions are still read across the
    complete temporal span; observation and image columns use strided rows.
    """

    def __init__(self, dataset: Any) -> None:
        required = (
            "clip_indices",
            "offsets",
            "frameskip",
            "num_steps",
            "span",
            "column_names",
            "get_col_data",
            "get_row_data",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "The Lance dataset is missing required public attributes: "
                + ", ".join(missing)
            )
        columns = list(dataset.column_names)
        if "pixels" not in columns or "action" not in columns:
            raise ValueError(
                "The stride-aware LeWM loader requires pixels and action columns."
            )
        self.dataset = dataset

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def prefetch(self, indices: Sequence[int]) -> PrefetchedStrideBlock:
        """Read and decode the strided rows needed by a local clip block once."""
        if not indices:
            raise ValueError("Cannot prefetch an empty clip block.")

        plan = build_stride_batch_plan(
            clip_indices=self.dataset.clip_indices,
            offsets=self.dataset.offsets,
            indices=indices,
            frameskip=int(self.dataset.frameskip),
            num_steps=int(self.dataset.num_steps),
        )
        frame_rows = list(plan.unique_frame_rows)
        row_data = self.dataset.get_row_data(frame_rows)
        image_columns = {
            name
            for name in self.dataset.column_names
            if name == "pixels" or name.startswith("pixels_")
        }
        decoded_images = {
            name: _decode_images(np.asarray(row_data[name], dtype=object).tolist())
            for name in image_columns
        }
        return PrefetchedStrideBlock(
            global_starts=plan.global_starts,
            frame_gathers=plan.frame_gathers,
            row_data=row_data,
            decoded_images=decoded_images,
            dense_actions=self.dataset.get_col_data("action"),
        )

    def materialize_prefetched(
        self,
        prefetched: PrefetchedStrideBlock,
        positions: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Assemble selected clips from an already-decoded local block."""

        image_columns = set(prefetched.decoded_images)
        block_size = len(prefetched.global_starts)

        results: list[dict[str, Any]] = []
        for raw_position in positions:
            position = int(raw_position)
            if position < 0 or position >= block_size:
                raise IndexError(f"Block position {raw_position} is out of range.")
            global_start = prefetched.global_starts[position]
            gather = prefetched.frame_gathers[position]
            gather_indices = list(gather)
            steps: dict[str, Any] = {}
            for column in self.dataset.column_names:
                if column in image_columns:
                    steps[column] = prefetched.decoded_images[column][gather_indices]
                elif column == "action":
                    action_end = global_start + int(self.dataset.span)
                    steps[column] = _numeric_tensor(
                        prefetched.dense_actions[global_start:action_end]
                    )
                else:
                    steps[column] = _numeric_tensor(
                        np.asarray(prefetched.row_data[column])[gather_indices]
                    )

            if self.dataset.transform:
                steps = self.dataset.transform(steps)
            steps["action"] = steps["action"].reshape(
                int(self.dataset.num_steps), -1
            )
            results.append(steps)
        return results

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        prefetched = self.prefetch(indices)
        return self.materialize_prefetched(prefetched, range(len(indices)))


class BlockPrefetchBatchDataset(IterableDataset):
    """Yield collated LeWM batches from sequentially fetched, decoded blocks.

    The train split is sorted by backing clip index before a block is fetched,
    so Lance receives one local row request and one JPEG decode pass per block.
    Mini-batches are shuffled only after that data is resident in the worker's
    memory.  This keeps the model inputs and sample coverage unchanged while
    removing the batch-by-batch remote-storage round trips.
    """

    def __init__(
        self,
        dataset: StrideAwareLanceDataset,
        source_indices: Sequence[int],
        *,
        batch_size: int,
        block_size: int,
        drop_last: bool,
        generator: Any,
        shuffle_batches_within_block: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if block_size < batch_size:
            raise ValueError("block_size must be at least batch_size.")
        if block_size % batch_size:
            raise ValueError("block_size must be divisible by batch_size.")
        self._dataset = dataset
        self._source_indices = tuple(sorted(int(index) for index in source_indices))
        self._batch_size = batch_size
        self._block_size = block_size
        self._drop_last = drop_last
        self._generator = generator
        self._shuffle_batches_within_block = shuffle_batches_within_block

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._source_indices) // self._batch_size
        return (len(self._source_indices) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        import torch
        from torch.utils.data import default_collate, get_worker_info

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        if worker is None:
            schedule_generator = self._generator
        else:
            # DataLoader derives worker seeds from its checkpointed generator.
            # Removing the worker offset makes every worker build the same
            # global block permutation before taking its disjoint share.
            schedule_generator = torch.Generator().manual_seed(
                int(worker.seed) - worker_id
            )

        block_starts = list(range(0, len(self._source_indices), self._block_size))
        if len(block_starts) > 1:
            order = torch.randperm(
                len(block_starts), generator=schedule_generator
            ).tolist()
            block_starts = [block_starts[position] for position in order]

        for block_start in block_starts[worker_id::worker_count]:
            source_block = self._source_indices[
                block_start : block_start + self._block_size
            ]
            prefetched = self._dataset.prefetch(source_block)
            batches = [
                list(range(offset, min(offset + self._batch_size, len(source_block))))
                for offset in range(0, len(source_block), self._batch_size)
            ]
            if self._drop_last and batches and len(batches[-1]) != self._batch_size:
                batches.pop()
            if self._shuffle_batches_within_block and len(batches) > 1:
                order = torch.randperm(
                    len(batches), generator=schedule_generator
                ).tolist()
                batches = [batches[position] for position in order]
            for positions in batches:
                yield default_collate(
                    self._dataset.materialize_prefetched(prefetched, positions)
                )
