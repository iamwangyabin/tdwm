from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


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

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []

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
        dense_actions = self.dataset.get_col_data("action")

        results: list[dict[str, Any]] = []
        for global_start, gather in zip(plan.global_starts, plan.frame_gathers):
            gather_indices = list(gather)
            steps: dict[str, Any] = {}
            for column in self.dataset.column_names:
                if column in image_columns:
                    steps[column] = decoded_images[column][gather_indices]
                elif column == "action":
                    action_end = global_start + int(self.dataset.span)
                    steps[column] = _numeric_tensor(
                        dense_actions[global_start:action_end]
                    )
                else:
                    steps[column] = _numeric_tensor(
                        np.asarray(row_data[column])[gather_indices]
                    )

            if self.dataset.transform:
                steps = self.dataset.transform(steps)
            steps["action"] = steps["action"].reshape(
                int(self.dataset.num_steps), -1
            )
            results.append(steps)
        return results
