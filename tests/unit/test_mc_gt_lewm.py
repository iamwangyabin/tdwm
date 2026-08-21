from __future__ import annotations

import numpy as np
import torch

from tdwm.training.mc_gt_lewm import (
    CachedCubeGoalTailDataset,
    load_mc_gt_protocol,
)


def test_cached_cube_dataset_maps_clip_rows_and_action_history(tmp_path):
    latent_path = tmp_path / "latents.npy"
    latents = np.arange(40, dtype=np.float32).reshape(20, 2)
    np.save(latent_path, latents)
    actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    clips = [(0, start) for start in range(4)] + [
        (1, start) for start in range(4)
    ]
    dataset = CachedCubeGoalTailDataset(
        latent_cache_path=latent_path,
        normalized_actions=actions,
        clip_indices=clips,
        episode_offsets=[0, 10],
        source_indices=[1, 5],
        frame_skip=2,
        num_steps=3,
        history_size=3,
    )

    first = dataset[0]
    second = dataset[1]

    assert torch.equal(first["latents"], torch.from_numpy(latents[[1, 3, 5]]))
    assert torch.equal(
        first["action_history"], torch.from_numpy(actions[1:5].reshape(2, 4))
    )
    assert torch.equal(second["latents"], torch.from_numpy(latents[[11, 13, 15]]))


def test_cached_cube_dataset_supports_batched_worker_loading(tmp_path):
    latent_path = tmp_path / "latents.npy"
    np.save(latent_path, np.arange(80, dtype=np.float32).reshape(40, 2))
    dataset = CachedCubeGoalTailDataset(
        latent_cache_path=latent_path,
        normalized_actions=np.arange(80, dtype=np.float32).reshape(40, 2),
        clip_indices=[(0, start) for start in range(8)],
        episode_offsets=[0],
        source_indices=np.arange(8),
        frame_skip=2,
        num_steps=3,
        history_size=3,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    batch = next(iter(loader))

    assert batch["latents"].shape == (4, 3, 2)
    assert batch["action_history"].shape == (4, 2, 4)


def test_formal_mc_gt_protocol_names_and_covers_the_full_method():
    protocol = load_mc_gt_protocol(
        "configs/experiment/mc_gt_lewm_cube_train.yaml"
    )

    assert protocol["method"] == "mc_gt_lewm"
    assert protocol["display_name"] == "MC-GT-LeWM"
    assert protocol["stage"] == "full_training"
    assert protocol["base_model"]["frozen"] is True
    assert protocol["sequence"]["history_frames"] == 3
    assert protocol["sequence"]["max_goal_offset"] == 16
    assert protocol["training"]["coverage"] == "all_training_clips_each_epoch"
    assert protocol["training"]["checkpoint_selection"] == "minimum_validation_mse"
    assert protocol["planner"]["connected"] is False
    assert "td_horizon" not in protocol["tail_value"]
