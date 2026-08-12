import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_METHODS = ("pldm", "dino_wm", "gcbc", "gcivl", "gciql")


def _yaml(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_push_offline_methods_use_the_pinned_public_platform() -> None:
    for method_id in OFFLINE_METHODS:
        method = _yaml(f"configs/methods/{method_id}.yaml")
        assert method["id"] == method_id
        assert method["implementation"]["package"] == "stable-worldmodel[all]"
        assert method["implementation"]["version"] == "0.1.1"
        assert method["input"]["pixels"] is True
        assert method["input"]["proprioception"] is False


def test_world_model_factories_and_action_conditioning_are_auditable() -> None:
    pldm = _yaml("configs/methods/pldm.yaml")
    dino_wm = _yaml("configs/methods/dino_wm.yaml")

    assert pldm["factory"]["_target_"] == (
        "stable_worldmodel.wm.pldm.pldm.PLDM"
    )
    assert pldm["factory"]["action_encoder"]["input_dim"] == (
        "infer_from_dataset"
    )
    assert dino_wm["model"]["action_encoder"]["embedding_dimension"] == 10
    assert dino_wm["sequence"]["num_steps"] == (
        dino_wm["sequence"]["history_size"]
        + dino_wm["sequence"]["num_predictions"]
    )


def test_goal_sampling_is_valid_and_value_methods_are_staged() -> None:
    gcbc = _yaml("configs/methods/gcbc.yaml")
    assert sum(gcbc["goal_sampling"].values()) == 1.0

    for method_id, expected_stages in {
        "gcivl": ["value", "policy"],
        "gciql": ["value_and_q", "policy"],
    }.items():
        method = _yaml(f"configs/methods/{method_id}.yaml")
        assert sum(method["rl"]["value_goal_sampling"].values()) == 1.0
        assert sum(method["rl"]["actor_goal_sampling"].values()) == 1.0
        assert method["checkpoint"]["stages"] == expected_stages


def test_train_dispatcher_exposes_all_offline_methods_and_gates_tdmpc2() -> None:
    source = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert set(OFFLINE_METHODS).issubset(constants)
    assert "tdmpc2" in constants
    assert "stable_worldmodel as swm" in (
        ROOT / "src/tdwm/training/baselines.py"
    ).read_text(encoding="utf-8")


def test_parallel_launcher_queues_only_the_five_additional_methods() -> None:
    source = (ROOT / "scripts/launch_pusht_parallel.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_METHODS"
            for target in node.targets
        )
    )
    methods = tuple(
        element.value
        for element in assignment.value.elts
        if isinstance(element, ast.Constant)
    )
    assert methods == OFFLINE_METHODS


def test_dino_snapshot_path_is_runtime_injected() -> None:
    source = (ROOT / "src/tdwm/training/baselines.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("TDWM_DINO_BACKBONE", "dinov2_small")' in source
    assert "/gemini/" not in source


def test_baseline_exports_a_loadable_root_model_config() -> None:
    source = (ROOT / "src/tdwm/training/baselines.py").read_text(
        encoding="utf-8"
    )
    assert '"_target_": "tdwm.training.baselines.build_baseline_model"' in source
    assert "config=OmegaConf.create(self.model_config)" in source
    pldm_source = source.split("class PLDMTrainingModule", maxsplit=1)[1].split(
        "class PreJEPATrainingModule", maxsplit=1
    )[0]
    assert "target = embeddings[:, self.num_predictions :].detach()" not in pldm_source
