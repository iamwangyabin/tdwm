import ast
from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_platform_dependency_is_exactly_pinned() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["dependencies"] == ["stable-worldmodel[all]==0.1.1"]


def test_first_executable_experiment_matches_lewm_pusht_protocol() -> None:
    environment = _yaml("configs/envs/pusht.yaml")
    method = _yaml("configs/methods/lewm.yaml")

    assert environment["world"]["env_name"] == "swm/PushT-v1"
    assert environment["dataset"]["training_loader"]["split"]["seed"] == 42
    assert method["implementation"]["version"] == "0.1.1"
    assert method["factory"]["_target_"] == "stable_worldmodel.wm.LeWM"
    assert method["sequence"] == {
        "frameskip": 5,
        "history_size": 3,
        "num_predictions": 1,
        "num_steps": 4,
    }
    assert method["loss"]["sigreg"]["weight"] == 0.09
    assert method["training"]["epochs"] == 10
    assert method["training"]["batch_size"] == 128


def test_checkpoint_export_converts_plain_mapping_to_omegaconf() -> None:
    source = (ROOT / "src/tdwm/training/lewm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    save_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_pretrained"
    ]

    assert len(save_calls) == 1
    config = next(
        keyword.value
        for keyword in save_calls[0].keywords
        if keyword.arg == "config"
    )
    assert isinstance(config, ast.Call)
    assert isinstance(config.func, ast.Attribute)
    assert isinstance(config.func.value, ast.Name)
    assert (config.func.value.id, config.func.attr) == ("OmegaConf", "create")
