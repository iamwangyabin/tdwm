from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_pldm_checkpoint_roundtrip_through_public_api(tmp_path: Path) -> None:
    pytest.importorskip("stable_worldmodel")
    pytest.importorskip("lightning")
    pytest.importorskip("transformers")
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from tdwm.training.baselines import _export_model_config

    with (ROOT / "configs" / "methods" / "pldm.yaml").open(
        encoding="utf-8"
    ) as handle:
        method = yaml.safe_load(handle)
    config = _export_model_config(
        method,
        effective_action_dimension=2,
        backbone_source=None,
    )
    model = instantiate(OmegaConf.create(config))
    swm.wm.save_pretrained(
        model,
        run_name="pldm_roundtrip",
        config=OmegaConf.create(config),
        filename="weights.pt",
        cache_dir=str(tmp_path),
    )

    loaded = swm.wm.load_pretrained("pldm_roundtrip", cache_dir=str(tmp_path))
    expected = model.state_dict()
    actual = loaded.state_dict()
    assert actual.keys() == expected.keys()
    assert all(actual[key].shape == expected[key].shape for key in expected)
