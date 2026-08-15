"""Training orchestration for auditable baseline reproductions."""

from .cube_data import validate_cube_training_dataset
from .lewm import load_training_protocol, train_lewm
from .td_jepa import apply_tdjepa_cube_overrides, train_tdjepa_cube

__all__ = [
    "apply_tdjepa_cube_overrides",
    "load_training_protocol",
    "train_tdjepa_cube",
    "train_lewm",
    "validate_cube_training_dataset",
]
