"""Public API for dataset helpers and dynamically generated datasets."""

from . import dataset as _dataset_mod  # triggers dynamic class registration
from .dataset import EEGChallengeDataset
from .registry import register_openneuro_datasets

# Re-export dynamically generated dataset classes at the package level so that
# ``eegdash.dataset`` shows them in the API docs and users can import as
# ``from eegdash.dataset import DSXXXXX``.
_dyn_names = []
for _name in getattr(_dataset_mod, "__all__", []):
    if _name == "EEGChallengeDataset":
        # Already imported explicitly above
        continue
    _obj = getattr(_dataset_mod, _name, None)
    if _obj is not None:
        globals()[_name] = _obj
        _dyn_names.append(_name)

__all__ = ["EEGChallengeDataset", "register_openneuro_datasets"] + _dyn_names

del _dataset_mod, _name, _obj, _dyn_names
