# if python version >= 3.11:
# we run the test, otherwise skip
import sys

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "Skipping test: requires Python 3.11 or higher", allow_module_level=True
    )
else:
    from eegdash.features.datasets import FeaturesConcatDataset, FeaturesDataset


@pytest.fixture
def dummy_features_dataset():
    """Creates a dummy FeaturesDataset for testing."""
    features = pd.DataFrame(
        np.random.rand(10, 5), columns=[f"feat_{i}" for i in range(5)]
    )
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": range(10),
            "i_start_in_trial": range(10),
            "i_stop_in_trial": range(10),
            "target": np.random.randint(0, 2, 10),
        }
    )
    return FeaturesDataset(features, metadata)


def test_zscore_dtype(dummy_features_dataset):
    """Tests if the zscore method correctly casts the output to float32."""
    concat_ds = FeaturesConcatDataset([dummy_features_dataset])
    concat_ds.zscore()
    assert concat_ds.datasets[0].features.dtypes.iloc[0] == "float32"
