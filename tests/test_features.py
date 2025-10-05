import numpy as np
import pandas as pd
import pytest
from eegdash.features.datasets import FeaturesDataset, FeaturesConcatDataset

@pytest.fixture
def dummy_features_dataset():
    """Creates a dummy FeaturesDataset for testing."""
    features = pd.DataFrame(np.random.rand(10, 5), columns=[f"feat_{i}" for i in range(5)])
    metadata = pd.DataFrame({
        "i_window_in_trial": range(10),
        "i_start_in_trial": range(10),
        "i_stop_in_trial": range(10),
        "target": np.random.randint(0, 2, 10),
    })
    return FeaturesDataset(features, metadata)

def test_zscore_dtype(dummy_features_dataset):
    """Tests if the zscore method correctly casts the output to float32."""
    concat_ds = FeaturesConcatDataset([dummy_features_dataset])
    concat_ds.zscore()
    assert concat_ds.datasets[0].features.dtypes.iloc[0] == "float32"