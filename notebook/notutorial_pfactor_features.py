"""# Predicting p-factor from EEG - Example

The code below provides an example of using the *braindecode* and *EEGDash* libraries in combination with LightGBM to predict a subject's p-factor.

1. **Data Retrieval Using EEGDash**: An instance of *EEGDashDataset* is created to search and retrieve resting state data. At this step, only the metadata is transferred.

2. **Data Preprocessing Using BrainDecode**: This process preprocesses EEG data using Braindecode by selecting specific channels, resampling, filtering, and extracting 10-second epochs.

3. **Extracting EEG Features Using EEGDash.features**: Building a feature extraction tree and extracting features per EEG window.

4. **Model Training and Evaluation Process**: This section normalizes input data, trains a LightGBM model, and evaluates regression MSE.
"""
# %% [markdown]
# ## Data Retrieval Using EEGDash

# %%
from eegdash import EEGDashDataset

# hbn_datasets_train = ['ds005505', 'ds005506', 'ds005507', 'ds005508', 'ds005510', 'ds005511', 'ds005512', 'ds005514', 'ds005515', 'ds005516']
hbn_datasets_train = ["ds005505", "ds005506", "ds005507"]
hbn_datasets_valid = ["ds005509"]
desc_fields = ["subject", "session", "run", "task", "age", "gender", "sex", "p_factor"]
task_name = "RestingState"
target_name = "p_factor"

datasets_train = [
    EEGDashDataset(
        {"dataset": ds, "task": task_name},
        description_fields=desc_fields,
        target_name=target_name,
    )
    for ds in hbn_datasets_train
]
datasets_valid = [
    EEGDashDataset(
        {"dataset": ds, "task": task_name},
        description_fields=desc_fields,
        target_name=target_name,
    )
    for ds in hbn_datasets_valid
]

# %%
from braindecode.datasets import BaseConcatDataset

raw_train = BaseConcatDataset(
    [ds for dataset in datasets_train for ds in dataset.datasets]
)
raw_valid = BaseConcatDataset(
    [ds for dataset in datasets_valid for ds in dataset.datasets]
)

# %% [markdown]
# ## Data Preprocessing Using Braindecode
#
# [BrainDecode](https://braindecode.org/stable/install/install.html) is a specialized library for preprocessing EEG and MEG data.
#
# We apply three preprocessing steps in Braindecode:
# 1.	**Selection** of 24 specific EEG channels from the original 128.
# 2.	**Resampling** the EEG data to a frequency of 128 Hz.
# 3.	**Filtering** the EEG signals to retain frequencies between 1 Hz and 55 Hz.
#
# When calling the **preprocess** function, the data is retrieved from the remote repository.
#
# Finally, we use **create_windows_from_events** to extract 10-second epochs from the data. These epochs serve as the dataset samples.

# %%
import os

from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)


def preprocess_and_window(raw_ds):
    # preprocessing using a Braindecode pipeline:
    preprocessors = [
        Preprocessor(
            "pick_channels",
            ch_names=[
                "E22",
                "E9",
                "E33",
                "E24",
                "E11",
                "E124",
                "E122",
                "E29",
                "E6",
                "E111",
                "E45",
                "E36",
                "E104",
                "E108",
                "E42",
                "E55",
                "E93",
                "E58",
                "E52",
                "E62",
                "E92",
                "E96",
                "E70",
                "Cz",
            ],
        ),
        Preprocessor("resample", sfreq=128),
        Preprocessor("filter", l_freq=1, h_freq=55),
    ]
    preprocess(raw_ds, preprocessors, n_jobs=-1)

    # extract windows and save to disk
    sfreq = raw_ds.datasets[0].raw.info["sfreq"]
    windows_ds = create_fixed_length_windows(
        raw_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=int(10 * sfreq),
        window_stride_samples=int(5 * sfreq),
        drop_last_window=True,
        preload=False,
    )
    return windows_ds


windows_train = preprocess_and_window(raw_train)
os.makedirs(f"data/hbn_preprocessed_{task_name}_train", exist_ok=True)
windows_train.save(f"data/hbn_preprocessed_{task_name}_train", overwrite=True)

windows_valid = preprocess_and_window(raw_valid)
os.makedirs(f"data/hbn_preprocessed_{task_name}_valid", exist_ok=True)
windows_valid.save(f"data/hbn_preprocessed_{task_name}_valid", overwrite=True)

# %% [markdown]
# ## Extracting EEG Features Using EEGDash.features

from functools import partial

# %%
from eegdash import features
from eegdash.features import extract_features

sfreq = windows_train.datasets[0].raw.info["sfreq"]
filter_freqs = dict(windows_train.datasets[0].raw_preproc_kwargs)["filter"]
features_dict = {
    "sig": features.FeatureExtractor(
        {
            "std": features.signal_std,
            "line_len": features.signal_line_length,
            "zero_x": features.signal_zero_crossings,
        },
    ),
    "spec": features.SpectralFeatureExtractor(
        {
            "rtot_power": features.spectral_root_total_power,
            "band_power": features.spectral_bands_power,
            0: features.NormalizedSpectralFeatureExtractor(
                {
                    "moment": features.spectral_moment,
                    "entropy": features.spectral_entropy,
                    "edge": partial(features.spectral_edge, edge=0.9),
                },
            ),
            1: features.DBSpectralFeatureExtractor(
                {
                    "slope": features.spectral_slope,
                },
            ),
        },
        fs=sfreq,
        f_min=filter_freqs["l_freq"],
        f_max=filter_freqs["h_freq"],
        nperseg=4 * sfreq,
        noverlap=3 * sfreq,
    ),
}

features_train = extract_features(
    windows_train, features_dict, batch_size=64, n_jobs=-1
)
os.makedirs(f"data/hbn_features_{task_name}_train", exist_ok=True)
features_train.save(f"data/hbn_features_{task_name}_train", overwrite=True)

features_valid = extract_features(
    windows_valid, features_dict, batch_size=64, n_jobs=-1
)
os.makedirs(f"data/hbn_features_{task_name}_valid", exist_ok=True)
features_valid.save(f"data/hbn_features_{task_name}_valid", overwrite=True)

# %%
features_train.to_dataframe()

# %% [markdown]
# Replace Inf and NaN values:

# %%
import numpy as np

features_train.replace([-np.inf, +np.inf], np.nan)
features_train.fillna(0)

features_valid.replace([-np.inf, +np.inf], np.nan)
features_valid.fillna(0)

# %%
features_train.to_dataframe()

# %% [markdown]
# ## Model Training and Evaluation

# %% [markdown]
# Convert to pandas dataframes and normalize:

# %%
mean_train = features_train.mean(n_jobs=-1)
std_train = features_train.std(eps=1e-14, n_jobs=-1)

X_train = features_train.to_dataframe()
X_train = (X_train - mean_train) / std_train
y_train = features_train.get_metadata()["target"]

X_valid = features_valid.to_dataframe()
X_valid = (X_valid - mean_train) / std_train
y_valid = features_valid.get_metadata()["target"]

# %% [markdown]
# ### Train

# %%
from lightgbm import LGBMRegressor, record_evaluation

random_seed = 137

model = LGBMRegressor(
    random_state=random_seed,
    n_jobs=-1,
    n_estimators=10000,
    num_leaves=5,
    max_depth=2,
    min_data_in_leaf=4,
    learning_rate=0.1,
    early_stopping_round=5,
    first_metric_only=True,
)

eval_results = dict()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_names=["train", "validation"],
    eval_metric="l2",
    callbacks=[record_evaluation(eval_results)],
)

y_hat_train = model.predict(X_train)
correct_train = ((y_train - y_hat_train) ** 2).mean()
y_hat_valid = model.predict(X_valid)
correct_valid = ((y_valid - y_hat_valid) ** 2).mean()
print(f"Train MSE: {correct_train:.2f}, Validation MSE: {correct_valid:.2f}\n")

# %% [markdown]
# ### Plot Results

# %%
from lightgbm import plot_metric

plot_metric(model, "l2")

# %%
from lightgbm import plot_importance

plot_importance(model, importance_type="split", max_num_features=10)

# %%
plot_importance(model, importance_type="gain", max_num_features=10)
