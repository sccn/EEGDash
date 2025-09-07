""".. _tutorial-surround-suppression:
    Surround Suppression Contrastive Learning Tutorial
    =================================================

    This script demonstrates how to use the EEGDash and Braindecode libraries to perform contrastive learning on EEG data from a surround suppression task. The workflow includes data retrieval, preprocessing, contrastive pair sampling, model definition, and training.

    Key Steps:
    1. **Data Retrieval**: Uses EEGDashDataset to query and retrieve metadata for a surround suppression EEG dataset.
    2. **Preprocessing**: Applies channel selection, resampling, filtering, and epoch extraction using Braindecode to prepare the data for analysis.
    3. **Contrastive Dataset and Sampler**: Implements custom classes to generate pairs of EEG samples for contrastive learning, ensuring pairs are from the same or different stimulus conditions.
    4. **Model Architecture**: Defines a shallow convolutional neural network (ShallowFBCSPNet) as an encoder to produce embeddings, and a classification head that predicts whether paired samples are from the same class.
    5. **Contrastive Training**: Trains the encoder and classification head jointly using binary cross-entropy loss on paired samples.

    ---------------
    - **Pretrained Encoder Usage**: After training, only the encoder model (ShallowFBCSPNet) is of interest. The classification head is discarded.
    - **Downstream Applications**: For future downstream tasks, the pretrained encoder will be used to transform EEG data into an embedding space. Additional fine-tuning or task-specific heads can then be applied to these embeddings for specific analyses or predictions.
"""

# %%
# Data Retrieval Using EEGDash
# ----------------------------
#
# First we find one resting state dataset. This dataset contains both eyes open
# and eyes closed data.
from pathlib import Path

cache_folder = Path.home() / "eegdash"
# %%
from eegdash import EEGDashDataset
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset

ds_sus = EEGDashDataset(
    query={"dataset": "ds005514", "task": "surroundSupp", "subject": "NDARDB033FW5"},
    cache_dir=cache_folder, download=False
)

sfreq = ds_sus.datasets[0].raw.info["sfreq"]

# %%
# Data Preprocessing Using Braindecode
# ------------------------------------
#
# [BrainDecode](https://braindecode.org/stable/install/install.html) is a
# specialized library for preprocessing EEG and MEG data. In this dataset, the
# key event is **stim_ON**, marking presentation of visual stimuli. 
# We extract non-overlapping 2-second epochs after the event onset till the onset
# of the next event. The event extraction is handled by the custom
# preprocessor **SelectEventMarker()**.
#
# Next, we apply four preprocessing steps in Braindecode:
# 1. **Reannotation** of event markers using `SelectEventMarker()`.
# 2. **Selection** of 24 specific EEG channels from the original 128.
# 3. **Resampling** the EEG data to a frequency of 128 Hz.
# 4. **Filtering** the EEG signals to retain frequencies between 1 Hz and 55 Hz.
#
# When calling the `preprocess` function, the data is retrieved from the remote
# repository.
#
# Finally, we use `create_windows_from_events` to extract 2-second epochs from
# the data. These epochs serve as the dataset samples. At this stage, each
# sample is automatically labeled with the corresponding event type (eyes-open
# or eyes-closed). `windows_ds` is a PyTorch dataset, and when queried, it
# returns labels for eyes-open and eyes-closed (assigned as labels 0 and 1,
# corresponding to their respective event markers).

# %%
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
import numpy as np
import mne
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# BrainDecode preprocessors
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
preprocess(ds_sus, preprocessors)

# Extract 2-second segments
windows_ds = create_windows_from_events(
    ds_sus,
    mapping={"stim_ON": 1,
             "fixpoint_ON": 0},  # Map event markers to labels
    trial_start_offset_samples=0,
    trial_stop_offset_samples=int(128 * 2),
    preload=True,
)

# %%
# Plotting a Single Channel for One Sample
# ----------------------------------------
#
# It’s always a good practice to verify that the data has been properly loaded
# and processed. Here, we plot a single channel from one sample to ensure the
# signal is present and looks as expected.

# %%
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(windows_ds[2][0][0, :].transpose())  # first channel of first epoch
# plt.show()


# %%
# Creating the Contrastive DataLoader
# -----------------------------------
#
# We use a custom **ContrastiveSampler** and **ContrastiveDataset** 
# to generate pairs of EEG samples for contrastive learning. 
# The sampler presamples pairs of indices (with labels indicating
# whether they are from the same class), and the dataset returns the
# corresponding EEG data for each pair.
#
# 1. **Set Random Seed** – The random seed is fixed using
#    `torch.manual_seed(random_state)` and `np.random.seed(random_state)` to
#    ensure reproducibility in sampling and model training.
# 2. **Create ContrastiveDataset** – Wrap the windowed EEG datasets in a
#    `ContrastiveDataset` to enable paired sample retrieval.
# 3. **Create ContrastiveSampler** – Use `ContrastiveSampler` to generate
#    presampled pairs of indices and labels for contrastive learning.
# 4. **Create DataLoader** – The DataLoader uses the custom dataset and sampler,
#    providing batches of paired EEG samples and their labels for training.
#
# This setup is tailored for contrastive learning, where each batch contains
# pairs of EEG samples and a label indicating if they belong to the same class.
#
from braindecode.samplers import RecordingSampler

class ContrastiveSampler(RecordingSampler):
    """Sample examples for the contrastive task.

    Sample examples as tuples of two window indices, with a label indicating
    whether the windows are from the same class (1) or not (0).

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_examples : int
        Number of pairs to extract.
    random_state : None | np.RandomState | int
        Random state.
    """
    def __init__(
        self,
        metadata,
        n_examples,
        random_state=None,
    ):
        super().__init__(metadata, random_state=random_state)
        self.n_examples = n_examples

    def _sample_pair(self):
        """Sample a pair of two windows."""
        # Sample first window
        win_ind1, rec_ind1 = self.sample_window()
        ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
        ts1_target = self.metadata.iloc[win_ind1]["target"]

        # Sample second window
        win_ind2, rec_ind2 = self.sample_window()
        ts2 = self.metadata.iloc[win_ind2]["i_start_in_trial"]
        ts2_target = self.metadata.iloc[win_ind2]["target"]

        current_pair_type = 0 if ts1_target != ts2_target else 1
        selected_pair_type = 1  # self.rng.binomial(1, 0.5)  # 0 for different type, 1 for same type

        # Keep sampling until we have two different windows and the right pair type
        while ts1 == ts2 or current_pair_type != selected_pair_type:
            win_ind2, rec_ind2 = self.sample_window()
            ts2 = self.metadata.iloc[win_ind2]["i_start_in_trial"]
            ts2_target = self.metadata.iloc[win_ind2]["target"]
            current_pair_type = 0 if ts1_target != ts2_target else 1

        return win_ind1, win_ind2, float(selected_pair_type)

    def presample(self):
        """Presample examples.

        Once presampled, the examples are the same from one epoch to another.
        """
        self.examples = [self._sample_pair() for _ in range(self.n_examples)]
        return self

    def __iter__(self):
        for i in range(self.n_examples):
            if hasattr(self, "examples"):
                yield self.examples[i]
            else:
                yield self._sample_pair()

    def __len__(self):
        return self.n_examples

class ContrastiveDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0], super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value

contrastive_windows_ds = ContrastiveDataset(windows_ds.datasets)

multiply_factor = 10  # How many times to oversample the dataset
n_examples = multiply_factor * len(contrastive_windows_ds.datasets)
random_state = 42

contrastive_sampler = ContrastiveSampler(
    contrastive_windows_ds.get_metadata(),
    n_examples=n_examples,
    random_state=random_state,
)

# %%
import torch
from torch.utils.data import DataLoader

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

train_dataloader = DataLoader(
    contrastive_windows_ds, batch_size=10, sampler=contrastive_sampler
)

# %%
# Create model
# ------------
#
# The model consists of a shallow convolutional neural network encoder (ShallowFBCSPNet)
# that outputs a 64-dimensional embedding for each EEG sample (24 channels, 256 time points).
# For contrastive learning, two samples are passed through the encoder, and the absolute
# difference of their embeddings is computed. This difference is then passed through a
# linear classification head to predict whether the two samples belong to the same class.

# %%
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

torch.manual_seed(random_state)
embedding_dim = 64
encoder_model = ShallowFBCSPNet(24, embedding_dim, n_times=256, final_conv_length="auto")
classification_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    nn.Dropout(0.5),
    torch.nn.Linear(embedding_dim, 1)  
)
summary(encoder_model, input_size=(1, 24, 256))

# %%
# Model Training Loop for Contrastive Learning
# --------------------------------------------
#
# This section trains the encoder and classification head using the Adamax optimizer.
# Each batch contains pairs of EEG samples (x1, x2) and a label y (1 if same class, 0 otherwise).
# Both samples are normalized, passed through the encoder, and their embeddings' absolute difference
# is classified. Binary cross-entropy loss is used. Accuracy is tracked per epoch.

# %%
params = list(encoder_model.parameters()) + list(classification_model.parameters())
optimizer = torch.optim.Adamax(params, lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
encoder_model = encoder_model.to(device=device)  # move the model parameters to CPU/GPU
classification_model = classification_model.to(device=device)
epochs = 6

def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
    x = (x - mean) / std
    x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
    return x

encoder_model.train()  # put model to training mode
classification_model.train()  # put model to training mode
for e in range(epochs):
    # training
    for t, batch in enumerate(train_dataloader):
        (x1, x2), y = batch
        y = y.to(device=device, dtype=torch.long)

        x1, x2 = normalize_data(x1), normalize_data(x2)

        z1, z2 = encoder_model(x1), encoder_model(x2)

        y_pred = classification_model(torch.abs(z1 - z2).squeeze()).squeeze()

        loss = F.binary_cross_entropy(y_pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

# %%
# Save encoder model weight.
# %%
torch.save(encoder_model.state_dict(), "encoder_model_weights.pth")