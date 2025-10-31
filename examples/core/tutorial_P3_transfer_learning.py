""".. _tutorial-p3-transfer-learning:

EEG P3 Transfer Learning with AS-MMD
====================================


This tutorial's corresponding method/paper: Chen, W., Delorme, A. (2025). Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG Classification. arXiv: [2510.21969](https://arxiv.org/abs/2510.21969).

Overview
--------
This tutorial demonstrates transfer learning for EEG P3 component analysis using 
Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD) to adapt between two oddball 
paradigm datasets: ERP CORE P3 and Active Visual Oddball (AVO).

The goal is to train a deep learning classifier (EEGConformer) that performs well 
on both datasets despite differences in participant populations, recording equipment, 
and experimental protocols. We use domain adaptation techniques including:

1. **Prototype-based Discriminative Transfer**: Align class representations across domains
2. **Mixup Data Augmentation**: Enhance learning from smaller sample sizes
3. **MMD Alignment**: Minimize distribution mismatch in feature space
4. **Nested Cross-Validation**: Robust performance estimation
"""
# %%
# Data paths - update these to your local directories
AVO_DATA_DIR = '/home/vivian/eeg/ds005863'  
P3_DATA_DIR = '/home/vivian/eeg/P3_Raw_Data_BIDS-Compatible'

# %%
# Dataset Wrapper Classes
# -----------------------
#
# Helper classes for managing BIDS datasets and windowed EEG data.

from pathlib import Path
from braindecode.preprocessing import Preprocessor

class EEGBIDSDataset:
    """Simple BIDS dataset wrapper."""
    
    def __init__(self, data_dir, dataset=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.base_path = Path(data_dir)
        
    def get_files(self):
        """Get all files in the dataset directory."""
        files = []
        if self.base_path.exists():
            for file_path in self.base_path.rglob('*'):
                if file_path.is_file():
                    files.append(file_path)
        return files


class ManualWindowsDataset:
    """Custom dataset that ensures one window per event."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimplePreprocessorBase:
    """Simple base preprocessor when braindecode is not available."""
    def __init__(self, fn, apply_on_array=False):
        self.fn = fn
        self.apply_on_array = apply_on_array

# %%
# Data Loading Utilities
# ----------------------
#
# Functions for loading raw EEG data and extracting subject information from
# both P3 and AVO datasets.

import os
import mne

# Suppress warnings for cleaner output
mne.set_log_level('ERROR')

def load_raw(file_path, dataset_type):
    """Load raw EEG data based on dataset type."""
    if dataset_type == 'P3':
        return mne.io.read_raw_eeglab(file_path, preload=True)
    else:  # AVO
        return mne.io.read_raw_brainvision(file_path, preload=True)


def get_dataset_subjects(dataset_type, dataset_obj):
    """Get list of subjects from dataset."""
    if dataset_type == 'P3':
        return sorted([d for d in os.listdir(dataset_obj) if d.startswith('sub-')])
    else:  # AVO
        all_files = [str(f) for f in dataset_obj.get_files()]
        return sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))


def process_subject_data(subject_id, dataset_obj, preprocessor, dataset_type='P3'):
    """Process a single subject's data.
    
    This function:
    1. Loads the raw EEG file(s) for the subject
    2. Applies the preprocessing pipeline
    3. Returns the windowed data and labels
    """
    try:
        if dataset_type == 'P3':
            eeg_file = os.path.join(
                dataset_obj, subject_id, 'eeg',
                f'{subject_id}_task-P3_eeg.set'
            )
            raw = load_raw(eeg_file, dataset_type)
        else:  # AVO
            all_files = [str(f) for f in dataset_obj.get_files()]
            vhdr_files = [
                f for f in all_files
                if f"sub-{subject_id}" in f and 'visualoddball' in f and f.endswith('.vhdr')
            ]
            if not vhdr_files:
                return None, None
            
            raws = [load_raw(f, dataset_type) for f in vhdr_files]
            for r in raws:
                r.load_data()
            raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

        # Process data
        windows = preprocessor.transform(raw)
        data = windows.data
        labels = windows.labels

        return data, labels

    except Exception as e:
        print(f"Error processing {dataset_type} subject {subject_id}: {e}")
        return None, None

# %%
# Data Normalization and Augmentation
# -----------------------------------
#
# Functions for normalizing EEG data and applying data augmentation techniques
# (Gaussian noise and time shifting) to improve model generalization.

import numpy as np
import torch

# Data augmentation parameters
NOISE_STD = 0.006
TIME_SHIFT_RANGE = 6

def normalize_data(x, eps=1e-7):
    """Normalize EEG data along time dimension.
    
    Applies z-score normalization: (x - mean) / std
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std


def augment_data(x, noise_std=NOISE_STD, time_shift=TIME_SHIFT_RANGE):
    """Apply data augmentation to EEG signals.
    
    Two augmentation techniques:
    1. Gaussian noise: Adds random noise to simulate natural variability
    2. Time shifting: Shifts signals along time axis to improve temporal robustness
    """
    augmented = x.clone()
    
    # Add Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
    
    # Time shifting
    if time_shift > 0:
        for i in range(x.size(0)):
            shift = np.random.randint(-time_shift, time_shift + 1)
            if shift != 0:
                augmented[i] = torch.roll(augmented[i], shift, dims=1)
    
    return augmented



# %%
# Harmonized Preprocessing Pipeline - Overview
# --------------------------------------------
#
# The preprocessing pipeline implements the 7-step harmonized approach from the
# AS-MMD paper. We'll break it down into modular components for clarity.

# %%
# Step 1: Preprocessor Initialization
# -----------------------------------
#
# The `OddballPreprocessor` class manages configuration for both P3 and AVO datasets.
# It handles event codes, trial budgets, and timing parameters specific to each dataset.

# Preprocessing parameters
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128
TRIAL_START_OFFSET = -0.1  # seconds before stimulus
TRIAL_DURATION = 1.1  # seconds total window

# Artifact removal and filtering
REMOVE_ARTIFACTS = True  # ICA for eye/muscle artifacts
APPLY_NOTCH_FILTER = True  # Remove power line interference
BASELINE_CORRECT = True  # Baseline correction using pre-stimulus interval
NOTCH_FREQS = [50, 60]  # Power line frequencies (Hz)

# Event codes
RESPONSE_EVENTS = [6, 7, 201, 202]
ODDBALL_EVENTS_P3 = [1, 9, 15, 21, 27]
ODDBALL_EVENTS_AVO = [11, 22, 33, 44, 55]

# Trials per subject
TRIALS_PER_SUBJECT_P3 = 80  # balanced: 40 oddball + 40 standard
TRIALS_PER_SUBJECT_AVO = 10  # smaller due to data availability

class OddballPreprocessor(Preprocessor):
    """Preprocessor for oddball paradigm EEG data with balanced sampling."""

    def __init__(self, eeg_channels, dataset_type='P3', random_seed=42):
        """Initialize preprocessor with dataset-specific parameters.
        
        Parameters
        ----------
        eeg_channels : list
            List of EEG channel names to use (e.g., ['Fz', 'Pz', 'P3', 'P4', 'Oz'])
        dataset_type : str
            Either 'P3' or 'AVO' to select appropriate event codes
        random_seed : int
            Random seed for reproducible sampling
        """
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.dataset_type = dataset_type
        self.random_seed = random_seed
        
        # Convert time offsets to samples
        self.trial_start_offset_samples = int(TRIAL_START_OFFSET * RESAMPLE_FREQ)
        self.trial_stop_offset_samples = int(TRIAL_DURATION * RESAMPLE_FREQ)
        
        # Set event codes based on dataset type
        if dataset_type == 'AVO':
            self.response_events = RESPONSE_EVENTS
            self.oddball_events = ODDBALL_EVENTS_AVO
            self.trials_per_class = TRIALS_PER_SUBJECT_AVO
        else:  # P3
            self.response_events = RESPONSE_EVENTS
            self.oddball_events = ODDBALL_EVENTS_P3
            self.trials_per_class = TRIALS_PER_SUBJECT_P3

# %%
# Step 2: ICA Artifact Removal Method
# -----------------------------------
#
# ICA (Independent Component Analysis) identifies and removes artifacts from EEG:
# - Eye movements (EOG)
# - Muscle activity
# - Other non-brain signals
#
# This step is critical for clean ERP analysis and follows the paper's methodology.

    def remove_artifacts_ica(self, raw_full):
        """Remove eye movement and muscle artifacts using ICA.
        
        The process:
        1. High-pass filter at 1 Hz (recommended for ICA stability)
        2. Fit ICA to decompose signals into independent components
        3. Automatically detect artifact components using frontal channels
        4. Remove bad components and back-project to get clean data
        """
        if not REMOVE_ARTIFACTS:
            return raw_full
        
        try:
            # Create copy for ICA (use high-pass 1 Hz as recommended for ICA)
            raw_ica = raw_full.copy()
            raw_ica.filter(l_freq=1.0, h_freq=None)
            
            # Set up ICA (up to 15 components, FastICA method)
            n_components = min(15, len(raw_ica.ch_names) - 1)
            
            if n_components < 2:
                return raw_full  # Too few channels for meaningful ICA
            
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                random_state=self.random_seed,
                method='fastica',
                max_iter=500
            )
            
            # Fit ICA on full-channel data
            ica.fit(raw_ica)
            
            # Detect EOG artifacts using frontal channels as reference
            frontal_channels = [ch for ch in raw_full.ch_names 
                              if any(front in ch.lower() 
                                   for front in ['fp1', 'fp2', 'af3', 'af4', 'f3', 'f4', 'fz', 'fpz'])]
            
            eog_indices = []
            if frontal_channels:
                for ch in frontal_channels[:2]:
                    try:
                        indices, scores = ica.find_bads_eog(raw_ica, ch_name=ch, threshold=3.0)
                        eog_indices.extend(indices)
                    except:
                        continue
            
            # Remove artifact components and back-project
            eog_indices = list(set(eog_indices))
            
            if eog_indices:
                ica.exclude = eog_indices
                raw_clean = ica.apply(raw_full.copy())
                return raw_clean
            
            return raw_full
            
        except Exception as e:
            print(f"  Warning: ICA artifact removal skipped ({str(e)[:50]})")
            return raw_full

# %%
# Step 3: Main Transform Pipeline - Initial Setup
# -----------------------------------------------
#
# The `transform` method orchestrates all preprocessing steps in sequence.
# First, we prepare the data: standardize channel names, set reference, and convert units.

    def transform(self, raw):
        """Transform raw EEG data into balanced windowed dataset.
        
        This method applies all 7 preprocessing steps in sequence.
        """
        # Standardize channel names to lowercase
        raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})

        # Set average reference on all channels (common practice for EEG)
        try:
            raw.set_eeg_reference('average', projection=True)
        except Exception:
            pass  # Use original reference if average reference fails

        # Check and convert data units if needed (V to µV)
        raw_data = raw.get_data()
        if np.std(raw_data) < 1e-6 and np.std(raw_data) > 0:
            raw._data *= 1e6

# %%
# Step 4: Resampling and Filtering
# --------------------------------
#
# Apply temporal filtering to remove unwanted frequencies:
# - Resample to 128 Hz (computational efficiency + sufficient for ERPs)
# - Notch filter at 50/60 Hz (remove power line interference)
# - Band-pass filter 0.5-30 Hz (retain ERP-relevant frequencies)
        
        # Resample to 128 Hz
        raw.resample(RESAMPLE_FREQ)
        
        # Apply notch filter to remove power line interference
        if APPLY_NOTCH_FILTER:
            for freq in NOTCH_FREQS:
                try:
                    raw.notch_filter(freq, verbose=False)
                except:
                    pass
        
        # Apply band-pass filter (0.5-30 Hz)
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)

# %%
# Step 5: Artifact Removal and Channel Selection
# ----------------------------------------------
#
# Now we clean the data with ICA (on all channels), then select our target channels.
# ICA works best on full-channel data, so we do it before channel selection.
        
        # ICA artifact removal (on full-channel data)
        raw = self.remove_artifacts_ica(raw)
        
        # Select target channels (after ICA cleaning)
        available_channels = [ch for ch in self.eeg_channels if ch in raw.ch_names]
        if not available_channels:
            raise ValueError(f"No requested channels found. Available: {raw.ch_names}")
        raw.pick_channels(available_channels)

# %%
# Step 6: Event Extraction and Balanced Sampling
# ----------------------------------------------
#
# Extract stimulus events and create balanced datasets:
# - Remove response events (button presses)
# - Separate oddball (rare) from standard (frequent) stimuli
# - Sample equal numbers of each class to prevent bias

        # Extract events from annotations
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found")

        # Remove response events
        response_mask = np.isin(events[:, 2], self.response_events)
        events = events[~response_mask]
        
        # Remove last event to avoid window overflow
        if len(events) > 0:
            events = events[:-1]
        
        # Separate oddball and standard events
        oddball_mask = np.isin(events[:, 2], self.oddball_events)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]
        
        if len(oddball_events) == 0 or len(standard_events) == 0:
            raise ValueError("Missing oddball or standard events")
        
        # Balanced sampling: same number of oddball and standard trials
        np.random.seed(self.random_seed)
        n_trials = min(len(oddball_events), len(standard_events), self.trials_per_class)
        
        oddball_indices = np.random.choice(len(oddball_events), size=n_trials, replace=False)
        standard_indices = np.random.choice(len(standard_events), size=n_trials, replace=False)
        
        selected_oddball = oddball_events[oddball_indices]
        selected_standard = standard_events[standard_indices]
        
        # Combine events and create labels (1=oddball, 0=standard)
        selected_events = np.vstack([selected_oddball, selected_standard])
        labels = np.concatenate([np.ones(n_trials), np.zeros(n_trials)]).astype(int)

# %%
# Step 7: Epoching (Windowing)
# ----------------------------
#
# Extract time-locked windows around each stimulus:
# - Window: -100ms to +1000ms relative to stimulus onset
# - Pre-stimulus interval will be used for baseline correction

        # Extract windows manually
        raw_data = raw.get_data()
        windows_data = []
        windows_labels = []
        
        for i, (event_sample, _, _) in enumerate(selected_events):
            start_sample = event_sample + self.trial_start_offset_samples
            end_sample = event_sample + self.trial_stop_offset_samples
            
            if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                window_data = raw_data[:, start_sample:end_sample]
                windows_data.append(window_data)
                windows_labels.append(labels[i])
        
        windows_data = np.array(windows_data)
        windows_labels = np.array(windows_labels)

# %%
# Step 8: Baseline Correction
# ---------------------------
#
# Subtract the pre-stimulus baseline (-100ms to 0ms) from each trial.
# This removes slow drifts and ensures we're measuring stimulus-evoked responses.
        
        # Baseline correction using pre-stimulus interval
        if BASELINE_CORRECT:
            baseline_end_idx = abs(self.trial_start_offset_samples) if self.trial_start_offset_samples < 0 else 0
            
            if baseline_end_idx > 0:
                # Calculate baseline mean for each channel and trial
                baseline_mean = np.mean(windows_data[:, :, :baseline_end_idx], axis=2, keepdims=True)
                windows_data = windows_data - baseline_mean
        
        return ManualWindowsDataset(windows_data, windows_labels)


# %%
# Plot a Sample Trial
# -------------------
#
# Let's visualize one EEG trial to verify the data quality.

import matplotlib.pyplot as plt

# Common channels available in both datasets
COMMON_CHANNELS = ['Fz', 'Pz', 'P3', 'P4', 'Oz']

# Load a small sample for visualization
print("\nLoading sample data for visualization...")
p3_subjects = get_dataset_subjects('P3', P3_DATA_DIR)
if len(p3_subjects) > 0:
    sample_preprocessor = OddballPreprocessor(COMMON_CHANNELS, dataset_type='P3')
    sample_data, sample_labels = process_subject_data(
        p3_subjects[0], P3_DATA_DIR, sample_preprocessor, dataset_type='P3'
    )
    
    if sample_data is not None and len(sample_data) > 0:
        # Plot first channel of first oddball trial
        oddball_idx = np.where(sample_labels == 1)[0][0]
        time_axis = np.linspace(TRIAL_START_OFFSET, TRIAL_DURATION, sample_data.shape[2])
        
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, sample_data[oddball_idx, 0, :])
        plt.axvline(x=0, color='r', linestyle='--', label='Stimulus onset')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'Sample EEG Trial - {COMMON_CHANNELS[0]} channel (Oddball stimulus)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Sample data shape: {sample_data.shape} (trials, channels, timepoints)")
        print(f"Label distribution: {np.sum(sample_labels==0)} standard, {np.sum(sample_labels==1)} oddball")

# %%
# Positional Encoding Utility
# ---------------------------
#
# Sine-cosine positional encodings added to the token sequence before entering
# the Transformer, enabling the model to represent temporal order.

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# %%
# Deep Learning Model: EEGConformer
# ----------------------------------
#
# Combines CNN layers (temporal + spatial) with a Transformer encoder and a
# lightweight classification head for ERP decoding.

import torch.nn.functional as F

# EEGConformer architecture parameters
CONFORMER_CONV_SPATIAL_DIM = 68
CONFORMER_CONV_TEMPORAL_DIM = 44
CONFORMER_EMBEDDING_DIM = 68
CONFORMER_NUM_HEADS = 4
CONFORMER_NUM_LAYERS = 5
DROPOUT_RATE = 0.18

class EEGConformer(nn.Module):
    """EEGConformer: CNN + Transformer for EEG classification."""
    
    def __init__(self, n_chans, n_outputs, n_times, 
                 conv_spatial_dim=40, conv_temporal_dim=25,
                 embedding_dim=40, num_heads=10, num_layers=3,
                 dropout=0.5, activation='gelu'):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.embedding_dim = embedding_dim
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, conv_temporal_dim, (1, 25), padding=(0, 12))
        self.temporal_bn = nn.BatchNorm2d(conv_temporal_dim)
        
        # Spatial convolution
        self.spatial_conv = nn.Conv2d(conv_temporal_dim, conv_spatial_dim, (n_chans, 1))
        self.spatial_bn = nn.BatchNorm2d(conv_spatial_dim)
        
        # Pooling and dropout
        self.avg_pool = nn.AvgPool2d((1, 4), (1, 4))
        self.dropout = nn.Dropout(dropout)
        
        # Calculate sequence length after convolutions
        seq_length = n_times // 4
        
        # Projection to embedding dimension
        self.projection = nn.Linear(conv_spatial_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim, n_outputs)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.elu(x)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, conv_spatial_dim)
        x = self.projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        x = self.classifier(x)
        return x


def create_model(n_channels, n_times):
    """Create EEGConformer model with configured hyperparameters."""
    return EEGConformer(
        n_chans=n_channels,
        n_outputs=2,  # Binary classification: oddball vs standard
        n_times=n_times,
        conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
        conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
        embedding_dim=CONFORMER_EMBEDDING_DIM,
        num_heads=CONFORMER_NUM_HEADS,
        num_layers=CONFORMER_NUM_LAYERS,
        dropout=DROPOUT_RATE,
        activation='gelu'
    )


# %%
# AS-MMD Domain Adaptation
# -------------------------
#
# Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD) aligns feature distributions
# between source (P3) and target (AVO) domains using three key components:
#
# 1. **Mixup augmentation**: Interpolates between samples to enhance generalization
# 2. **Prototype alignment**: Aligns class-wise feature representations
# 3. **MMD loss**: Minimizes distribution discrepancy in feature space
# %%
# Mixup and Augmentation Utilities
# --------------------------------
#
# Functions for mixup data augmentation and focal loss used in AS-MMD training.

# Label smoothing parameter (not used in current implementation but defined for completeness)
LABEL_SMOOTHING = 0.06

def mixup_data(x, y, alpha=0.4):
    """Perform mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def compute_focal_loss(scores, targets, gamma=2.0, alpha=0.25):
    """Compute focal loss for handling class imbalance."""
    ce_loss = F.cross_entropy(scores, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def mixup_criterion(pred, y_a, y_b, lam, gamma=2.0, alpha=0.5):
    """Compute mixup focal loss."""
    loss_a = compute_focal_loss(pred, y_a, gamma=gamma, alpha=alpha)
    loss_b = compute_focal_loss(pred, y_b, gamma=gamma, alpha=alpha)
    return lam * loss_a + (1 - lam) * loss_b
# %%
# Prototype-based Alignment Utilities
# -----------------------------------
#
# Functions to compute class prototypes and prototype alignment loss.

def compute_prototypes(features, labels, n_classes=2):
    """Compute class prototypes (mean features per class)."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    prototypes = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            proto = features[mask].mean(dim=0)
        else:
            proto = torch.zeros(features.size(1), device=features.device)
        prototypes.append(proto)

    return torch.stack(prototypes)


def compute_prototype_loss(features, labels, prototypes):
    """Compute prototype alignment loss."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    loss = 0.0
    n_samples = 0
    for i, label in enumerate(labels):
        proto = prototypes[label]
        dist = F.mse_loss(features[i], proto)
        loss += dist
        n_samples += 1

    return loss / max(1, n_samples)
# %%
# MMD Alignment Utility
# ---------------------
#
# Compute the RBF-kernel Maximum Mean Discrepancy for distribution alignment.

def compute_mmd_rbf(x, y, eps=1e-8):
    """Compute RBF-kernel Maximum Mean Discrepancy."""
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)
    
    with torch.no_grad():
        # Median heuristic for bandwidth
        z = torch.cat([x, y], dim=0)
        if z.size(0) > 1:
            dists = torch.cdist(z, z, p=2.0)
            sigma = torch.median(dists)
            sigma = torch.clamp(sigma, min=eps)
        else:
            sigma = torch.tensor(1.0, device=z.device)
    
    gamma = 1.0 / (2.0 * (sigma ** 2) + eps)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2.0) ** 2)
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2.0) ** 2)
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2.0) ** 2)
    
    m = x.size(0)
    n = y.size(0)
    if m <= 1 or n <= 1:
        return torch.tensor(0.0, device=x.device)
    
    # Unbiased estimate
    mmd = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    mmd += (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    mmd -= 2.0 * k_xy.mean()
    
    return mmd
# %%
# Symmetric Weighting Schedule
# ----------------------------
#
# Compute domain weights and warmup schedule based on relative dataset sizes.

# Training parameters
MAX_EPOCHS = 100

def get_symmetric_adjustments(n_train_a, n_train_b):
    """Compute symmetric domain weights based on dataset sizes."""
    n_train_a = max(1, n_train_a)
    n_train_b = max(1, n_train_b)
    ratio_ab = n_train_a / float(n_train_b)
    
    # Gentle weight scaling using square root
    w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 3.0, 1.0, 12.0))
    
    # MMD weight based on domain size difference
    overall_ratio = max(ratio_ab, 1.0 / ratio_ab)
    lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.3 if overall_ratio < 4.0 else 0.45)
    
    # Prototype loss weight
    lambda_proto = 0.5 if overall_ratio < 4.0 else 0.8
    
    # Warmup epochs
    warmup = max(20, min(40, int(0.4 * MAX_EPOCHS)))
    
    return w_small, lambda_mmd, lambda_proto, warmup
# %%
# Stratified Sampling Utility
# ---------------------------
#
# Helper for balanced per-subject sampling while preserving class ratios.

# Fixed random seeds for reproducibility
SEEDS = [42, 123, 456, 789, 321]

def stratified_sample_trials(data, labels, n_trials, random_seed):
    """Stratified sampling of trials maintaining class balance."""
    np.random.seed(random_seed)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2 or n_trials >= len(data):
        return data, labels
    
    sampled_indices = []
    for label in unique_labels:
        label_mask = labels == label
        available = np.sum(label_mask)
        
        # Proportional sampling
        proportion = available / len(data)
        n_for_label = max(1, int(n_trials * proportion))
        
        if len(sampled_indices) + n_for_label > n_trials:
            n_for_label = n_trials - len(sampled_indices)
        
        if n_for_label > 0 and available >= n_for_label:
            label_indices = np.where(label_mask)[0]
            selected = np.random.choice(label_indices, size=n_for_label, replace=False)
            sampled_indices.extend(selected)
    
    sampled_indices = np.array(sampled_indices)
    return data[sampled_indices], labels[sampled_indices]


# %%
# Combined Loaders (P3 + AVO)
# ---------------------------
#
# Load both datasets, apply stratified sampling per subject, and combine.



def load_combined_arrays(channels):
    """Load and combine P3 and AVO datasets with stratified sampling."""
    X_list = []
    y_list = []
    src_list = []

    for dataset_name in ['P3', 'AVO']:
        if dataset_name == 'P3':
            subjects = get_dataset_subjects('P3', P3_DATA_DIR)
            dataset_obj = P3_DATA_DIR
            n_trials_ps = TRIALS_PER_SUBJECT_P3
        else:
            avo_dataset = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            dataset_obj = avo_dataset
            n_trials_ps = TRIALS_PER_SUBJECT_AVO

        preproc = OddballPreprocessor(channels, dataset_type=dataset_name)

        for s in subjects:
            data, labels = process_subject_data(s, dataset_obj, preproc, dataset_type=dataset_name)
            if data is None or labels is None or len(data) == 0:
                continue
            
            # Stratified sampling to target budget
            if len(data) > n_trials_ps:
                data, labels = stratified_sample_trials(data, labels, n_trials_ps, SEEDS[0])
            
            X_list.append(data)
            y_list.append(labels)
            src_list.append(np.array([dataset_name] * len(labels)))

    if not X_list:
        raise RuntimeError("No valid data loaded")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    src_all = np.concatenate(src_list, axis=0)

    return X_all, y_all, src_all


# %%
# AS-MMD Training Loop
# --------------------
#
# Joint training combining classification, mixup, prototype alignment, and MMD.

from torch.utils.data import TensorDataset, DataLoader

# Model and training parameters
BATCH_SIZE = 22
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2.5e-4
EARLY_STOPPING_PATIENCE = 10

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def asmmd_train_fold(Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                     Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                     n_channels, n_times, seed=42):
    """Train a single fold with AS-MMD method."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build data loaders
    def make_loader(X, y, shuffle): 
        return DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), 
                         batch_size=BATCH_SIZE, shuffle=shuffle)
    train_loader_p3 = make_loader(Xtr_p3, ytr_p3, True)
    val_loader_p3 = make_loader(Xva_p3, yva_p3, False)
    train_loader_avo = make_loader(Xtr_avo, ytr_avo, True)
    val_loader_avo = make_loader(Xva_avo, yva_avo, False)

    # Initialize model, optimizer, scheduler
    model = create_model(n_channels, n_times).to(DEVICE)
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Domain size analysis
    n_train_avo, n_train_p3 = len(Xtr_avo), len(Xtr_p3)
    w_small, lambda_mmd_target, lambda_proto_target, warmup_epochs = get_symmetric_adjustments(n_train_avo, n_train_p3)
    small_domain = 'P3' if n_train_p3 <= n_train_avo else 'AVO'
    large_domain = 'AVO' if small_domain == 'P3' else 'P3'

    # Early stopping & prototypes
    best_val, best_state, patience_cnt = 0.0, None, 0
    large_prototypes = None

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup schedule
        alpha = min(1.0, epoch / max(1, warmup_epochs))
        w_small_cur, w_large_cur = 1.0 + alpha * (w_small - 1.0), 1.0
        lambda_mmd, lambda_proto = alpha * lambda_mmd_target, alpha * lambda_proto_target

        # Domain iterators
        train_loaders = {'P3': train_loader_p3, 'AVO': train_loader_avo}
        itr_large = iter(train_loaders[large_domain])
        itr_small = iter(train_loaders[small_domain]) if len(train_loaders[small_domain]) > 0 else None

        # Training steps
        while True:
            try:
                xb_large, yb_large = next(itr_large)
            except StopIteration:
                break

            # Get small domain batch (with cycling)
            xb_small, yb_small = None, None
            if itr_small is not None:
                try:
                    xb_small, yb_small = next(itr_small)
                except StopIteration:
                    itr_small = iter(train_loaders[small_domain])
                    xb_small, yb_small = next(itr_small) if len(train_loaders[small_domain]) > 0 else (None, None)

            optimizer.zero_grad()

            # Large domain forward & loss
            x_large, y_large = augment_data(normalize_data(xb_large)).to(DEVICE), yb_large.to(DEVICE)
            scores_large = model(x_large)
            loss_large = F.cross_entropy(scores_large, y_large)

            # Update prototypes (exponential moving average)
            with torch.no_grad():
                current_prototypes = compute_prototypes(scores_large.detach(), y_large, n_classes=2)
                large_prototypes = current_prototypes if large_prototypes is None else 0.9 * large_prototypes + 0.1 * current_prototypes

            # Small domain forward & losses
            loss_small, loss_proto, scores_small = torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE), None
            if xb_small is not None:
                x_small, y_small = normalize_data(xb_small).to(DEVICE), yb_small.to(DEVICE)
                x_mixed, y_a, y_b, lam = mixup_data(x_small, y_small, alpha=0.4)
                scores_small = model(augment_data(x_mixed))
                loss_small = mixup_criterion(scores_small, y_a, y_b, lam, gamma=2.0, alpha=0.5)
                
                # Prototype alignment
                if large_prototypes is not None and lambda_proto > 0:
                    loss_proto = compute_prototype_loss(model(augment_data(x_small)), y_small, large_prototypes)

            # MMD alignment loss
            loss_align = torch.tensor(0.0, device=DEVICE)
            if scores_small is not None and lambda_mmd > 0.0:
                try:
                    scores_orig_small = model(augment_data(normalize_data(xb_small).to(DEVICE)))
                    b = min(scores_large.size(0), scores_orig_small.size(0))
                    loss_align = compute_mmd_rbf(scores_large[:b].detach(), scores_orig_small[:b].detach())
                except:
                    pass

            # Combined loss & optimization
            total_loss = w_large_cur * loss_large + w_small_cur * loss_small + lambda_mmd * loss_align + lambda_proto * loss_proto
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        scheduler.step()

        # Validation & early stopping
        # if epoch % 10 == 0:
        p3_val, avo_val = evaluate_domain(model, val_loader_p3, DEVICE), evaluate_domain(model, val_loader_avo, DEVICE)
        print(f"Epoch {epoch}: Val(P3)={p3_val:.3f} | Val(AVO)={avo_val:.3f}")

        small_val = p3_val if small_domain == 'P3' else avo_val
        if small_val > best_val + 1e-4:
            best_val, best_state, patience_cnt = small_val, model.state_dict(), 0
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# %%
# Model Evaluation Utilities
# --------------------------
#
# Functions for model evaluation: simple accuracy and detailed metrics.

from sklearn.metrics import confusion_matrix, roc_auc_score

def evaluate_domain(model, loader, device):
    """Evaluate model accuracy on a specific domain."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = normalize_data(x).to(device), y.to(device)
            _, pred = model(x).max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_with_metrics(model, loader, device):
    """Evaluate with detailed metrics: accuracy, precision, recall, F1, AUC."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = normalize_data(x).to(device), y.to(device)
            scores = model(x)
            all_preds.extend(scores.max(1)[1].cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(scores, dim=1)[:, 1].cpu().numpy())
    
    all_preds, all_targets, all_probs = map(np.array, [all_preds, all_targets, all_probs])
    
    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy, precision, recall, f1 = np.mean(all_preds == all_targets), 0.0, 0.0, 0.0
    
    try:
        auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    except:
        auc = 0.5
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }


# %%
# Nested Cross-Validation and Execution
# -------------------------------------
#
# Nested CV to robustly evaluate AS-MMD across multiple splits.

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

# Cross-validation parameters
NESTED_CV_OUTER_FOLDS = 5
NESTED_CV_REPEATS = 5

# Train/val/test split
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1

def run_nested_cv_asmmd(channels):
    """Run nested cross-validation with AS-MMD."""
    print("Loading datasets...")
    X_all, y_all, src_all = load_combined_arrays(channels)
    print(f"Total: {len(X_all)} (P3: {np.sum(src_all=='P3')}, AVO: {np.sum(src_all=='AVO')})")
    
    n_channels, n_times = X_all.shape[1], X_all.shape[2]
    fold_results = []
    
    for repeat in range(NESTED_CV_REPEATS):
        print(f"\n{'='*60}\nRepeat {repeat + 1}/{NESTED_CV_REPEATS}\n{'='*60}")
        cv = StratifiedKFold(n_splits=NESTED_CV_OUTER_FOLDS, shuffle=True, 
                            random_state=SEEDS[repeat % len(SEEDS)])
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            print(f"\nFold {fold_idx + 1}/{NESTED_CV_OUTER_FOLDS}")
            X_tr_fold, y_tr_fold, src_tr_fold = X_all[train_idx], y_all[train_idx], src_all[train_idx]
            X_te_fold, y_te_fold, src_te_fold = X_all[test_idx], y_all[test_idx], src_all[test_idx]
            
            # Split train/val
            train_ratio = TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE) if (TRAIN_SIZE + VAL_SIZE) > 0 else 0.875
            tr_idx, va_idx = train_test_split(np.arange(len(X_tr_fold)), train_size=train_ratio, 
                                             stratify=y_tr_fold, random_state=42)
            
            # Per-domain splits (helper function)
            def get_domain_data(X, y, src, indices, domain):
                mask = (src == domain)
                idx = np.intersect1d(np.where(mask)[0], indices)
                return X[idx], y[idx]
            
            Xtr_p3, ytr_p3 = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, tr_idx, 'P3')
            Xtr_avo, ytr_avo = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, tr_idx, 'AVO')
            Xva_p3, yva_p3 = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, va_idx, 'P3')
            Xva_avo, yva_avo = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, va_idx, 'AVO')
            
            # Train model
            model = asmmd_train_fold(Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                                    Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                                    n_channels, n_times, seed=SEEDS[0])
            
            # Evaluate on test set
            def eval_domain(mask):
                if not np.any(mask):
                    return {'accuracy': 0.0, 'auc': 0.5}, 0
                loader = DataLoader(TensorDataset(torch.FloatTensor(X_te_fold[mask]), 
                                                  torch.LongTensor(y_te_fold[mask])),
                                   batch_size=BATCH_SIZE, shuffle=False)
                return evaluate_with_metrics(model, loader, DEVICE), int(np.sum(mask))
            
            m_p3, n_p3 = eval_domain(src_te_fold == 'P3')
            m_avo, n_avo = eval_domain(src_te_fold == 'AVO')
            acc_overall = (m_p3['accuracy'] * n_p3 + m_avo['accuracy'] * n_avo) / max(1, n_p3 + n_avo)
            
            print(f"  P3 Test: Acc={m_p3['accuracy']:.3f}, AUC={m_p3['auc']:.3f} (n={n_p3})")
            print(f"  AVO Test: Acc={m_avo['accuracy']:.3f}, AUC={m_avo['auc']:.3f} (n={n_avo})")
            print(f"  Overall: Acc={acc_overall:.3f}")
            
            fold_results.append({
                'repeat': repeat + 1, 'fold': fold_idx + 1, 'overall_accuracy': acc_overall,
                'p3_accuracy': m_p3['accuracy'], 'p3_auc': m_p3['auc'],
                'avo_accuracy': m_avo['accuracy'], 'avo_auc': m_avo['auc'],
            })
    
    # Calculate statistics
    df = pd.DataFrame(fold_results)
    return {
        'mean_accuracy': float(df['overall_accuracy'].mean()),
        'std_accuracy': float(df['overall_accuracy'].std()),
        'p3_mean_accuracy': float(df['p3_accuracy'].mean()),
        'p3_std_accuracy': float(df['p3_accuracy'].std()),
        'p3_mean_auc': float(df['p3_auc'].mean()),
        'p3_std_auc': float(df['p3_auc'].std()),
        'avo_mean_accuracy': float(df['avo_accuracy'].mean()),
        'avo_std_accuracy': float(df['avo_accuracy'].std()),
        'avo_mean_auc': float(df['avo_auc'].mean()),
        'avo_std_auc': float(df['avo_auc'].std()),
        'detailed_fold_results': fold_results
    }

# Execute nested cross-validation
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("Starting Nested Cross-Validation with AS-MMD")

results = run_nested_cv_asmmd(COMMON_CHANNELS)

print("\n" + "="*70 + "\nFINAL RESULTS\n" + "="*70)
print(f"Overall Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
print(f"\nP3 Dataset:")
print(f"  Accuracy: {results['p3_mean_accuracy']:.4f} ± {results['p3_std_accuracy']:.4f}")
print(f"  AUC: {results['p3_mean_auc']:.4f} ± {results['p3_std_auc']:.4f}")
print(f"\nAVO Dataset:")
print(f"  Accuracy: {results['avo_mean_accuracy']:.4f} ± {results['avo_std_accuracy']:.4f}")
print(f"  AUC: {results['avo_mean_auc']:.4f} ± {results['avo_std_auc']:.4f}")
print("="*70)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pd.DataFrame(results['detailed_fold_results']).to_csv(f'asmmd_detailed_results_{timestamp}.csv', index=False)
print(f"\nDetailed results saved to: asmmd_detailed_results_{timestamp}.csv")

summary_stats = {k: v for k, v in results.items() if k != 'detailed_fold_results'}
pd.DataFrame([summary_stats]).to_csv(f'asmmd_summary_{timestamp}.csv', index=False)
print(f"Summary statistics saved to: asmmd_summary_{timestamp}.csv")

# %%
# Discussion and Hyperparameters
# -------------------------------
#
# **Key Hyperparameters:**
#
# - **Learning Rate (0.001)**: Moderate learning rate with Adamax optimizer
# - **Batch Size (22)**: Small batches help with domain adaptation
# - **Dropout (0.18)**: Prevents overfitting in the transformer layers
# - **Early Stopping Patience (10)**: Allows sufficient training time
# - **Domain Weights**: Automatically adjusted based on dataset size ratio
# - **MMD Weight (0.2-0.4)**: Balances classification and domain alignment
# - **Prototype Weight (0.5-0.8)**: Encourages discriminative class alignment
#
# **References:**
#
# - Song et al. (2020). "EEGConformer: Convolutional Transformer for EEG Decoding"
# - Long et al. (2015). "Learning Transferable Features with Deep Adaptation Networks"
# - Zhang et al. (2018). "Mixup: Beyond Empirical Risk Minimization"

# %%
# Conclusion
# ----------
'''
In this tutorial, we demonstrated a complete workflow for EEG P3 transfer learning
using AS-MMD on two public oddball datasets (ERP CORE P3 and AVO). By combining
prototype-based alignment, mixup augmentation, and MMD alignment on the logit space
within an EEGConformer backbone, we achieved robust cross-dataset performance while
keeping the inference-time model unchanged. The nested cross-validation protocol
provides reliable estimates across subjects and splits. For practical use, consider
tuning the domain weights, model capacity, and data budgets per subject, and extend
this pipeline to other ERP components or datasets. For methodological details and
ablations, see the accompanying paper: https://arxiv.org/abs/2510.21969
'''
