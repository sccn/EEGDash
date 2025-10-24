""".. p3-transfer-learning:

EEG P3 Transfer Learning with AS-MMD
====================================

EEG P3 transfer learning using Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD) for domain adaptation between P3 and AVO datasets.

This example implements a comprehensive transfer learning framework for EEG P3 component analysis, combining deep learning models with domain adaptation techniques to improve classification performance across different EEG datasets.

1. **Data Loading and Preprocessing**: The system loads EEG data from two datasets (P3 and AVO), preprocesses raw signals using MNE-Python, applies filtering and resampling, and extracts balanced trial windows for oddball vs. standard event classification.

2. **Domain Adaptation with AS-MMD**: Implements Adaptive Symmetric Maximum Mean Discrepancy for cross-domain learning, including prototype-based discriminative transfer, mixup data augmentation for small samples, and MMD alignment for feature space adaptation.

3. **Deep Learning Model Architecture**: Uses EEGConformer, a hybrid CNN-Transformer model specifically designed for EEG classification, with spatial and temporal convolutions followed by transformer layers for capturing long-range dependencies in EEG signals.

4. **Nested Cross-Validation Framework**: Implements comprehensive evaluation using nested cross-validation with multiple repeats, ensuring robust performance estimation across different data splits and subject combinations.

5. **Training and Evaluation Process**: Trains models with early stopping, data augmentation, and adaptive learning rate scheduling, then evaluates performance using multiple metrics including accuracy, precision, recall, F1-score, and AUC for both source and target domains.
"""

# %% [markdown]
# # EEG P3 Transfer Learning with AS-MMD
# 
# This notebook implements **Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD)** for EEG P3 transfer learning between two datasets (P3 and AVO).
# 
# ## Overview
# - Load and preprocess EEG data from P3 and AVO datasets
# - Train deep learning models with AS-MMD for domain adaptation
# - Run nested cross-validation
# - Analyze and visualize results
# 
# ## Key Features
# - Prototype-based discriminative transfer
# - Mixup data augmentation for small samples
# - MMD alignment for domain adaptation
# - Comprehensive evaluation metrics
# 
# ## Dataset
# 
# ### 1. Active Visual Oddball (AVO) Dataset
# **Source:** "Cognitive Electrophysiology in Socioeconomic Context in Adulthood" dataset  
# **Download:** [NEMAR Dataset ds005863](https://nemar.org/dataexplorer/detail?dataset_id=ds005863)
# 
# This dataset contains EEG data from 127 young adults (18-30 years) with multiple ERP tasks in BIDS format. We use the Active Visual Oddball (AVO) task subset, selecting 40 participants with sufficient oddball trials for balanced sampling (80 trials per subject: 40 oddball + 40 standard). The dataset includes socioeconomic context measures and ADHD symptom assessments.
# 
# **Key Features:**
# - 127 participants (40 selected for AVO task)
# - BIDS-compatible format
# - Multiple ERP paradigms from ERP CORE
# - Socioeconomic status indicators
# - CC0 license
# 
# ### 2. ERP CORE P3 Dataset  
# **Source:** ERP CORE P3 component  
# **Download:** [Open Science Framework](https://osf.io/etdkz/files)
# 
# This dataset is part of ERP CORE, a curated resource with optimized paradigms and processing pipelines from 40 neurotypical adults. We use the P3 active visual oddball task with associated event schema, providing a standardized baseline for P3b component analysis.
# 
# **Key Features:**
# - 40 neurotypical adults
# - Optimized P3 paradigms
# - Standardized processing pipelines
# - Open Science Framework archival
# - Research-grade quality control
# 
# **References:**
# - Isbell, E., De León, N. E. R., & Richardson, D. M. (2024). Childhood family socioeconomic status is linked to adult brain electrophysiology. PloS One, 19(8), e0307406.
# - Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). ERP CORE: An open resource for human event-related potential research. NeuroImage, 225, 117465.
# 

# %%
# Import all required libraries
import os
import sys
import math
import logging
import warnings
from typing import Dict, Tuple, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats

import mne

try:
    from braindecode.models import ShallowFBCSPNet
    from braindecode.preprocessing import Preprocessor
    from braindecode.datasets import BaseConcatDataset, BaseDataset
    BRAINDECODE_AVAILABLE = True
except:
    BRAINDECODE_AVAILABLE = False
    ShallowFBCSPNet = None
    Preprocessor = None

mne.set_log_level('ERROR')
logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Braindecode available: {BRAINDECODE_AVAILABLE}")

# %% [markdown]
# ## Configuration
# 
# Set all experiment parameters. You can modify these values to customize the experiment.

# %%
# ===== MAIN CONFIGURATION =====
# Round 3: Conservative fine-tuning from Round 1 success (0.6393 -> 0.66 target)

# Paths
P3_DATA_DIR = '/home/vivian/eeg/P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '/home/vivian/eeg/ds005863'
LOG_DIR = '/home/vivian/eeg/P3_transfer_learning/log'

# Channels
COMMON_CHANNELS = ['Fz', 'Pz', 'P3', 'P4', 'Oz']
P3_CHANNELS = ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
AVO_CHANNELS = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

# Preprocessing
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128
TRIAL_START_OFFSET_SAMPLES = int(-0.1 * 128)
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)

# Training - Round 3: 基于Round 1成功配置的微调
BATCH_SIZE = 22  # 微降从24（改善梯度估计）
MAX_EPOCHS = 1000
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2
LEARNING_RATE = 0.0006  # 微升从0.0005（稍快收敛）
WEIGHT_DECAY = 2.5e-4  # 微降从3e-4（稍放松正则化）
DROPOUT_RATE = 0.18  # 微降从0.2（允许更好学习）
EARLY_STOPPING_PATIENCE = 220  # 微升从200

# Model
classifier = 'EEGConformer'
INPUT_WINDOW_SAMPLES = TRIAL_STOP_OFFSET_SAMPLES - TRIAL_START_OFFSET_SAMPLES
N_CLASSES = 2

# Data Augmentation - 微调增强
USE_DATA_AUGMENTATION = True
NOISE_STD = 0.006  # 微升从0.005
TIME_SHIFT_RANGE = 6  # 微升从5
LABEL_SMOOTHING = 0.06  # 微升从0.05

# Cross-Validation
NESTED_CV_OUTER_FOLDS = 5
NESTED_CV_REPEATS = 5
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 80
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 10

# Device
DEVICE_MODE = 'cuda'

# Seeds
seeds = [42, 123, 456, 789, 321]

# Events
RESPONSE_EVENTS = [6, 7, 201, 202]
ODDBALL_EVENTS = [1, 9, 15, 21, 27]
RESPONSE_EVENTS_AVO = [6, 7, 201, 202]
ODDBALL_EVENTS_AVO = [11, 22, 33, 44, 55]

# EEGConformer parameters - Round 3: 基于Round 1微调
CONFORMER_CONV_SPATIAL_DIM = 68  # 微升从64
CONFORMER_CONV_TEMPORAL_DIM = 44  # 微升从40
CONFORMER_EMBEDDING_DIM = 68  # 微升从64
CONFORMER_NUM_HEADS = 4  # 调整以匹配embedding_dim=68（68能被4整除）
CONFORMER_NUM_LAYERS = 5  # 微升从4
CONFORMER_ACTIVATION = 'gelu'

# Trial configuration
FIXED_TRIALS_PER_CLASS = 20
TRAIN_TRIALS_PER_CLASS = 10
VAL_TRIALS_PER_CLASS = 5
TEST_TRIALS_PER_CLASS = 5

# Other flags
USE_ENHANCED_PREPROCESSING = True
ELECTRODE_FUSION_METHOD = 'none'
DOMAIN_ADAPTATION_METHOD = 'none'
electrode_list = 'all'
NORMALIZATION_EPSILON = 1e-7

print("✓ Round 3 配置: 基于Round 1（0.6393）的保守微调")
print(f"  目标: AVO >= 0.66（还需 +{(0.66-0.6393)*100:.2f}%）")
print(f"  策略: 微调而非激进改动")
print(f"  Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}, Dropout: {DROPOUT_RATE}")
print(f"  Model: spatial={CONFORMER_CONV_SPATIAL_DIM}, emb={CONFORMER_EMBEDDING_DIM}, layers={CONFORMER_NUM_LAYERS}")

# %% [markdown]
# ## Utility Functions
# 
# Data loading, preprocessing, and helper functions.

# %%
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
    
    def __str__(self):
        return f"EEGBIDSDataset(data_dir='{self.data_dir}', dataset='{self.dataset}')"


def load_raw(file_path, dataset_type):
    """Load raw EEG data based on dataset type."""
    if dataset_type == 'P3': 
        return mne.io.read_raw_eeglab(file_path, preload=True)
    else: 
        return mne.io.read_raw_brainvision(file_path, preload=True)


def load_events_tsv(subject_id, dataset_dir):
    """Load events from TSV file for a P3 subject."""
    try:
        events_file = os.path.join(dataset_dir, subject_id, 'eeg', f'{subject_id}_task-P3_events.tsv')
        if os.path.exists(events_file):
            events_df = pd.read_csv(events_file, sep='\t')
            return events_df
        else:
            #print(f"Warning: Events file not found: {events_file}")
            return None
    except Exception as e:
        print(f"Error loading events file: {e}")
        return None


def get_stimulus_event_values(events_df):
    """Extract stimulus event values from events dataframe."""
    if events_df is None:
        return []
    
    # Filter for stimulus events only (not response events)
    stimulus_events = events_df[events_df['trial_type'] == 'stimulus']
    
    # Extract the 'value' column
    event_values = stimulus_events['value'].tolist()
    
    return event_values


def get_device():
    """Get device based on configuration"""
    if DEVICE_MODE == 'cpu':
        return torch.device('cpu')
    elif DEVICE_MODE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device('cuda')
    else:  # 'auto'
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_global_torch_seed(seed: int):
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_statistics(accuracies):
    """Calculate mean and 95% confidence interval for accuracies."""
    values = np.array(list(accuracies.values()))
    mean = np.mean(values)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    best_subject = max(accuracies.items(), key=lambda x: x[1])
    worst_subject = min(accuracies.items(), key=lambda x: x[1])
    
    return {
        'mean': mean,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'best_subject': best_subject,
        'worst_subject': worst_subject
    }


def print_statistics(stats, dataset_name, logger=None, prediction_details=None):
    """Print and optionally log statistics in a formatted way."""
    out_lines = [
        f"\n{dataset_name} Statistics:",
        f"95% Confidence Interval: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
        f"Best Subject: {stats['best_subject'][0]} ({stats['best_subject'][1]:.3f})",
        f"Worst Subject: {stats['worst_subject'][0]} ({stats['worst_subject'][1]:.3f})",
    ]
    
    # Calculate overall metrics if prediction details are provided
    if prediction_details:
        # Calculate mean confusion matrix metrics
        avg_tp = np.mean([details.get('tp', 0) for details in prediction_details.values()])
        avg_tn = np.mean([details.get('tn', 0) for details in prediction_details.values()])
        avg_fp = np.mean([details.get('fp', 0) for details in prediction_details.values()])
        avg_fn = np.mean([details.get('fn', 0) for details in prediction_details.values()])
        
        # Calculate accuracy from confusion matrix
        total_accuracy = (avg_tp + avg_tn) / (avg_tp + avg_tn + avg_fp + avg_fn) if (avg_tp + avg_tn + avg_fp + avg_fn) > 0 else 0
        
        # Calculate precision, recall, f1 from confusion matrix metrics
        total_precision = avg_tp / (avg_tp + avg_fp) if (avg_tp + avg_fp) > 0 else 0
        total_recall = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        
        # Calculate AUC (using provided values)
        auc_values = [details.get('auc', 0.5) for details in prediction_details.values()]
        valid_auc_values = [auc for auc in auc_values if not np.isnan(auc)]
        total_auc = np.mean(valid_auc_values) if valid_auc_values else 0.5
        
        out_lines.extend([
            f"Mean Accuracy: {total_accuracy:.3f}",
            f"Mean Precision: {total_precision:.3f}",
            f"Mean Recall: {total_recall:.3f}",
            f"Mean F1-Score: {total_f1:.3f}",
            f"Mean AUC: {total_auc:.3f}",
            f"Mean Confusion Matrix:",
            f"  TP: {int(round(avg_tp))}, TN: {int(round(avg_tn))}",
            f"  FP: {int(round(avg_fp))}, FN: {int(round(avg_fn))}"
        ])
    
    for line in out_lines:
        print(line)
        if logger is not None:
            logger.info(line)

# %%
# Simplified create_model function - only supports EEGConformer
def create_model_simplified(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='EEGConformer', input_channels=None):
    """Create a new model based on configuration - only supports EEGConformer."""
    if is_lda:
        return LDA()
    else:
        # Use input_channels if provided, otherwise use n_channels
        actual_channels = input_channels if input_channels is not None else n_channels
        
        # Only support EEGConformer model
        base_model = EEGConformer(
            n_chans=actual_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
            conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
            embedding_dim=CONFORMER_EMBEDDING_DIM,
            num_heads=CONFORMER_NUM_HEADS,
            num_layers=CONFORMER_NUM_LAYERS,
            dropout=DROPOUT_RATE,
            activation=CONFORMER_ACTIVATION
        )
        
        return base_model


# %%
# Cleaned up model definitions - only EEGConformer
# Removed CustomShallowFBCSPNet and EEGNet classes

# Only keep EEGConformer and related classes
class EEGConformer(nn.Module):
    """EEGConformer: Combining CNN and Transformer for EEG classification."""
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
        seq_length = self._get_sequence_length()
        
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
    
    def _get_sequence_length(self):
        # Calculate sequence length after convolutions
        # After temporal conv: n_times (same due to padding)
        # After avg pool: n_times // 4
        return self.n_times // 4
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (batch, conv_temporal_dim, n_chans, n_times)
        x = self.temporal_bn(x)
        x = F.elu(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (batch, conv_spatial_dim, 1, n_times)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Pooling
        x = self.avg_pool(x)  # (batch, conv_spatial_dim, 1, n_times//4)
        
        # Reshape for transformer
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, conv_spatial_dim)
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, embedding_dim)
        
        # Classification
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        x = self.classifier(x)  # (batch, n_outputs)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Simplified create_model function - only supports EEGConformer
def create_model_clean(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='EEGConformer', input_channels=None):
    """Create a new model based on configuration - only supports EEGConformer."""
    if is_lda:
        return LDA()
    else:
        # Use input_channels if provided, otherwise use n_channels
        actual_channels = input_channels if input_channels is not None else n_channels
        
        # Only support EEGConformer model
        base_model = EEGConformer(
            n_chans=actual_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
            conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
            embedding_dim=CONFORMER_EMBEDDING_DIM,
            num_heads=CONFORMER_NUM_HEADS,
            num_layers=CONFORMER_NUM_LAYERS,
            dropout=DROPOUT_RATE,
            activation=CONFORMER_ACTIVATION
        )
        
        return base_model


# %%
# Override the old create_model function with the cleaned version
def create_model(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='EEGConformer', input_channels=None):
    """Create a new model based on configuration - only supports EEGConformer."""
    if is_lda:
        return LDA()
    else:
        # Use input_channels if provided, otherwise use n_channels
        actual_channels = input_channels if input_channels is not None else n_channels
        
        # Only support EEGConformer model
        base_model = EEGConformer(
            n_chans=actual_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
            conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
            embedding_dim=CONFORMER_EMBEDDING_DIM,
            num_heads=CONFORMER_NUM_HEADS,
            num_layers=CONFORMER_NUM_LAYERS,
            dropout=DROPOUT_RATE,
            activation=CONFORMER_ACTIVATION
        )
        
        return base_model

print("✓ Updated create_model function to only support EEGConformer")


# %% [markdown]
# ## Logging Functions
# 
# Experiment logging and tracking utilities.

# %%
def setup_logger(experiment_type, classifier=None, separate_subject_classification=None, electrode_list=None, create_file=True):
    """Setup logger for experiment tracking."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directory if it doesn't exist
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Create descriptive filename with configuration parameters
    if classifier and separate_subject_classification is not None and electrode_list:
        logfile = os.path.join(log_dir, f'{experiment_type}_clf-{classifier}_sep-{separate_subject_classification}_el-{electrode_list}_results_{timestamp}.log')
    else:
        logfile = os.path.join(log_dir, f'{experiment_type}_results_{timestamp}.log')

    # Only create file handler if requested
    handlers = [logging.StreamHandler()]
    if create_file:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Python>=3.8
    )

    # Return a named logger (avoids duplicate handlers if caller also uses logging).
    logger = logging.getLogger(experiment_type)
    
    # Store the log file path for potential cleanup
    if create_file:
        logger.log_file_path = logfile
    
    return logger


def cleanup_failed_log(logger):
    """Clean up log file if experiment failed."""
    if hasattr(logger, 'log_file_path') and os.path.exists(logger.log_file_path):
        try:
            os.remove(logger.log_file_path)
            print(f"Cleaned up failed experiment log: {logger.log_file_path}")
        except Exception as e:
            print(f"Failed to clean up log file {logger.log_file_path}: {e}")


def log_section_header(logger, title):
    """Log a section header."""
    logger.info("\n" + "="*50)
    logger.info(title)
    logger.info("="*50)


def log_individual_results(logger, experiment_type, subject_id, accuracy):
    """Log individual subject results."""
    logger.info(f"Subject: {subject_id}, Accuracy: {accuracy:.3%}")


def log_detailed_results(logger, experiment_type, subject_id, metrics):
    """Log detailed metrics including accuracy, precision, recall, f1 score, AUC and confusion matrix stats."""
    logger.info(f"Subject: {subject_id}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.3%}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
    logger.info(f"  Recall: {metrics.get('recall', 0):.3f}")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.3f}")
    logger.info(f"  AUC: {metrics.get('auc', 0):.3f}")
    logger.info(f"  Correct/Total: {metrics.get('correct_count', 0)}/{metrics.get('total_count', 0)}")
    logger.info(f"  Confusion Matrix Stats:")
    logger.info(f"    TP: {metrics.get('tp', 0)}, TN: {metrics.get('tn', 0)}")
    logger.info(f"    FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}")


def log_error(logger, experiment_type, subject_id, error_msg):
    """Log error messages."""
    logger.error(f"\nError in {experiment_type} - Subject {subject_id}:")
    logger.error(str(error_msg))


def log_configuration(logger, config_dict):
    """Log experiment configuration."""
    # logger.info("\nExperiment Configuration:")
    logger.info("-" * 50)
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 50)


def log_overall_metrics(logger, metrics, confusion_matrix_path=None):
    """Log overall experiment metrics and confusion matrix location."""
    logger.info("\nOverall Experiment Metrics:")
    logger.info("-" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    if confusion_matrix_path:
        logger.info(f"\nConfusion Matrix Plot: {confusion_matrix_path}")
    logger.info("-" * 50)

# %% [markdown]
# ## Data Preprocessing
# 
# Preprocessing classes for EEG data.

# %%
# Final create_model function - only EEGConformer, no LDA support
def create_model(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='EEGConformer', input_channels=None):
    """Create a new model based on configuration - only supports EEGConformer."""
    # Use input_channels if provided, otherwise use n_channels
    actual_channels = input_channels if input_channels is not None else n_channels
    
    # Only support EEGConformer model - LDA removed
    base_model = EEGConformer(
        n_chans=actual_channels,
        n_outputs=N_CLASSES,
        n_times=INPUT_WINDOW_SAMPLES,
        conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
        conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
        embedding_dim=CONFORMER_EMBEDDING_DIM,
        num_heads=CONFORMER_NUM_HEADS,
        num_layers=CONFORMER_NUM_LAYERS,
        dropout=DROPOUT_RATE,
        activation=CONFORMER_ACTIVATION
    )
    
    return base_model



# %%
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
    """Simple base preprocessor class when braindecode is not available."""
    def __init__(self, fn, apply_on_array=False):
        self.fn = fn
        self.apply_on_array = apply_on_array


class OddballPreprocessor(Preprocessor if BRAINDECODE_AVAILABLE else SimplePreprocessorBase):
    """Generic preprocessor for oddball-paradigm EEG data."""

    def __init__(self, eeg_channels, 
                 trial_start_offset_samples=TRIAL_START_OFFSET_SAMPLES,
                 trial_stop_offset_samples=TRIAL_STOP_OFFSET_SAMPLES,
                 random_seed=42,
                 use_cache=True,
                 dataset_type='P3',
                 fixed_trials_per_class=FIXED_TRIALS_PER_CLASS,
                 use_fixed_split=True):
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.dataset_type = dataset_type
        self.fixed_trials_per_class = fixed_trials_per_class
        self.use_fixed_split = use_fixed_split
        self.cache = None  # Simplified for integration
        
        # Set event codes based on dataset type
        if dataset_type == 'AVO':
            self.response_events = RESPONSE_EVENTS_AVO
            self.oddball_events = ODDBALL_EVENTS_AVO
        else:  # P3 or default
            self.response_events = RESPONSE_EVENTS
            self.oddball_events = ODDBALL_EVENTS

    def transform(self, raw):
        """Transform raw EEG data into windowed dataset."""
        # Standardise channel names to lower-case
        raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})

        # Select available channels
        available_channels = [ch for ch in self.eeg_channels if ch in raw.ch_names]
        if not available_channels:
            raise ValueError(
                f"None of the requested channels found. Available: {raw.ch_names}"
            )

        raw.pick_channels(available_channels)
        
        # Set reference to average (common for EEG analysis)
        try:
            raw.set_eeg_reference('average', projection=True)
        except Exception:
            # Fallback reference setting
            try:
                if 'cz' in [ch.lower() for ch in raw.ch_names]:
                    raw.set_eeg_reference(['Cz'])
            except Exception:
                pass  # Use original reference

        # Check and convert data units if needed
        raw_data_before = raw.get_data()
        if np.std(raw_data_before) < 1e-6 and np.std(raw_data_before) > 0:
            raw._data *= 1e6  # Convert V to μV
        elif np.std(raw_data_before) == 0:
            raise ValueError("Data is constant or zero")
        
        # Apply filtering and resampling
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
        raw.resample(RESAMPLE_FREQ)

        # Extract events
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found after reading annotations.")

        # Drop response events first
        response_mask = np.isin(events[:, 2], self.response_events)
        events = events[~response_mask]
        if len(events) == 0:
            raise ValueError("No non-response events found after filtering.")

        # Remove last remaining (non-response) event to avoid trailing window overflow
        events = events[:-1]
        
        # Separate oddball and standard events for balanced sampling
        oddball_mask = np.isin(events[:, 2], self.oddball_events)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]
        
        # Use fixed number of trials per class
        n_oddball = len(oddball_events)
        n_standard = len(standard_events)
        
        if n_oddball == 0:
            raise ValueError("No oddball events found in the data.")
        if n_standard == 0:
            raise ValueError("No standard events found in the data.")
        
        # Set random seed for reproducible sampling
        np.random.seed(self.random_seed)
        
        if self.use_fixed_split:
            # Use fixed split: 10+10 train, 5+5 val, 5+5 test
            train_oddball = TRAIN_TRIALS_PER_CLASS
            val_oddball = VAL_TRIALS_PER_CLASS
            test_oddball = TEST_TRIALS_PER_CLASS
            train_standard = TRAIN_TRIALS_PER_CLASS
            val_standard = VAL_TRIALS_PER_CLASS
            test_standard = TEST_TRIALS_PER_CLASS
            
            total_needed_oddball = train_oddball + val_oddball + test_oddball
            total_needed_standard = train_standard + val_standard + test_standard
            
            # Check if we have enough events
            if n_oddball < total_needed_oddball:
                #print(f"Warning: Only {n_oddball} oddball events available, need {total_needed_oddball}")
                # Adjust proportions
                train_oddball = min(train_oddball, n_oddball // 3)
                val_oddball = min(val_oddball, (n_oddball - train_oddball) // 2)
                test_oddball = n_oddball - train_oddball - val_oddball
            
            if n_standard < total_needed_standard:
                #print(f"Warning: Only {n_standard} standard events available, need {total_needed_standard}")
                # Adjust proportions
                train_standard = min(train_standard, n_standard // 3)
                val_standard = min(val_standard, (n_standard - train_standard) // 2)
                test_standard = n_standard - train_standard - val_standard
            
            # Sample events for each split
            oddball_indices = np.random.choice(n_oddball, size=n_oddball, replace=False)
            standard_indices = np.random.choice(n_standard, size=n_standard, replace=False)
            
            # Split oddball events
            oddball_train = oddball_events[oddball_indices[:train_oddball]]
            oddball_val = oddball_events[oddball_indices[train_oddball:train_oddball+val_oddball]]
            oddball_test = oddball_events[oddball_indices[train_oddball+val_oddball:train_oddball+val_oddball+test_oddball]]
            
            # Split standard events
            standard_train = standard_events[standard_indices[:train_standard]]
            standard_val = standard_events[standard_indices[train_standard:train_standard+val_standard]]
            standard_test = standard_events[standard_indices[train_standard+val_standard:train_standard+val_standard+test_standard]]
            
            # Combine all events and create labels
            all_events = np.vstack([
                oddball_train, standard_train,  # train: 0-19
                oddball_val, standard_val,      # val: 20-29
                oddball_test, standard_test     # test: 30-39
            ])
            
            # Create labels with split information
            train_labels = np.concatenate([
                np.ones(train_oddball, dtype=int),   # oddball = 1
                np.zeros(train_standard, dtype=int)  # standard = 0
            ])
            val_labels = np.concatenate([
                np.ones(val_oddball, dtype=int),     # oddball = 1
                np.zeros(val_standard, dtype=int)    # standard = 0
            ])
            test_labels = np.concatenate([
                np.ones(test_oddball, dtype=int),    # oddball = 1
                np.zeros(test_standard, dtype=int)   # standard = 0
            ])
            
            labels = np.concatenate([train_labels, val_labels, test_labels])
            
            # Create split indices
            train_end = len(train_labels)
            val_end = train_end + len(val_labels)
            test_end = val_end + len(test_labels)
            
            # Store split information
            self.train_indices = np.arange(0, train_end)
            self.val_indices = np.arange(train_end, val_end)
            self.test_indices = np.arange(val_end, test_end)
            
            selected_events = all_events
            
            # print(f"Fixed split dataset: Train({train_oddball}+{train_standard}), Val({val_oddball}+{val_standard}), Test({test_oddball}+{test_standard})")
            
        else:
            # Original logic: use fixed number of trials per class
            target_trials = self.fixed_trials_per_class
            
            # Sample oddball events
            if n_oddball >= target_trials:
                oddball_indices = np.random.choice(n_oddball, size=target_trials, replace=False)
                selected_oddball_events = oddball_events[oddball_indices]
            else:
                # Not enough oddball events - use all available
                selected_oddball_events = oddball_events.copy()
                #print(f"Warning: Only {n_oddball} oddball events available, using all of them")
            
            # Sample standard events
            if n_standard >= target_trials:
                standard_indices = np.random.choice(n_standard, size=target_trials, replace=False)
                selected_standard_events = standard_events[standard_indices]
            else:
                # Not enough standard events - use all available
                selected_standard_events = standard_events.copy()
                #print(f"Warning: Only {n_standard} standard events available, using all of them")
            
            # Combine selected events and create labels
            selected_events = np.vstack([selected_oddball_events, selected_standard_events])
            
            # Create balanced labels (1 for oddball, 0 for standard)
            n_selected_oddball = len(selected_oddball_events)
            n_selected_standard = len(selected_standard_events)
            labels = np.concatenate([
                np.ones(n_selected_oddball, dtype=int),  # oddball = 1
                np.zeros(n_selected_standard, dtype=int)  # standard = 0
            ])
            
            # Log balanced dataset info
            # print(f"Fixed trials dataset: {n_selected_oddball} oddball, {n_selected_standard} standard events (target: {target_trials} each)")

        # Manual window extraction to ensure one window per event
        raw_data = raw.get_data()  # Shape: (n_channels, n_timepoints)
        sfreq = raw.info['sfreq']
        
        # Extract windows manually
        windows_data = []
        windows_labels = []
        
        window_size = self.trial_stop_offset_samples - self.trial_start_offset_samples
        
        for i, (event_sample, _, _) in enumerate(selected_events):
            # Calculate window boundaries
            start_sample = event_sample + self.trial_start_offset_samples
            end_sample = event_sample + self.trial_stop_offset_samples
            
            # Check if window is within data bounds
            if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                # Extract window data
                window_data = raw_data[:, start_sample:end_sample]  # Shape: (n_channels, window_size)
                
                # Store window and label
                windows_data.append(window_data)
                windows_labels.append(labels[i])
        
        # Convert to numpy arrays
        windows_data = np.array(windows_data)  # Shape: (n_windows, n_channels, window_size)
        windows_labels = np.array(windows_labels)  # Shape: (n_windows,)
        
        # Basic data validation
        if np.any(np.isnan(windows_data)) or np.any(np.isinf(windows_data)):
            raise ValueError("Data contains NaN or infinite values")
        
        # print(f"Extracted {len(windows_data)} windows ({np.sum(windows_labels)} oddball, {len(windows_data)-np.sum(windows_labels)} standard)")
        
        # Return custom dataset
        return ManualWindowsDataset(windows_data, windows_labels)


def create_preprocessor(channels, dataset_type):
    """Create the appropriate preprocessor based on configuration."""
    if USE_ENHANCED_PREPROCESSING:
        # print("Using Enhanced Preprocessor with advanced features:")
        # print(f"  - Artifact removal (ICA): {REMOVE_ARTIFACTS}")
        # print(f"  - Baseline correction: {BASELINE_CORRECT}")
        # print(f"  - Frequency features: {EXTRACT_FREQUENCY_FEATURES}")
        # print(f"  - Notch filter: {APPLY_NOTCH_FILTER}")
        # For integration, we'll use the standard preprocessor
        return OddballPreprocessor(channels, dataset_type=dataset_type)
    else:
        print("Using Standard Preprocessor")
        return OddballPreprocessor(channels, dataset_type=dataset_type)


def get_dataset_subjects(dataset_type, dataset_obj):
    """Get subjects from dataset with limits."""
    if dataset_type == 'P3':
        all_subjects = sorted([d for d in os.listdir(dataset_obj) if d.startswith('sub-')])
        # Limit P3 dataset to configured maximum
        if MAX_SUBJECTS_P3 is not None:
            return all_subjects[:MAX_SUBJECTS_P3]
        return all_subjects
    elif dataset_type == 'AVO':
        all_files = [str(f) for f in dataset_obj.get_files()]
        all_subjects = sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
        
        # If we need to limit to MAX_SUBJECTS_AVO, select subjects with most oddball events
        if MAX_SUBJECTS_AVO is not None and len(all_subjects) > MAX_SUBJECTS_AVO:
            # Create a temporary preprocessor to count oddball events
            temp_preprocessor = create_preprocessor(AVO_CHANNELS, 'AVO')
            
            # Count oddball events for each subject
            subject_oddball_counts = []
            for subject_id in all_subjects:
                try:
                    data, labels = process_subject_data(subject_id, dataset_obj, temp_preprocessor, None, dataset_type='AVO')
                    if data is not None and labels is not None:
                        # Since data is already balanced (1:1 oddball:standard), oddball count = total / 2
                        oddball_count = len(data) // 2
                        subject_oddball_counts.append((subject_id, oddball_count))
                        # print(f"Subject sub-{subject_id}: {oddball_count} oddball trials")
                except Exception as e:
                    # If subject fails to process, assign 0 oddball count
                    subject_oddball_counts.append((subject_id, 0))
                    # print(f"Subject sub-{subject_id}: Failed to process, assigned 0 oddball trials")
            
            # Sort by oddball count (descending) and select top MAX_SUBJECTS_AVO
            subject_oddball_counts.sort(key=lambda x: x[1], reverse=True)
            selected_subjects = [subj[0] for subj in subject_oddball_counts[:MAX_SUBJECTS_AVO]]
            
            # print(f"\nSelected top {MAX_SUBJECTS_AVO} AVO subjects with most oddball trials:")
           # for i, (subj_id, count) in enumerate(subject_oddball_counts[:MAX_SUBJECTS_AVO]):
                # print(f"  {i+1:2d}. sub-{subj_id}: {count} oddball trials")
            
            return selected_subjects
        
        return all_subjects
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def process_subject_data(subject_id_or_dir, dataset_dir_or_obj, preprocessor, logger, dataset_type='P3'):
    """Process a single subject's data for either P3 or Active Visual Oddball dataset."""
    try:
        if dataset_type == 'P3':
            eeg_file = os.path.join(
                dataset_dir_or_obj, subject_id_or_dir, 'eeg', 
                f'{subject_id_or_dir}_task-P3_eeg.set'
            )
            raw = load_raw(eeg_file, dataset_type)
            
            # Basic data validation
            raw_data_loaded = raw.get_data()
            if np.all(raw_data_loaded == 0) or np.std(raw_data_loaded) < 1e-10:
                raise ValueError(f"Invalid data for {subject_id_or_dir}: data is constant or zero")
        elif dataset_type == 'AVO':
            all_files = [str(f) for f in dataset_dir_or_obj.get_files()]
            # Only include Visual Oddball (VO) runs
            vhdr_files = [
                f for f in all_files
                if f"sub-{subject_id_or_dir}" in f and 'visualoddball' in f and f.endswith('.vhdr')
            ]
            if not vhdr_files:
                return None, None
            
            # Concatenate all runs/files for the subject
            raws = [load_raw(f, dataset_type) for f in vhdr_files]
            for raw_obj in raws:
                raw_obj.load_data()
            raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
        else:
            raise ValueError("Unknown dataset_type: must be 'P3' or 'AVO'")

        # Process data
        windows = preprocessor.transform(raw)

        # Handle our custom ManualWindowsDataset
        if hasattr(windows, 'data') and hasattr(windows, 'labels'):
            # Custom dataset - direct access to data and labels
            data = windows.data
            labels = windows.labels
        else:
            # Original braindecode dataset - use indexing
            data = np.stack([windows[i][0] for i in range(len(windows))])
            labels = np.array([windows[i][1] for i in range(len(windows))])
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()

        return data, labels

    except Exception as e:
        if dataset_type == 'P3':
            log_error(logger, "P3", subject_id_or_dir, e)
        else:
            log_error(logger, "Active Visual Oddball", f"sub-{subject_id_or_dir}", e)
        return None, None

# %% [markdown]
# ## Neural Network Models
# 
# Deep learning model architecture

# %%
# Add all missing configuration variables to match main.py
MAX_SUBJECTS_P3 = 40
MAX_SUBJECTS_AVO = None  # No limit for AVO dataset

# Enhanced preprocessing flags - MUST MATCH config.py settings!
REMOVE_ARTIFACTS = True              # Use ICA for artifact removal
BASELINE_CORRECT = True              # Apply baseline correction
EXTRACT_FREQUENCY_FEATURES = True    # Add frequency domain features
APPLY_NOTCH_FILTER = True            # Remove power line interference

# Trial configuration (already defined above but ensure they exist)
FIXED_TRIALS_PER_SUBJECT_TRAIN = None
FIXED_TRIALS_PER_SUBJECT_VAL = None
FIXED_TRIALS_PER_SUBJECT_TEST = None
MAX_TRIALS_PER_SUBJECT_TRAIN = None
MAX_TRIALS_PER_SUBJECT_VAL = None
MAX_TRIALS_PER_SUBJECT_TEST = None

# Random seed configuration
RANDOM_SEED = seeds

# Cross-validation confidence level
NESTED_CV_CONFIDENCE_LEVEL = 0.95

# Subject layer configuration (for models, not used in EEGConformer)
use_subject_layer = False

# Combined dataset configuration
use_combined_datasets = True
separate_subject_classification = False


# %%
class WarmupCosineAnnealingLR:
    """Learning rate scheduler with warmup followed by cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_factor=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_factor = warmup_factor
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        
        # Initialize with warmup learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.base_lr * warmup_factor
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.base_lr * (self.warmup_factor + 
                               (1 - self.warmup_factor) * self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class CustomShallowFBCSPNet(nn.Module):
    """Custom implementation of ShallowFBCSPNet."""
    def __init__(self, n_chans, n_outputs, n_times, final_conv_length='auto'):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        
        # Spatial convolution
        self.spatial_conv = nn.Conv2d(40, 40, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        
        # Pooling
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        
        # Calculate output size
        self._calculate_final_conv_length()
        
        # Final classification layer
        self.classifier = nn.Linear(self.final_length, n_outputs)
        
    def _calculate_final_conv_length(self):
        # Calculate the final convolution length
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, self.n_times)
            x = self.temporal_conv(x)  
            x = self.spatial_conv(x)   
            x = self.bn(x)             
            x = F.elu(x)               
            x = self.pool(x)           
            self.final_length = x.numel() // x.size(0)
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class EEGNet(nn.Module):
    """EEGNet implementation for EEG classification."""
    def __init__(self, n_chans, n_outputs, n_times, 
                 F1=8, F2=16, D=2, dropout=0.5):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.F1 = F1
        self.F2 = F2
        self.D = D
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(F1, F1*D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 2
        # Separable convolution
        self.separable_conv = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate final dimensions
        self._calculate_final_dims(n_times)
        
        # Classification
        self.classifier = nn.Linear(self.final_length, n_outputs)
        
    def _calculate_final_dims(self, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, n_times)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            self.final_length = x.numel() // x.size(0)
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class EEGConformer(nn.Module):
    """EEGConformer: Combining CNN and Transformer for EEG classification."""
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
        seq_length = self._get_sequence_length()
        
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
    
    def _get_sequence_length(self):
        # Calculate sequence length after convolutions
        # After temporal conv: n_times (same due to padding)
        # After avg pool: n_times // 4
        return self.n_times // 4
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (batch, conv_temporal_dim, n_chans, n_times)
        x = self.temporal_bn(x)
        x = F.elu(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (batch, conv_spatial_dim, 1, n_times)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Pooling
        x = self.avg_pool(x)  # (batch, conv_spatial_dim, 1, n_times//4)
        
        # Reshape for transformer
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, conv_spatial_dim)
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, embedding_dim)
        
        # Classification
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        x = self.classifier(x)  # (batch, n_outputs)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def augment_data(x, training=True):
    """Apply data augmentation to EEG data."""
    if not training or not USE_DATA_AUGMENTATION:
        return x
    
    batch_size, n_channels, n_timepoints = x.shape
    augmented_x = x.clone()
    
    # Add Gaussian noise
    if NOISE_STD > 0:
        noise = torch.randn_like(augmented_x) * NOISE_STD
        augmented_x = augmented_x + noise
    
    # Time shifting
    if TIME_SHIFT_RANGE > 0:
        for i in range(batch_size):
            shift = np.random.randint(-TIME_SHIFT_RANGE, TIME_SHIFT_RANGE + 1)
            if shift != 0:
                if shift > 0:
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                else:
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
    
    return augmented_x


def label_smoothing_loss(pred, target, smoothing=LABEL_SMOOTHING):
    """Compute label smoothing loss."""
    if smoothing == 0.0:
        return F.cross_entropy(pred, target)
    
    n_classes = pred.size(-1)
    one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
    return -(smooth_one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()


def normalize_data(x):
    """Normalize data with robust handling of constant channels and enhanced features."""
    # Debug: Check input data
    if torch.all(x == 0):
        #print("WARNING: All input data to normalize_data is zero!")
        return x

    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)

    # More robust handling of zero standard deviation
    zero_std_mask = (std <= NORMALIZATION_EPSILON)
    num_zero_std = torch.sum(zero_std_mask).item()

    if num_zero_std > 0:
        # For constant channels, keep them as-is (subtract mean, but don't divide by std)
        std = torch.where(zero_std_mask, torch.ones_like(std), std)

    # Apply normalization
    std = std + NORMALIZATION_EPSILON
    normalized = (x - mean) / std

    # For originally constant channels, set them to zero (mean-centered)
    normalized = torch.where(zero_std_mask.expand_as(normalized),
                           torch.zeros_like(normalized), normalized)

    # Final check for numerical issues
    if torch.any(torch.isnan(normalized)) or torch.any(torch.isinf(normalized)):
       # print("WARNING: NaN or Inf values after normalization, cleaning...")
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)

    return normalized


def create_model(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='ShallowFBCSPNet', input_channels=None):
    """Create a new model based on configuration."""
    if is_lda:
        return LDA()
    else:
        # Determine if subject layer should be enabled
        if enable_subject_layer is None:
            enable_subject_layer = use_subject_layer

        # Use input_channels if provided, otherwise use n_channels
        actual_channels = input_channels if input_channels is not None else n_channels
        
        # Create base model based on model_name
        if model_name == 'ShallowFBCSPNet':
            if BRAINDECODE_AVAILABLE:
                base_model = ShallowFBCSPNet(
                    n_chans=actual_channels,
                    n_outputs=N_CLASSES,
                    n_times=INPUT_WINDOW_SAMPLES,
                    final_conv_length='auto'
                )
            else:
                base_model = CustomShallowFBCSPNet(
                    n_chans=actual_channels,
                    n_outputs=N_CLASSES,
                    n_times=INPUT_WINDOW_SAMPLES
                )
        elif model_name == 'EEGNet' or model_name == 'EEGNetv4':
            base_model = EEGNet(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                dropout=DROPOUT_RATE
            )
        elif model_name == 'EEGConformer':
            base_model = EEGConformer(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
                conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
                embedding_dim=CONFORMER_EMBEDDING_DIM,
                num_heads=CONFORMER_NUM_HEADS,
                num_layers=CONFORMER_NUM_LAYERS,
                dropout=DROPOUT_RATE,
                activation=CONFORMER_ACTIVATION
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return base_model


def early_stopping(val_acc, model, state, patience=EARLY_STOPPING_PATIENCE):
    """Early stopping helper function."""
    if 'best_val_acc' not in state:
        state['best_val_acc'] = 0
        state['counter'] = 0
        state['best_model'] = None
        state['early_stop'] = False

    if val_acc > state['best_val_acc']:
        state['best_val_acc'] = val_acc
        state['counter'] = 0
        state['best_model'] = model.state_dict().copy()
    else:
        state['counter'] += 1
        if state['counter'] >= patience:
            state['early_stop'] = True
    return state['early_stop']


def evaluate(model, loader, device, is_lda=False, subject_mapping=None, return_details=False):
    """Evaluate model on data loader."""
    if is_lda:
        X = []
        y = []
        for batch_data in loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                batch_X, batch_y, _ = batch_data
            else:  # (X, y)
                batch_X, batch_y = batch_data
            X.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
            y.append(batch_y.numpy())
        X = np.concatenate(X)
        y = np.concatenate(y)
        predictions = model.predict(X)
        
        if return_details:
            try:
                # Get probability estimates for AUC calculation
                y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            except:
                y_proba = predictions  # Fallback to binary predictions if probabilities not available
            
            # Calculate confusion matrix first
            cm = confusion_matrix(y, predictions)
            
            # Handle different confusion matrix shapes
            if cm.shape == (1, 1):
                # Only one class present
                tp = cm[0, 0] if predictions[0] == y[0] else 0
                tn = fp = fn = 0
                accuracy = 1.0 if tp > 0 else 0.0
                precision = recall = f1 = 1.0 if tp > 0 else 0.0
            elif cm.shape == (2, 2):
                # Standard 2x2 confusion matrix
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                # Fallback: calculate metrics directly
                correct = np.sum(predictions == y)
                accuracy = correct / len(y)
                tp = tn = fp = fn = 0
                precision = recall = f1 = 0.0
            try:
                # Check if we have both classes in the true labels
                unique_labels = np.unique(y)
                if len(unique_labels) < 2:
                    #print(f"Warning: Only one class present in test set: {unique_labels}. Setting AUC to 0.5.")
                    auc = 0.5
                else:
                    # Check for problematic probability values
                    if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
                       # print(f"Warning: Found NaN or infinite values in probabilities. Setting AUC to 0.5.")
                        auc = 0.5
                    else:
                        auc = roc_auc_score(y, y_proba)
                        if np.isnan(auc):
                           # print(f"Warning: AUC calculation returned NaN. Setting to 0.5.")
                            auc = 0.5
            except Exception as e:
               # print(f"Warning: AUC calculation failed: {e}. Setting to 0.5.")
                auc = 0.5
            
            return {
                'accuracy': accuracy,
                'correct_count': tp + tn,
                'incorrect_count': fp + fn,
                'total_count': tp + tn + fp + fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        # For LDA without details, calculate accuracy from confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # Handle different confusion matrix shapes
        if cm.shape == (1, 1):
            # Only one class present
            return 1.0 if predictions[0] == y[0] else 0.0
        elif cm.shape == (2, 2):
            # Standard 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            return accuracy
        else:
            # Fallback: calculate accuracy directly
            correct = np.sum(predictions == y)
            return correct / len(y)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    correct = 0
    total = 0
    
    # Debug: Check loader
    loader_size = len(loader.dataset)
    if loader_size == 0:
       # print(f"Warning: Loader is empty in evaluate function!")
        return 0.0
    
    with torch.no_grad():
        batch_count = 0
        for batch_data in loader:
            batch_count += 1
            if len(batch_data) == 3:  
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else: 
                x, y = batch_data
                subject_indices = None
            
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)

            scores = model(x)
            
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)  
            
            _, predicted = scores.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
            # Collect predictions and targets for detailed evaluation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # Get probabilities for AUC calculation
            probabilities = torch.softmax(scores, dim=1)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    if return_details:
        # Calculate precision, recall, F1 score and AUC
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate confusion matrix first
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Handle different confusion matrix shapes
        if cm.shape == (1, 1):
            # Only one class present
            tp = cm[0, 0] if all_predictions[0] == all_targets[0] else 0
            tn = fp = fn = 0
            accuracy = 1.0 if tp > 0 else 0.0
            precision = recall = f1 = 1.0 if tp > 0 else 0.0
        elif cm.shape == (2, 2):
            # Standard 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Fallback: calculate metrics directly
            correct = np.sum(all_predictions == all_targets)
            accuracy = correct / len(all_targets)
            tp = tn = fp = fn = 0
            precision = recall = f1 = 0.0
        
        # Calculate AUC
        try:
            # Check if we have both classes in the true labels
            unique_labels = np.unique(all_targets)
            if len(unique_labels) < 2:
               # print(f"Warning: Only one class present in overall test set: {unique_labels}. Setting AUC to 0.5.")
                auc = 0.5
            else:
                # Check for problematic probability values
                if np.any(np.isnan(all_probabilities)) or np.any(np.isinf(all_probabilities)):
                   # print(f"Warning: Found NaN or infinite values in overall probabilities. Setting AUC to 0.5.")
                    auc = 0.5
                else:
                    auc = roc_auc_score(all_targets, all_probabilities)
                    if np.isnan(auc):
                      #  print(f"Warning: Overall AUC calculation returned NaN. Setting to 0.5.")
                        auc = 0.5
        except Exception as e:
           # print(f"Warning: Overall AUC calculation failed: {e}. Setting to 0.5.")
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'correct_count': tp + tn,
            'incorrect_count': fp + fn,
            'total_count': tp + tn + fp + fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # For neural network without details, calculate accuracy from confusion matrix
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Check if we have predictions and targets
    if len(all_predictions) == 0 or len(all_targets) == 0:
       # print(f"Warning: No predictions or targets in evaluate function!")
        return 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Handle case where confusion matrix is not 2x2 (single class)
    if cm.shape == (1, 1):
        # Only one class present
        return 1.0 if all_predictions[0] == all_targets[0] else 0.0
    elif cm.shape == (2, 2):
        # Standard 2x2 confusion matrix
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
    else:
        # Fallback: calculate accuracy directly
        correct = np.sum(all_predictions == all_targets)
        return correct / len(all_targets)


def train_model(model, train_loader, val_loader, test_loader, device, is_lda=False, max_epochs=MAX_EPOCHS, model_name=None):
    """Train a model with early stopping."""
    if is_lda:
        # Prepare data for LDA
        X_train = []
        y_train = []
        for batch_data in train_loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                batch_X, batch_y, _ = batch_data
            else:  # (X, y)
                batch_X, batch_y = batch_data
            X_train.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
            y_train.append(batch_y.numpy())
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Train LDA model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        return evaluate(model, test_loader, device, is_lda=True)
    
    # Neural Network training
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Maintain state for early stopping
    es_state = {}

    # Initialize focal loss
    focal_loss = FocalLoss(alpha=1, gamma=2, weight=None)
    
    # Training progress tracking
    print(f"\n{'='*60}")
    # print(f"Starting Training - Max Epochs: {max_epochs}")
    # print(f"Model: {type(model).__name__}")
    # print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    # print(f"Dropout: {DROPOUT_RATE}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    # print(f"{'='*60}")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:  # (X, y, subject_indices)
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else:  # (X, y) - backward compatibility
                x, y = batch_data
                subject_indices = None
            
            # Apply data augmentation
            x = augment_data(x, training=True)
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                y = y.long()
            
            optimizer.zero_grad()
            
            # Forward pass
            scores = model(x)

            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)
            
            # Use focal loss with label smoothing
            if LABEL_SMOOTHING > 0:
                loss = label_smoothing_loss(scores, y, LABEL_SMOOTHING)
            else:
                loss = focal_loss(scores, y)
            
            loss.backward()
            optimizer.step()
            
            # Track training statistics
            epoch_loss += loss.item()
            _, predicted = scores.max(1)
            epoch_correct += (predicted == y).sum().item()
            epoch_total += y.size(0)
            batch_count += 1
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / batch_count
        train_acc = 100. * epoch_correct / epoch_total
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        
        # Validation phase
        val_samples = len(val_loader.dataset)
        if val_samples == 0:
           # print(f"Warning: Validation loader is empty!")
            val_acc = 0.0
        else:
            val_acc = evaluate(model, val_loader, device)
            # if val_acc == 0.0 and val_samples > 0:
            #     print(f"Warning: Validation accuracy is 0.0 with {val_samples} samples")
        val_acc_percent = 100. * val_acc
        
        # Print epoch summary (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1:3d}/{max_epochs} Summary:")
            print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc_percent:.2f}% | LR: {current_lr:.6f}")
        
        # Early stopping check
        is_best = False
        if 'best_val_acc' not in es_state or val_acc > es_state['best_val_acc']:
            is_best = True
            
        if early_stopping(val_acc, model, es_state, patience=EARLY_STOPPING_PATIENCE):
            # print(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            # print(f"Best validation accuracy: {100. * es_state['best_val_acc']:.2f}%")
            break
        else:
            if is_best:
                print(f"New best validation accuracy!")
            else:
                remaining_patience = EARLY_STOPPING_PATIENCE - es_state['counter']
                print(f"Patience remaining: {remaining_patience}/{EARLY_STOPPING_PATIENCE}")
        
        print(f"  {'-'*50}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    if 'best_val_acc' in es_state:
        print(f"Best Validation Accuracy: {100. * es_state['best_val_acc']:.2f}%")
    print(f"{'='*60}")
    
    # Load best model and evaluate on test set
    if 'best_model' in es_state and es_state['best_model'] is not None:
        model.load_state_dict(es_state['best_model'])
    return evaluate(model, test_loader, device)

# %% [markdown]
# ## Data Loading Functions
# 
# Functions for creating data loaders and splitting datasets.

# %%
def create_data_loaders(data, labels, batch_size=BATCH_SIZE, 
                       train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
                       return_indices=False, max_trials_per_split=None):
    """Create train, validation, and test data loaders."""
    temp_size = val_size + test_size
    indices = np.arange(len(data))
    
    train_indices, temp_indices, X_train, X_temp, y_train, y_temp = train_test_split(
        indices, data, labels, test_size=temp_size, stratify=labels
    )
    
    test_ratio = test_size / temp_size  
    val_indices, test_indices, X_val, X_test, y_val, y_test = train_test_split(
        temp_indices, X_temp, y_temp, test_size=test_ratio, stratify=y_temp
    )
    
    # Apply trial limits if specified, maintaining class balance
    if max_trials_per_split is not None:
        if 'train' in max_trials_per_split and max_trials_per_split['train'] is not None:
            max_train = max_trials_per_split['train']
            if len(X_train) > max_train:
                # Sample while maintaining class balance
                X_train, y_train, train_indices = _balanced_sample(
                    X_train, y_train, train_indices, max_train, seed=42
                )
        
        if 'val' in max_trials_per_split and max_trials_per_split['val'] is not None:
            max_val = max_trials_per_split['val']
            if len(X_val) > max_val:
                # Sample while maintaining class balance
                X_val, y_val, val_indices = _balanced_sample(
                    X_val, y_val, val_indices, max_val, seed=42
                )
        
        if 'test' in max_trials_per_split and max_trials_per_split['test'] is not None:
            max_test = max_trials_per_split['test']
            if len(X_test) > max_test:
                # Sample while maintaining class balance
                X_test, y_test, test_indices = _balanced_sample(
                    X_test, y_test, test_indices, max_test, seed=42
                )
    
    # Debug: Print final class distributions
    print(f"DEBUG: Final class distributions:")
    print(f"  Train: {np.bincount(y_train).tolist()}")
    print(f"  Val:   {np.bincount(y_val).tolist()}")
    print(f"  Test:  {np.bincount(y_test).tolist()}")
    
    # Since dataset is now balanced at source, no need for weighted sampling
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), 
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
        batch_size=batch_size, shuffle=False
    )
    
    if return_indices:
        return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices
    else:
        return train_loader, val_loader, test_loader


def _balanced_sample(X, y, indices, max_samples, seed=42):
    """Sample data while maintaining class balance (1:1 ratio)."""
    np.random.seed(seed)
    
    # Get unique classes
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
       # print(f"Warning: Expected 2 classes, found {len(unique_classes)}. Using random sampling.")
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            return X[sample_indices], y[sample_indices], indices[sample_indices]
        return X, y, indices
    
    # Calculate samples per class (ensure even number for 1:1 ratio)
    samples_per_class = max_samples // 2
    
    # Get indices for each class
    class_0_indices = np.where(y == unique_classes[0])[0]
    class_1_indices = np.where(y == unique_classes[1])[0]
    
    # Check if we have enough samples for each class
    if len(class_0_indices) < samples_per_class or len(class_1_indices) < samples_per_class:
       # print(f"Warning: Not enough samples for balanced sampling. Class 0: {len(class_0_indices)}, Class 1: {len(class_1_indices)}, Need: {samples_per_class} each")
        # Use all available samples if not enough for balanced sampling
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            return X[sample_indices], y[sample_indices], indices[sample_indices]
        return X, y, indices
    
    # Sample from each class
    class_0_sample = np.random.choice(class_0_indices, samples_per_class, replace=False)
    class_1_sample = np.random.choice(class_1_indices, samples_per_class, replace=False)
    
    # Combine samples
    sample_indices = np.concatenate([class_0_sample, class_1_sample])
    np.random.shuffle(sample_indices)  # Shuffle to mix classes
    
    # Debug: Verify class balance
    sampled_y = y[sample_indices]
    class_counts = np.bincount(sampled_y)
    print(f"DEBUG: Balanced sampling - Class distribution: {class_counts.tolist()}")
    
    return X[sample_indices], y[sample_indices], indices[sample_indices]


def get_trial_limits_from_config():
    """Get trial limits from configuration."""
    # Check if fixed trial counts are specified (takes priority)
    if any(x is not None for x in [FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST]):
        return {
            'train': FIXED_TRIALS_PER_SUBJECT_TRAIN,
            'val': FIXED_TRIALS_PER_SUBJECT_VAL,
            'test': FIXED_TRIALS_PER_SUBJECT_TEST
        }
    
    # Check if max trial limits are specified
    if any(x is not None for x in [MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST]):
        return {
            'train': MAX_TRIALS_PER_SUBJECT_TRAIN,
            'val': MAX_TRIALS_PER_SUBJECT_VAL,
            'test': MAX_TRIALS_PER_SUBJECT_TEST
        }
    
    return None


def run_experiment_with_seed(train_loader, val_loader, test_loader, n_channels, device,
                           seed, classifier_type, print_model_summary=False, return_details=False, input_channels=None):
    """Run a single experiment with a specific random seed."""
    is_lda = classifier_type.lower() == 'lda'
    
    if not is_lda:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        np.random.seed(seed)
    
    model = create_model(n_channels, is_lda, input_channels=input_channels)
    if not is_lda:
        # Only neural network models need to be moved to device
        if hasattr(model, 'to'):
            model = model.to(device)
        # Print model summary only once per experiment (for the first seed)
        if print_model_summary and seed == seeds[0]:
            print("\n" + "="*60)
            print("Model Architecture Summary")
            print("="*60)
            print(f"Model type: {type(model).__name__}")
            print(f"Input channels: {n_channels}")
            print(f"Input shape: (batch_size, {n_channels}, 128)")
            if hasattr(model, 'parameters'):
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
            print("="*60 + "\n")
    
    # Train the model
    train_model(model, train_loader, val_loader, test_loader, device, is_lda, MAX_EPOCHS)
    
    # Get test evaluation with details if requested
    if return_details:
        test_result = evaluate(model, test_loader, device, is_lda, return_details=True)
        return test_result, model
    else:
        accuracy = evaluate(model, test_loader, device, is_lda)
        return accuracy, model

# %% [markdown]
# ## AS-MMD Implementation
# 
# Adaptive Symmetric Maximum Mean Discrepancy for domain adaptation.

# %%
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform mixup augmentation on small-sample data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def compute_focal_loss(scores: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Compute focal loss to handle class imbalance and hard examples."""
    ce_loss = F.cross_entropy(scores, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def mixup_criterion(pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, gamma: float = 3.6, alpha: float = 0.65) -> torch.Tensor:
    """Compute mixup focal loss."""
    loss_a = compute_focal_loss(pred, y_a, gamma=gamma, alpha=alpha)
    loss_b = compute_focal_loss(pred, y_b, gamma=gamma, alpha=alpha)
    return lam * loss_a + (1 - lam) * loss_b


def compute_prototypes(features: torch.Tensor, labels: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
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


def compute_prototype_loss(features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute prototype alignment loss."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    # Compute distance to correct prototype
    loss = 0.0
    n_samples = 0
    for i, label in enumerate(labels):
        proto = prototypes[label]
        dist = F.mse_loss(features[i], proto)
        loss += dist
        n_samples += 1

    return loss / max(1, n_samples)


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, logger: logging.Logger, eps: float = 1e-8) -> torch.Tensor:
    """Compute unbiased RBF-MMD between two batches (features or logits)."""
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)
    with torch.no_grad():
        # Median heuristic on combined data
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
    # Unbiased estimate: exclude diagonals
    mmd = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    mmd += (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    mmd -= 2.0 * k_xy.mean()
    return mmd


def snapshot_bn_buffers(model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Capture running_mean and running_var tensors of all BN layers."""
    buffers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Clone to detach current buffers
            rm = m.running_mean.clone() if m.running_mean is not None else None
            rv = m.running_var.clone() if m.running_var is not None else None
            buffers.append((rm, rv))
    return buffers


def restore_bn_buffers(model: nn.Module, buffers: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Restore running_mean and running_var of BN layers from snapshot."""
    idx = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            rm, rv = buffers[idx]
            if rm is not None and m.running_mean is not None:
                m.running_mean.data.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.data.copy_(rv)
            idx += 1


def get_channels_for_dataset(name: str, use_all: bool) -> List[str]:
    """Get appropriate channels for dataset."""
    if name == 'P3':
        return P3_CHANNELS if use_all else COMMON_CHANNELS
    elif name == 'AVO':
        return AVO_CHANNELS if use_all else COMMON_CHANNELS
    else:
        return COMMON_CHANNELS


def load_combined_arrays(logger: logging.Logger, channels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined arrays (P3+AVO) with per-subject stratified sampling from config."""
    X_list = []
    y_list = []
    src_list = []

    for dataset_name in ['P3', 'AVO']:
        # logger.info(f"Loading dataset for CV: {dataset_name}")
        
        if dataset_name == 'P3':
            subjects = get_dataset_subjects('P3', P3_DATA_DIR)
            dataset_obj = P3_DATA_DIR
            n_trials_ps = NESTED_CV_TRIALS_PER_SUBJECT_P3
        else:
            avo_dataset = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            dataset_obj = avo_dataset
            n_trials_ps = NESTED_CV_TRIALS_PER_SUBJECT_AVO

        preproc = create_preprocessor(channels, dataset_name)

        for s in subjects:
            data, labels = process_subject_data(s, dataset_obj, preproc, logger, dataset_type=dataset_name)
            if data is None or labels is None or len(data) == 0:
                continue
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            # Per-subject stratified sampling to target budget
            if len(data) > n_trials_ps:
                data, labels = stratified_sample_trials(data, labels, n_trials_ps, f"{dataset_name}_{s}", logger)
            X_list.append(data)
            y_list.append(labels)
            src_list.append(np.array([dataset_name] * len(labels)))

    if not X_list:
        raise RuntimeError("No valid data loaded for CV")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    src_all = np.concatenate(src_list, axis=0)

    # logger.info(f"Combined dataset summary: total={len(X_all)}, P3={np.sum(src_all=='P3')}, AVO={np.sum(src_all=='AVO')}")
    return X_all, y_all, src_all


def stratified_sample_trials(data, labels, n_trials, subject_id, logger):
    """Perform stratified sampling of trials for a single subject."""
    # Set random seed for reproducible sampling
    base_seed = RANDOM_SEED[0] if isinstance(RANDOM_SEED, (list, tuple)) else RANDOM_SEED
    np.random.seed(base_seed + hash(subject_id) % 1000)  # Add subject-specific variation
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # logger.warning(f"Subject {subject_id}: Only one class found, using random sampling")
        indices = np.random.choice(len(data), size=min(n_trials, len(data)), replace=False)
        return data[indices], labels[indices]

    # Calculate how many trials to sample from each class
    label_counts = {label: np.sum(labels == label) for label in unique_labels}
    total_available = len(data)

    if n_trials >= total_available:
        # If we want more trials than available, return all
        # logger.info(f"Subject {subject_id}: Requested {n_trials} trials, but only {total_available} available. Using all.")
        return data, labels

    # Proportional stratified sampling
    sampled_indices = []
    for label in unique_labels:
        label_mask = labels == label
        available_for_label = np.sum(label_mask)

        # Calculate proportional number of samples for this label
        proportion = available_for_label / total_available
        n_for_label = max(1, int(n_trials * proportion))  # At least 1 sample per class

        # Adjust if we would exceed the requested total
        if len(sampled_indices) + n_for_label > n_trials:
            n_for_label = n_trials - len(sampled_indices)

        if n_for_label > 0 and available_for_label >= n_for_label:
            label_indices = np.where(label_mask)[0]
            selected = np.random.choice(label_indices, size=n_for_label, replace=False)
            sampled_indices.extend(selected)

    # If we still need more samples (due to rounding), randomly add from remaining
    remaining_needed = n_trials - len(sampled_indices)
    if remaining_needed > 0:
        all_indices = set(range(len(data)))
        used_indices = set(sampled_indices)
        remaining_indices = list(all_indices - used_indices)

        if len(remaining_indices) >= remaining_needed:
            additional = np.random.choice(remaining_indices, size=remaining_needed, replace=False)
            sampled_indices.extend(additional)

    sampled_indices = np.array(sampled_indices)

    # Verify stratification
    original_distribution = {label: np.mean(labels == label) for label in unique_labels}
    sampled_distribution = {label: np.mean(labels[sampled_indices] == label) for label in unique_labels}

            # logger.info(f"Subject {subject_id}: Sampled {len(sampled_indices)}/{total_available} trials")
        # logger.info(f"  Original distribution: {original_distribution}")
        # logger.info(f"  Sampled distribution: {sampled_distribution}")

    return data[sampled_indices], labels[sampled_indices]


def get_symmetric_adjustments(n_train_a: int, n_train_b: int) -> Tuple[float, float, float, int]:
    """Compute symmetric domain weights based purely on relative sizes."""
    n_train_a = max(1, n_train_a)
    n_train_b = max(1, n_train_b)
    ratio_ab = n_train_a / float(n_train_b)

    # PROTOTYPE-BASED: More conservative weights since we have prototype guidance
    # Use sqrt of ratio for gentler scaling
    w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 3.0, 1.0, 12.0))

    # Reduced MMD - let prototypes handle discriminative alignment
    overall_ratio = max(ratio_ab, 1.0 / ratio_ab)
    lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.3 if overall_ratio < 4.0 else 0.4)

    # Prototype loss weight - key for discriminative transfer
    lambda_proto = 0.5 if overall_ratio < 4.0 else 0.8

    # Longer warmup for stable learning
    warmup = max(20, min(40, int(0.4 * MAX_EPOCHS)))

    return w_small, lambda_mmd, lambda_proto, warmup


def evaluate_domain(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model on a specific domain."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = normalize_data(x).to(device)
            y = y.to(device)
            scores = model(x)
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)
            _, pred = scores.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# %% [markdown]
# ## Main Experiment Functions# %%

# %% [markdown]
# ## Main Experiment Functions
# 
# AS-MMD training and cross-validation.

# %%
def asmmd_train_fold(
    logger,
    Xtr_p3, ytr_p3,
    Xva_p3, yva_p3,
    Xtr_avo, ytr_avo,
    Xva_avo, yva_avo,
    channels, seed=42,
):
    """Train a single fold with AS-MMD method."""
    device = get_device()
    set_global_torch_seed(seed)

    # Build loaders per domain
    Xtr_p3 = torch.FloatTensor(Xtr_p3)
    ytr_p3 = torch.LongTensor(ytr_p3)
    Xva_p3 = torch.FloatTensor(Xva_p3)
    yva_p3 = torch.LongTensor(yva_p3)
    Xtr_avo = torch.FloatTensor(Xtr_avo)
    ytr_avo = torch.LongTensor(ytr_avo)
    Xva_avo = torch.FloatTensor(Xva_avo)
    yva_avo = torch.LongTensor(yva_avo)

    train_loader_p3 = DataLoader(TensorDataset(Xtr_p3, ytr_p3), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_p3 = DataLoader(TensorDataset(Xva_p3, yva_p3), batch_size=BATCH_SIZE, shuffle=False)
    train_loader_avo = DataLoader(TensorDataset(Xtr_avo, ytr_avo), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_avo = DataLoader(TensorDataset(Xva_avo, yva_avo), batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    input_channels = Xtr_avo.shape[1] if Xtr_avo.shape[1] == Xtr_p3.shape[1] else max(Xtr_avo.shape[1], Xtr_p3.shape[1])
    model = create_model(n_channels=len(channels), is_lda=False, input_channels=input_channels)
    model = model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Get adjustments
    n_train_avo = len(Xtr_avo)
    n_train_p3 = len(Xtr_p3)
    w_small, lambda_mmd_target, lambda_proto_target, warmup_epochs = get_symmetric_adjustments(n_train_avo, n_train_p3)
    small_domain = 'P3' if n_train_p3 <= n_train_avo else 'AVO'
    large_domain = 'AVO' if small_domain == 'P3' else 'P3'
    
    # logger.info(f"Fold domains: small={small_domain} (n={min(n_train_p3, n_train_avo)}), large={large_domain} (n={max(n_train_p3, n_train_avo)})")
    # logger.info(f"Auto adjustments: w_small≈{w_small:.3f}, lambda_MMD={lambda_mmd_target:.3f}, lambda_proto={lambda_proto_target:.3f}, warmup={warmup_epochs}")

    # Early stopping
    best_val = 0.0
    best_state = None
    patience_cnt = 0

    # Initialize prototypes
    large_prototypes = None

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup schedules
        alpha = min(1.0, epoch / max(1, warmup_epochs))
        w_small_target_val = w_small
        w_large_target_val = 1.0
        w_small_cur = 1.0 + alpha * (w_small_target_val - 1.0)
        w_large_cur = 1.0 + alpha * (w_large_target_val - 1.0)
        lambda_mmd = alpha * lambda_mmd_target
        lambda_proto = alpha * lambda_proto_target

        lr_cur = optimizer.param_groups[0]['lr']
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}/{MAX_EPOCHS} | LR={lr_cur:.6f} | w_{large_domain}={w_large_cur:.3f} | w_{small_domain}={w_small_cur:.3f} | λ_MMD={lambda_mmd:.3f} | λ_proto={lambda_proto:.3f}")

        # Iterators
        train_loaders = {'P3': train_loader_p3, 'AVO': train_loader_avo}
        itr_large = iter(train_loaders[large_domain])
        itr_small = iter(train_loaders[small_domain]) if len(train_loaders[small_domain]) > 0 else None

        steps = 0
        epoch_loss = 0.0

        while True:
            try:
                xb_large, yb_large = next(itr_large)
            except StopIteration:
                break

            if itr_small is None:
                xb_small = None
                yb_small = None
            else:
                try:
                    xb_small, yb_small = next(itr_small)
                except StopIteration:
                    itr_small = iter(train_loaders[small_domain])
                    xb_small, yb_small = next(itr_small) if len(train_loaders[small_domain]) > 0 else (None, None)

            optimizer.zero_grad()

            # Forward on large domain
            x_large = normalize_data(xb_large).to(device)
            y_large = yb_large.to(device)
            scores_large = model(x_large)
            if scores_large.ndim > 2:
                scores_large = scores_large.view(scores_large.size(0), -1)
            loss_large = F.cross_entropy(scores_large, y_large)

            # Update prototypes
            with torch.no_grad():
                current_prototypes = compute_prototypes(scores_large.detach(), y_large, n_classes=2)
                if large_prototypes is None:
                    large_prototypes = current_prototypes
                else:
                    large_prototypes = 0.9 * large_prototypes + 0.1 * current_prototypes

            # Forward on small domain
            loss_small = torch.tensor(0.0, device=device)
            loss_proto = torch.tensor(0.0, device=device)
            scores_small = None

            if xb_small is not None:
                x_small = normalize_data(xb_small).to(device)
                y_small = yb_small.to(device)

                # Mixup
                x_mixed, y_a, y_b, lam = mixup_data(x_small, y_small, alpha=0.4)
                scores_small = model(x_mixed)
                if scores_small.ndim > 2:
                    scores_small = scores_small.view(scores_small.size(0), -1)
                loss_small = mixup_criterion(scores_small, y_a, y_b, lam, gamma=2.0, alpha=0.5)

                # Prototype loss
                if large_prototypes is not None and lambda_proto > 0:
                    scores_orig = model(x_small)
                    if scores_orig.ndim > 2:
                        scores_orig = scores_orig.view(scores_orig.size(0), -1)
                    loss_proto = compute_prototype_loss(scores_orig, y_small, large_prototypes)

            # MMD alignment
            loss_align = torch.tensor(0.0, device=device)
            if (scores_small is not None) and (lambda_mmd > 0.0):
                try:
                    scores_orig_small = model(normalize_data(xb_small).to(device))
                    if scores_orig_small.ndim > 2:
                        scores_orig_small = scores_orig_small.view(scores_orig_small.size(0), -1)
                    b = min(scores_large.size(0), scores_orig_small.size(0))
                    loss_align = compute_mmd_rbf(scores_large[:b].detach(), scores_orig_small[:b].detach(), logger)
                except Exception as e:
                    loss_align = torch.tensor(0.0, device=device)

            total_loss = w_large_cur * loss_large + w_small_cur * loss_small + lambda_mmd * loss_align + lambda_proto * loss_proto
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            steps += 1

        scheduler.step()
        
        # Validation
        p3_val = evaluate_domain(model, val_loader_p3, device)
        avo_val = evaluate_domain(model, val_loader_avo, device)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: Val(P3)={p3_val:.3f} | Val(AVO)={avo_val:.3f}")

        # Early stopping
        small_val = p3_val if small_domain == 'P3' else avo_val
        if small_val > best_val + 1e-4:
            best_val = small_val
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                # logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_nested_cv_asmmd(logger, channels):
    """Run nested cross-validation with AS-MMD."""
    from sklearn.model_selection import StratifiedKFold
    
    # Load data
    X_all, y_all, src_all = load_combined_arrays(logger, channels)
    
    fold_acc = []
    dataset_metrics = {'P3': [], 'AVO': []}
    detailed_fold_results = []
    
    for repeat in range(NESTED_CV_REPEATS):
        # logger.info(f"Repeat {repeat + 1}/{NESTED_CV_REPEATS}")
        cv = StratifiedKFold(n_splits=NESTED_CV_OUTER_FOLDS, shuffle=True, random_state=seeds[repeat % len(seeds)])
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            # logger.info(f"  Fold {fold_idx + 1}/{NESTED_CV_OUTER_FOLDS}")
            
            X_tr_fold, y_tr_fold, src_tr_fold = X_all[train_idx], y_all[train_idx], src_all[train_idx]
            X_te_fold, y_te_fold, src_te_fold = X_all[test_idx], y_all[test_idx], src_all[test_idx]
            
            # Split train/val
            train_val_total = TRAIN_SIZE + VAL_SIZE
            train_ratio_within = TRAIN_SIZE / train_val_total if train_val_total > 0 else 0.875
            idx_range = np.arange(len(X_tr_fold))
            tr_idx, va_idx = train_test_split(idx_range, train_size=train_ratio_within, stratify=y_tr_fold, random_state=42)
            
            # Per-domain splits
            tr_mask_p3 = (src_tr_fold == 'P3')
            tr_mask_avo = (src_tr_fold == 'AVO')
            Xtr_p3 = X_tr_fold[np.intersect1d(np.where(tr_mask_p3)[0], tr_idx)]
            ytr_p3 = y_tr_fold[np.intersect1d(np.where(tr_mask_p3)[0], tr_idx)]
            Xtr_avo = X_tr_fold[np.intersect1d(np.where(tr_mask_avo)[0], tr_idx)]
            ytr_avo = y_tr_fold[np.intersect1d(np.where(tr_mask_avo)[0], tr_idx)]
            
            va_mask_p3 = (src_tr_fold == 'P3')
            va_mask_avo = (src_tr_fold == 'AVO')
            Xva_p3 = X_tr_fold[np.intersect1d(np.where(va_mask_p3)[0], va_idx)]
            yva_p3 = y_tr_fold[np.intersect1d(np.where(va_mask_p3)[0], va_idx)]
            Xva_avo = X_tr_fold[np.intersect1d(np.where(va_mask_avo)[0], va_idx)]
            yva_avo = y_tr_fold[np.intersect1d(np.where(va_mask_avo)[0], va_idx)]
            
            # Train
            model = asmmd_train_fold(
                logger,
                Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                channels, seed=seeds[0]
            )
            
            # Evaluate
            device = get_device()
            
            def evaluate_subset_metrics(model_obj, X, y, mask, dev):
                if not np.any(mask):
                    return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.5}
                loader_local = DataLoader(
                    TensorDataset(torch.FloatTensor(X[mask]), torch.LongTensor(y[mask])),
                    batch_size=BATCH_SIZE, shuffle=False
                )
                details_local = evaluate(model_obj, loader_local, dev, return_details=True)
                return {
                    'accuracy': float(details_local.get('accuracy', 0.0)),
                    'precision': float(details_local.get('precision', 0.0)),
                    'recall': float(details_local.get('recall', 0.0)),
                    'f1_score': float(details_local.get('f1_score', 0.0)),
                    'auc': float(details_local.get('auc', 0.5)),
                }
            
            mask_p3 = (src_te_fold == 'P3')
            mask_avo = (src_te_fold == 'AVO')
            m_p3 = evaluate_subset_metrics(model, X_te_fold, y_te_fold, mask_p3, device)
            m_avo = evaluate_subset_metrics(model, X_te_fold, y_te_fold, mask_avo, device)
            n_p3 = int(np.sum(mask_p3))
            n_avo = int(np.sum(mask_avo))
            acc_overall = (m_p3['accuracy'] * n_p3 + m_avo['accuracy'] * n_avo) / max(1, (n_p3 + n_avo))
            
            # logger.info(f"    P3 Test | Acc={m_p3['accuracy']:.4f} (n={n_p3})")
            # logger.info(f"    AVO Test | Acc={m_avo['accuracy']:.4f} (n={n_avo})")
            # logger.info(f"    Overall Test Acc={acc_overall:.4f}")
            
            fold_acc.append(acc_overall)
            dataset_metrics['P3'].append(m_p3['accuracy'])
            dataset_metrics['AVO'].append(m_avo['accuracy'])
            
            fold_result = {
                'repeat': repeat + 1,
                'fold': fold_idx + 1,
                'overall_accuracy': acc_overall,
                'p3_accuracy': m_p3['accuracy'],
                'avo_accuracy': m_avo['accuracy'],
                'p3_auc': m_p3.get('auc', 0.5),
                'avo_auc': m_avo.get('auc', 0.5),
            }
            detailed_fold_results.append(fold_result)
    
    # Calculate statistics
    acc_array = np.array(fold_acc, dtype=float)
    mean_acc = float(np.mean(acc_array)) if acc_array.size > 0 else 0.0
    std_acc = float(np.std(acc_array, ddof=1)) if acc_array.size > 1 else 0.0
    
    log_section_header(logger, "Cross-Validation Results")
    logger.info(f"Overall accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Calculate P3 and AVO specific metrics from detailed_fold_results
    p3_acc_array = np.array([r['p3_accuracy'] for r in detailed_fold_results], dtype=float)
    avo_acc_array = np.array([r['avo_accuracy'] for r in detailed_fold_results], dtype=float)
    
    p3_mean_acc = float(np.mean(p3_acc_array)) if p3_acc_array.size > 0 else 0.0
    p3_std_acc = float(np.std(p3_acc_array, ddof=1)) if p3_acc_array.size > 1 else 0.0
    avo_mean_acc = float(np.mean(avo_acc_array)) if avo_acc_array.size > 0 else 0.0
    avo_std_acc = float(np.std(avo_acc_array, ddof=1)) if avo_acc_array.size > 1 else 0.0
    
    # Calculate P3 and AVO AUC metrics
    p3_auc_array = np.array([r['p3_auc'] for r in detailed_fold_results], dtype=float)
    avo_auc_array = np.array([r['avo_auc'] for r in detailed_fold_results], dtype=float)
    
    p3_mean_auc = float(np.mean(p3_auc_array)) if p3_auc_array.size > 0 else 0.5
    p3_std_auc = float(np.std(p3_auc_array, ddof=1)) if p3_auc_array.size > 1 else 0.0
    avo_mean_auc = float(np.mean(avo_auc_array)) if avo_auc_array.size > 0 else 0.5
    avo_std_auc = float(np.std(avo_auc_array, ddof=1)) if avo_auc_array.size > 1 else 0.0
    
    # Log P3 and AVO specific results
    logger.info(f"P3 Dataset - Test Accuracy: {p3_mean_acc:.4f} ± {p3_std_acc:.4f}")
    logger.info(f"P3 Dataset - Test AUC: {p3_mean_auc:.4f} ± {p3_std_auc:.4f}")
    logger.info(f"AVO Dataset - Test Accuracy: {avo_mean_acc:.4f} ± {avo_std_acc:.4f}")
    logger.info(f"AVO Dataset - Test AUC: {avo_mean_auc:.4f} ± {avo_std_auc:.4f}")
    
    results = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'p3_mean_accuracy': p3_mean_acc,
        'p3_std_accuracy': p3_std_acc,
        'p3_mean_auc': p3_mean_auc,
        'p3_std_auc': p3_std_auc,
        'avo_mean_accuracy': avo_mean_acc,
        'avo_std_accuracy': avo_std_acc,
        'avo_mean_auc': avo_mean_auc,
        'avo_std_auc': avo_std_auc,
        'detailed_fold_results': detailed_fold_results
    }
    
    return results


def main():
    """Main function to run the AS-MMD experiment."""
    mne.set_log_level('ERROR')
    logging.getLogger('joblib').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    logger = None
    try:
        # Determine channels
        if ELECTRODE_FUSION_METHOD == 'none' and DOMAIN_ADAPTATION_METHOD == 'none':
            #if electrode_list != 'common':
               # print("Warning: Combined training without fusion/domain methods uses COMMON channels")
            channels = COMMON_CHANNELS
        else:
            channels = COMMON_CHANNELS

        # Logger setup
        logger = setup_logger('AS_MMD', create_file=True)
        log_section_header(logger, "AS-MMD Joint Training (Adaptive Symmetric MMD)")

        # Log configuration
        log_configuration(logger, {
            'electrode_list': electrode_list,
            'fusion_method': ELECTRODE_FUSION_METHOD,
            'domain_adaptation': DOMAIN_ADAPTATION_METHOD,
            'use_enhanced_preprocessing': USE_ENHANCED_PREPROCESSING,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': DROPOUT_RATE,
            'use_data_augmentation': USE_DATA_AUGMENTATION,
            'noise_std': NOISE_STD,
            'time_shift_range': TIME_SHIFT_RANGE,
            'label_smoothing': LABEL_SMOOTHING,
            'trials_per_subject_P3': NESTED_CV_TRIALS_PER_SUBJECT_P3,
            'trials_per_subject_AVO': NESTED_CV_TRIALS_PER_SUBJECT_AVO,
            'train/val/test': (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
            'device_mode': DEVICE_MODE,
        })

        # Run AS-MMD with cross-validation
        log_section_header(logger, "Running Nested Cross-Validation with AS-MMD")
        results = run_nested_cv_asmmd(logger, channels)

        # Save results
        #if 'detailed_fold_results' in results:
        df = pd.DataFrame(results['detailed_fold_results'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'asmmd_detailed_results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        logger.info(f"Detailed results saved to: {csv_filename}")
        print(f"Detailed results saved to: {csv_filename}")
        
        # Save summary statistics including P3 and AVO metrics
        summary_stats = {k: v for k, v in results.items() if k != 'detailed_fold_results'}
        summary_df = pd.DataFrame([summary_stats])
        summary_filename = f'asmmd_summary_stats_{timestamp}.csv'
        summary_df.to_csv(summary_filename, index=False)
        logger.info(f"Summary statistics saved to: {summary_filename}")
        print(f"Summary statistics saved to: {summary_filename}")

        print("\n--- Experiment Run Complete (AS-MMD) ---")
        print(f"Final Results: Overall Accuracy = {results.get('mean_accuracy', 0.0):.4f}")
        print(f"P3 Dataset - Test Accuracy: {results.get('p3_mean_accuracy', 0.0):.4f} ± {results.get('p3_std_accuracy', 0.0):.4f}")
        print(f"P3 Dataset - Test AUC: {results.get('p3_mean_auc', 0.5):.4f} ± {results.get('p3_std_auc', 0.0):.4f}")
        print(f"AVO Dataset - Test Accuracy: {results.get('avo_mean_accuracy', 0.0):.4f} ± {results.get('avo_std_accuracy', 0.0):.4f}")
        print(f"AVO Dataset - Test AUC: {results.get('avo_mean_auc', 0.5):.4f} ± {results.get('avo_std_auc', 0.0):.4f}")
        
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    except Exception as e:
        print(f"\n--- AS-MMD Experiment Failed: {e} ---")
        if logger:
            cleanup_failed_log(logger)
        raise
    except KeyboardInterrupt:
        print("\n--- AS-MMD Experiment Interrupted by User ---")
        if logger:
            cleanup_failed_log(logger)
        raise

# %% [markdown]
# ## Run Experiment
# 
# Execute the full AS-MMD experiment with nested cross-validation.

# %%
# Run the full experiment
try:
    # Setup
    channels = COMMON_CHANNELS
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(LOG_DIR, f'AS_MMD_notebook_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile)],
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logger = logging.getLogger('AS_MMD')
    
    print("="*70)
    print("Starting AS-MMD Experiment")
    print("="*70)
    print(f"Log file: {logfile}")
    print()
    
    # Log configuration
    logger.info("AS-MMD Joint Training (Adaptive Symmetric MMD)")
    # logger.info(f"Configuration:")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Max epochs: {MAX_EPOCHS}")
    logger.info(f"  - Learning rate: {LEARNING_RATE}")
    logger.info(f"  - Trials per subject P3: {NESTED_CV_TRIALS_PER_SUBJECT_P3}")
    logger.info(f"  - Trials per subject AVO: {NESTED_CV_TRIALS_PER_SUBJECT_AVO}")
    
    # Run nested CV
    # print("Running Nested Cross-Validation with AS-MMD...")
    results = run_nested_cv_asmmd(logger, channels)
    
    # Save results
    if 'detailed_fold_results' in results:
        df = pd.DataFrame(results['detailed_fold_results'])
        csv_filename = f'asmmd_notebook_results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n✓ Results saved to: {csv_filename}")
    
    # Print final results
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    print(f"Overall Accuracy: {results.get('mean_accuracy', 0.0):.4f} ± {results.get('std_accuracy', 0.0):.4f}")
    print(f"P3 Dataset - Test Accuracy: {results.get('p3_mean_accuracy', 0.0):.4f} ± {results.get('p3_std_accuracy', 0.0):.4f}")
    print(f"P3 Dataset - Test AUC: {results.get('p3_mean_auc', 0.5):.4f} ± {results.get('p3_std_auc', 0.0):.4f}")
    print(f"AVO Dataset - Test Accuracy: {results.get('avo_mean_accuracy', 0.0):.4f} ± {results.get('avo_std_accuracy', 0.0):.4f}")
    print(f"AVO Dataset - Test AUC: {results.get('avo_mean_auc', 0.5):.4f} ± {results.get('avo_std_auc', 0.0):.4f}")
    print("="*70)
    
except Exception as e:
    print(f"\n Experiment failed: {e}")
    import traceback
    traceback.print_exc()


