# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Local BIDS dataset interface for EEGDash.

This module provides the EEGBIDSDataset class for interfacing with local BIDS
datasets on the filesystem, parsing metadata, and retrieving BIDS-related information.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from mne_bids import BIDSPath, find_matching_paths


class EEGBIDSDataset:
    """An interface to a local BIDS dataset containing EEG recordings.

    This class centralizes interactions with a BIDS dataset on the local
    filesystem, providing methods to parse metadata, find files, and
    retrieve BIDS-related information.

    Parameters
    ----------
    data_dir : str or Path
        The path to the local BIDS dataset directory.
    dataset : str
        A name for the dataset (e.g., "ds002718").

    """

    ALLOWED_FILE_FORMAT = ["eeglab", "brainvision", "biosemi", "european"]
    RAW_EXTENSIONS = {
        ".set": [".set", ".fdt"],  # eeglab
        ".edf": [".edf"],  # european
        ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
        ".bdf": [".bdf"],  # biosemi
    }
    METADATA_FILE_EXTENSIONS = [
        "eeg.json",
        "channels.tsv",
        "electrodes.tsv",
        "events.tsv",
        "events.json",
    ]

    def __init__(
        self,
        data_dir=None,  # location of bids dataset
        dataset="",  # dataset name
    ):
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError("data_dir must be specified and must exist")

        self.bidsdir = Path(data_dir)
        self.dataset = dataset
        self.data_dir = data_dir

        # Accept exact dataset folder or a variant with informative suffixes
        # (e.g., dsXXXXX-bdf, dsXXXXX-bdf-mini) to avoid collisions.
        dir_name = self.bidsdir.name
        if not (dir_name == self.dataset or dir_name.startswith(self.dataset + "-")):
            raise AssertionError(
                f"BIDS directory '{dir_name}' does not correspond to dataset '{self.dataset}'"
            )

        # Initialize BIDS paths using fast mne_bids approach instead of pybids
        self._init_bids_paths()

        # get all recording files in the bids directory
        assert len(self.files) > 0, ValueError(
            "Unable to construct EEG dataset. No EEG recordings found."
        )
        assert self.check_eeg_dataset(), ValueError("Dataset is not an EEG dataset.")

    def check_eeg_dataset(self) -> bool:
        """Check if the BIDS dataset contains EEG data.

        Returns
        -------
        bool
            True if the dataset's modality is EEG, False otherwise.

        """
        return self.get_bids_file_attribute("modality", self.files[0]).lower() == "eeg"

    def _init_bids_paths(self) -> None:
        """Initialize BIDS file paths using mne_bids for fast discovery.

        Uses mne_bids.find_matching_paths() for efficient pattern-based file
        discovery instead of heavy pybids BIDSLayout indexing.
        """
        # Initialize cache for BIDSPath objects
        self._bids_path_cache = {}

        # Find all EEG recordings using pattern matching (fast!)
        self.files = []
        for ext in self.RAW_EXTENSIONS.keys():
            # find_matching_paths returns BIDSPath objects
            paths = find_matching_paths(self.bidsdir, datatypes="eeg", extensions=ext)
            if paths:
                # Convert BIDSPath objects to filename strings
                self.files = [str(p.fpath) for p in paths]
                break

    def _get_bids_path_from_file(self, data_filepath: str):
        """Get a BIDSPath object for a data file with caching.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        BIDSPath
            The BIDSPath object for the file.

        """
        if data_filepath not in self._bids_path_cache:
            # Parse the filename to extract BIDS entities
            filepath = Path(data_filepath)
            filename = filepath.name

            # Extract entities from filename using BIDS pattern
            # Expected format: sub-<label>[_ses-<label>][_task-<label>][_run-<label>]_eeg.<ext>
            subject = re.search(r"sub-([^_]*)", filename)
            session = re.search(r"ses-([^_]*)", filename)
            task = re.search(r"task-([^_]*)", filename)
            run = re.search(r"run-([^_]*)", filename)

            bids_path = BIDSPath(
                subject=subject.group(1) if subject else None,
                session=session.group(1) if session else None,
                task=task.group(1) if task else None,
                run=int(run.group(1)) if run else None,
                datatype="eeg",
                extension=filepath.suffix,
                root=self.bidsdir,
            )
            self._bids_path_cache[data_filepath] = bids_path

        return self._bids_path_cache[data_filepath]

    def _get_json_with_inheritance(
        self, data_filepath: str, json_filename: str
    ) -> dict:
        """Get JSON metadata with BIDS inheritance handling.

        Walks up the directory tree to find and merge JSON files following
        BIDS inheritance principles.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.
        json_filename : str
            The name of the JSON file to find (e.g., "eeg.json").

        Returns
        -------
        dict
            The merged JSON metadata.

        """
        json_dict = {}
        current_dir = Path(data_filepath).parent
        root_dir = self.bidsdir

        # Walk up from file directory to root, collecting JSON files
        while current_dir >= root_dir:
            json_path = current_dir / json_filename
            if json_path.exists():
                with open(json_path) as f:
                    json_dict.update(json.load(f))

            # Stop at BIDS root (contains dataset_description.json)
            if (current_dir / "dataset_description.json").exists():
                break

            current_dir = current_dir.parent

        return json_dict

    def _merge_json_inheritance(self, json_files: list[str | Path]) -> dict:
        """Merge a list of JSON files according to BIDS inheritance."""
        json_files.reverse()
        json_dict = {}
        for f in json_files:
            with open(f) as fp:
                json_dict.update(json.load(fp))
        return json_dict

    def _get_bids_file_inheritance(
        self, path: str | Path, basename: str, extension: str
    ) -> list[Path]:
        """Find all applicable metadata files using BIDS inheritance."""
        top_level_files = ["README", "dataset_description.json", "participants.tsv"]
        bids_files = []

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"path {path} does not exist")

        for file in os.listdir(path):
            if os.path.isfile(path / file) and file.endswith(extension):
                bids_files.append(path / file)

        if any(file in os.listdir(path) for file in top_level_files):
            return bids_files
        else:
            bids_files.extend(
                self._get_bids_file_inheritance(path.parent, basename, extension)
            )
            return bids_files

    def get_bids_metadata_files(
        self, filepath: str | Path, metadata_file_extension: str
    ) -> list[Path]:
        """Retrieve all metadata files that apply to a given data file.

        Follows the BIDS inheritance principle to find all relevant metadata
        files (e.g., ``channels.tsv``, ``eeg.json``) for a specific recording.

        Parameters
        ----------
        filepath : str or Path
            The path to the data file.
        metadata_file_extension : str
            The extension of the metadata file to search for (e.g., "channels.tsv").

        Returns
        -------
        list of Path
            A list of paths to the matching metadata files.

        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"filepath {filepath} does not exist")
        path, filename = os.path.split(filepath)
        basename = filename[: filename.rfind("_")]
        meta_files = self._get_bids_file_inheritance(
            path, basename, metadata_file_extension
        )
        return meta_files

    def get_files(self) -> list[str]:
        """Get all EEG recording file paths in the BIDS dataset.

        Returns
        -------
        list of str
            A list of file paths for all valid EEG recordings.

        """
        return self.files

    def get_bids_file_attribute(self, attribute: str, data_filepath: str) -> Any:
        """Retrieve a specific attribute from BIDS metadata.

        Parameters
        ----------
        attribute : str
            The name of the attribute to retrieve (e.g., "sfreq", "subject").
        data_filepath : str
            The path to the data file.

        Returns
        -------
        Any
            The value of the requested attribute, or None if not found.

        """
        bids_path = self._get_bids_path_from_file(data_filepath)

        # Direct BIDSPath properties for entities
        direct_attrs = {
            "subject": bids_path.subject,
            "session": bids_path.session,
            "task": bids_path.task,
            "run": bids_path.run,
            "modality": bids_path.datatype,
        }

        if attribute in direct_attrs:
            return direct_attrs[attribute]

        # For JSON-based attributes, read and cache eeg.json
        eeg_json = self._get_json_with_inheritance(data_filepath, "eeg.json")
        json_attrs = {
            "sfreq": eeg_json.get("SamplingFrequency"),
            "ntimes": eeg_json.get("RecordingDuration"),
            "nchans": eeg_json.get("EEGChannelCount"),
        }

        return json_attrs.get(attribute)

    def channel_labels(self, data_filepath: str) -> list[str]:
        """Get a list of channel labels from channels.tsv.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        list of str
            A list of channel names.

        """
        # Find channels.tsv in the same directory as the data file
        # It can be named either "channels.tsv" or "*_channels.tsv"
        filepath = Path(data_filepath)
        parent_dir = filepath.parent

        # Try the standard channels.tsv first
        channels_tsv_path = parent_dir / "channels.tsv"
        if not channels_tsv_path.exists():
            # Try to find *_channels.tsv matching the filename prefix
            base_name = filepath.stem  # filename without extension
            for tsv_file in parent_dir.glob("*_channels.tsv"):
                # Check if it matches by looking at task/run components
                tsv_name = tsv_file.stem.replace("_channels", "")
                if base_name.startswith(tsv_name):
                    channels_tsv_path = tsv_file
                    break

        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")

        channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
        return channels_tsv["name"].tolist()

    def channel_types(self, data_filepath: str) -> list[str]:
        """Get a list of channel types from channels.tsv.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        list of str
            A list of channel types.

        """
        # Find channels.tsv in the same directory as the data file
        # It can be named either "channels.tsv" or "*_channels.tsv"
        filepath = Path(data_filepath)
        parent_dir = filepath.parent

        # Try the standard channels.tsv first
        channels_tsv_path = parent_dir / "channels.tsv"
        if not channels_tsv_path.exists():
            # Try to find *_channels.tsv matching the filename prefix
            base_name = filepath.stem  # filename without extension
            for tsv_file in parent_dir.glob("*_channels.tsv"):
                # Check if it matches by looking at task/run components
                tsv_name = tsv_file.stem.replace("_channels", "")
                if base_name.startswith(tsv_name):
                    channels_tsv_path = tsv_file
                    break

        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")

        channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
        return channels_tsv["type"].tolist()

    def num_times(self, data_filepath: str) -> int:
        """Get the number of time points in the recording.

        Calculated from ``SamplingFrequency`` and ``RecordingDuration`` in eeg.json.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        int
            The approximate number of time points.

        """
        eeg_json_dict = self._get_json_with_inheritance(data_filepath, "eeg.json")
        return int(
            eeg_json_dict.get("SamplingFrequency", 0)
            * eeg_json_dict.get("RecordingDuration", 0)
        )

    def subject_participant_tsv(self, data_filepath: str) -> dict[str, Any]:
        """Get the participants.tsv record for a subject.

        Parameters
        ----------
        data_filepath : str
            The path to a data file belonging to the subject.

        Returns
        -------
        dict
            A dictionary of the subject's information from participants.tsv.

        """
        participants_tsv_path = self.get_bids_metadata_files(
            data_filepath, "participants.tsv"
        )[0]
        participants_tsv = pd.read_csv(participants_tsv_path, sep="\t")
        if participants_tsv.empty:
            return {}
        participants_tsv.set_index("participant_id", inplace=True)
        subject = f"sub-{self.get_bids_file_attribute('subject', data_filepath)}"
        return participants_tsv.loc[subject].to_dict()

    def eeg_json(self, data_filepath: str) -> dict[str, Any]:
        """Get the merged eeg.json metadata for a data file.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        dict
            The merged eeg.json metadata.

        """
        return self._get_json_with_inheritance(data_filepath, "eeg.json")


__all__ = ["EEGBIDSDataset"]
