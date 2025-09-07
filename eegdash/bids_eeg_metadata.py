import logging
import re
from pathlib import Path
from typing import Any

from .const import ALLOWED_QUERY_FIELDS
from .const import config as data_config

logger = logging.getLogger("eegdash")

__all__ = [
    "build_query_from_kwargs",
    "load_eeg_attrs_from_bids_file",
    "merge_participants_fields",
    "normalize_key",
]


def build_query_from_kwargs(**kwargs) -> dict[str, Any]:
    """Build and validate a MongoDB query from user-friendly keyword arguments.

    Improvements:
    - Reject None values and empty/whitespace-only strings
    - For list/tuple/set values: strip strings, drop None/empties, deduplicate, and use `$in`
    - Preserve scalars as exact matches
    """
    # 1. Validate that all provided keys are allowed for querying
    unknown_fields = set(kwargs.keys()) - ALLOWED_QUERY_FIELDS
    if unknown_fields:
        raise ValueError(
            f"Unsupported query field(s): {', '.join(sorted(unknown_fields))}. "
            f"Allowed fields are: {', '.join(sorted(ALLOWED_QUERY_FIELDS))}"
        )

    # 2. Construct the query dictionary
    query = {}
    for key, value in kwargs.items():
        # None is not a valid constraint
        if value is None:
            raise ValueError(
                f"Received None for query parameter '{key}'. Provide a concrete value."
            )

        # Handle list-like values as multi-constraints
        if isinstance(value, (list, tuple, set)):
            cleaned: list[Any] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    item = item.strip()
                    if not item:
                        continue
                cleaned.append(item)
            # Deduplicate while preserving order
            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                raise ValueError(
                    f"Received an empty list for query parameter '{key}'. This is not supported."
                )
            query[key] = {"$in": cleaned}
        else:
            # Scalars: trim strings and validate
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise ValueError(
                        f"Received an empty string for query parameter '{key}'."
                    )
            query[key] = value

    return query


def _get_raw_extensions(bids_file: str, bids_dataset) -> list[str]:
    """Helper to find paths to additional "sidecar" files that may be associated
    with a given main data file in a BIDS dataset; paths are returned as relative to
    the parent dataset path.

    For example, if the input file is a .set file, this will return the relative path
    to a corresponding .fdt file (if any).
    """
    bids_file = Path(bids_file)
    extensions = {
        ".set": [".set", ".fdt"],  # eeglab
        ".edf": [".edf"],  # european
        ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
        ".bdf": [".bdf"],  # biosemi
    }
    return [
        str(bids_dataset._get_relative_bidspath(bids_file.with_suffix(suffix)))
        for suffix in extensions[bids_file.suffix]
        if bids_file.with_suffix(suffix).exists()
    ]


def load_eeg_attrs_from_bids_file(bids_dataset, bids_file: str) -> dict[str, Any]:
    """Build the metadata record for a given BIDS file (single recording) in a BIDS dataset.

    Attributes are at least the ones defined in data_config attributes (set to None if missing),
    but are typically a superset, and include, among others, the paths to relevant
    meta-data files needed to load and interpret the file in question.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object containing the file.
    bids_file : str
        The path to the BIDS file within the dataset.

    Returns
    -------
    dict:
        A dictionary representing the metadata record for the given file. This is the
        same format as the records stored in the database.

    """
    if bids_file not in bids_dataset.files:
        raise ValueError(f"{bids_file} not in {bids_dataset.dataset}")

    # Initialize attrs with None values for all expected fields
    attrs = {field: None for field in data_config["attributes"].keys()}

    file = Path(bids_file).name
    dsnumber = bids_dataset.dataset
    # extract openneuro path by finding the first occurrence of the dataset name in the filename and remove the path before that
    openneuro_path = dsnumber + bids_file.split(dsnumber)[1]

    # Update with actual values where available
    try:
        participants_tsv = bids_dataset.subject_participant_tsv(bids_file)
    except Exception as e:
        logger.error("Error getting participants_tsv: %s", str(e))
        participants_tsv = None

    try:
        eeg_json = bids_dataset.eeg_json(bids_file)
    except Exception as e:
        logger.error("Error getting eeg_json: %s", str(e))
        eeg_json = None

    bids_dependencies_files = data_config["bids_dependencies_files"]
    bidsdependencies = []
    for extension in bids_dependencies_files:
        try:
            dep_path = bids_dataset.get_bids_metadata_files(bids_file, extension)
            dep_path = [
                str(bids_dataset.get_relative_bidspath(dep)) for dep in dep_path
            ]
            bidsdependencies.extend(dep_path)
        except Exception:
            pass

    bidsdependencies.extend(_get_raw_extensions(bids_file, bids_dataset))

    # Define field extraction functions with error handling
    field_extractors = {
        "data_name": lambda: f"{bids_dataset.dataset}_{file}",
        "dataset": lambda: bids_dataset.dataset,
        "bidspath": lambda: openneuro_path,
        "subject": lambda: bids_dataset.get_bids_file_attribute("subject", bids_file),
        "task": lambda: bids_dataset.get_bids_file_attribute("task", bids_file),
        "session": lambda: bids_dataset.get_bids_file_attribute("session", bids_file),
        "run": lambda: bids_dataset.get_bids_file_attribute("run", bids_file),
        "modality": lambda: bids_dataset.get_bids_file_attribute("modality", bids_file),
        "sampling_frequency": lambda: bids_dataset.get_bids_file_attribute(
            "sfreq", bids_file
        ),
        "nchans": lambda: bids_dataset.get_bids_file_attribute("nchans", bids_file),
        "ntimes": lambda: bids_dataset.get_bids_file_attribute("ntimes", bids_file),
        "participant_tsv": lambda: participants_tsv,
        "eeg_json": lambda: eeg_json,
        "bidsdependencies": lambda: bidsdependencies,
    }

    # Dynamically populate attrs with error handling
    for field, extractor in field_extractors.items():
        try:
            attrs[field] = extractor()
        except Exception as e:
            logger.error("Error extracting %s : %s", field, str(e))
            attrs[field] = None

    return attrs


def normalize_key(key: str) -> str:
    """Normalize a metadata key for robust matching.

    Lowercase and replace non-alphanumeric characters with underscores, then strip
    leading/trailing underscores. This allows tolerant matching such as
    "p-factor" ≈ "p_factor" ≈ "P Factor".
    """
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def merge_participants_fields(
    description: dict[str, Any],
    participants_row: dict[str, Any] | None,
    description_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Merge participants.tsv fields into a dataset description dictionary.

    - Preserves existing entries in ``description`` (no overwrites).
    - Fills requested ``description_fields`` first, preserving their original names.
    - Adds all remaining participants columns generically using normalized keys
      unless a matching requested field already captured them.

    Parameters
    ----------
    description : dict
        Current description to be enriched in-place and returned.
    participants_row : dict | None
        A mapping of participants.tsv columns for the current subject.
    description_fields : list[str] | None
        Optional list of requested description fields. When provided, matching is
        performed by normalized names; the original requested field names are kept.

    Returns
    -------
    dict
        The enriched description (same object as input for convenience).

    """
    if not isinstance(description, dict) or not isinstance(participants_row, dict):
        return description

    # Normalize participants keys and keep first non-None value per normalized key
    norm_map: dict[str, Any] = {}
    for part_key, part_value in participants_row.items():
        norm_key = normalize_key(part_key)
        if norm_key not in norm_map and part_value is not None:
            norm_map[norm_key] = part_value

    # Ensure description_fields is a list for matching
    requested = list(description_fields or [])

    # 1) Fill requested fields first using normalized matching, preserving names
    for key in requested:
        if key in description:
            continue
        requested_norm_key = normalize_key(key)
        if requested_norm_key in norm_map:
            description[key] = norm_map[requested_norm_key]

    # 2) Add remaining participants columns generically under normalized names,
    #    unless a requested field already captured them
    requested_norm = {normalize_key(k) for k in requested}
    for norm_key, part_value in norm_map.items():
        if norm_key in requested_norm:
            continue
        if norm_key not in description:
            description[norm_key] = part_value
    return description
