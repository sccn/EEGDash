import time
from pathlib import Path

import pytest

from eegdash.api import EEGDash
from eegdash.dataset import EEGChallengeDataset

RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
FILES_PER_RELEASE = [1342, 1405, 1812, 3342, 3326, 1227, 3100, 2320, 2885, 2516, 3397]

RELEASE_FILES = list(zip(RELEASES, FILES_PER_RELEASE))

CACHE_DIR = Path("~/mne_data").resolve() / "eeg_challenge_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_release(release):
    ds = EEGChallengeDataset(release=release, cache_dir=CACHE_DIR)
    getattr(ds, "description", None)
    return ds


@pytest.fixture(scope="session")
def warmed_mongo():
    try:
        EEGDash()
    except Exception:
        pytest.skip("Mongo not reachable")


def test_eeg_challenge_dataset_initialization():
    """Test the initialization of EEGChallengeDataset."""
    dataset = EEGChallengeDataset(release="R5", cache_dir=CACHE_DIR)

    release = "R5"
    expected_bucket_prefix = f"s3://nmdatasets/NeurIPS25/{release}_L100"
    assert dataset.s3_bucket == expected_bucket_prefix, (
        f"Unexpected s3_bucket: {dataset.s3_bucket} (expected {expected_bucket_prefix})"
    )

    # Expected components (kept explicit for readability & easier future edits)
    expected_dataset = "ds005509"
    expected_subject = "sub-NDARAC350XUM"
    expected_task = "DespicableMe"
    expected_suffix = (
        f"{expected_dataset}/{expected_subject}/eeg/"
        f"{expected_subject}_task-{expected_task}_eeg.set"
    )

    expected_full_path = f"{dataset.s3_bucket}/{expected_suffix}"
    first_file = dataset.datasets[0].s3file

    assert first_file == expected_full_path, (
        "Mismatch in first dataset s3 file path.\n"
        f"Got     : {first_file}\n"
        f"Expected: {expected_full_path}"
    )


@pytest.mark.parametrize("release, number_files", RELEASE_FILES)
def test_eeg_challenge_dataset_amount_files(release, number_files):
    dataset = EEGChallengeDataset(release=release, cache_dir=CACHE_DIR)
    assert len(dataset.datasets) == number_files


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_benchmark(benchmark, warmed_mongo, release):
    # Group makes the report nicer when comparing releases
    benchmark.group = "EEGChallengeDataset.load"

    result = benchmark.pedantic(
        _load_release,
        args=(release,),
        iterations=1,  # I/O-bound → 1 iteration per round
        rounds=5,  # take min/median across several cold-ish runs
        warmup_rounds=1,  # do one warmup round
    )

    assert result is not None


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_under_sometime(release):
    start_time = time.perf_counter()
    _ = EEGChallengeDataset(release=release, cache_dir=CACHE_DIR)
    duration = time.perf_counter() - start_time
    assert duration < 30, f"{release} took {duration:.2f}s"


def test_consuming_data_r5():
    dataset_obj = EEGChallengeDataset(
        release="R5",
        query=dict(task="RestingState", subject="NDARAC350XUM"),
        cache_dir=CACHE_DIR,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None
