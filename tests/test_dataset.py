import time
from pathlib import Path

import pytest

from eegdash.api import EEGDash
from eegdash.dataset import EEGChallengeDataset

RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
FILES_PER_RELEASE = [1342, 1405, 1812, 3342, 3326, 1227, 3100, 2320, 2885, 2516, 3397]

RELEASE_FILES = list(zip(RELEASES, FILES_PER_RELEASE))


def _load_release(release, cache_dir: Path):
    ds = EEGChallengeDataset(release=release, mini=False, cache_dir=cache_dir)
    getattr(ds, "description", None)
    return ds


def test_eeg_challenge_dataset_initialization(cache_dir: Path):
    """Test the initialization of EEGChallengeDataset."""
    dataset = EEGChallengeDataset(release="R5", mini=False, cache_dir=cache_dir)

    release = "R5"
    expected_bucket_prefix = f"s3://nmdatasets/NeurIPS25/{release}_L100_bdf"
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
def test_eeg_challenge_dataset_amount_files(release, number_files, cache_dir: Path):
    dataset = EEGChallengeDataset(release=release, mini=False, cache_dir=cache_dir)
    assert len(dataset.datasets) == number_files


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_benchmark(benchmark, release, cache_dir: Path):
    # Group makes the report nicer when comparing releases
    benchmark.group = "EEGChallengeDataset.load"

    result = benchmark.pedantic(
        _load_release,
        args=(release, cache_dir),
        iterations=1,  # I/O-bound → 1 iteration per round
        rounds=5,  # take min/median across several cold-ish runs
        warmup_rounds=1,  # do one warmup round
    )

    assert result is not None


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_under_sometime(release, cache_dir: Path):
    start_time = time.perf_counter()
    _ = EEGChallengeDataset(release=release, cache_dir=cache_dir)
    duration = time.perf_counter() - start_time
    assert duration < 30, f"{release} took {duration:.2f}s"


@pytest.mark.parametrize("mini", [True, False])
@pytest.mark.parametrize("release", RELEASES)
def test_consuming_one_raw(release, mini, cache_dir: Path):
    print(f"Testing release {release} mini={mini} with cache dir {cache_dir}")
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=cache_dir,
        mini=mini,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None


@pytest.mark.parametrize("eeg_dash_instance", [None, EEGDash()])
def test_eeg_dash_integration(
    eeg_dash_instance, cache_dir: Path, release="R5", mini=True
):
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=cache_dir,
        mini=mini,
        eeg_dash_instance=eeg_dash_instance,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None
