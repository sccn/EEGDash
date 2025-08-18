from unittest.mock import MagicMock, patch

import pytest

from eegdash import EEGDash


# Mock the MongoConnectionManager to prevent actual DB connections during tests
@pytest.fixture(autouse=True)
def mock_mongo_connection():
    """Automatically mocks the MongoDB connection for all tests."""
    with patch("eegdash.mongodb.MongoConnectionManager.get_client") as mock_get_client:
        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, mock_db, mock_collection)
        yield mock_collection


@pytest.fixture
def eegdash_instance(mock_mongo_connection):
    """Provides a clean instance of EEGDash for each test."""
    return EEGDash(is_public=True)


def test_build_query_with_single_values(eegdash_instance):
    """Test 1: Validates that the query builder correctly handles simple
    key-value pairs.
    """
    kwargs = {"dataset": "ds001", "subject": "sub-01"}
    expected_query = {"dataset": "ds001", "subject": "sub-01"}

    # _build_query_from_kwargs is a protected method, but we test it
    # to ensure the core logic is sound.
    query = eegdash_instance._build_query_from_kwargs(**kwargs)

    assert query == expected_query


def test_build_query_with_list_value(eegdash_instance):
    """Test 2: Validates that the query builder correctly translates a list
    of values into a MongoDB `$in` operator.
    """
    kwargs = {"dataset": "ds002", "subject": ["sub-01", "sub-02", "sub-03"]}
    expected_query = {
        "dataset": "ds002",
        "subject": {"$in": ["sub-01", "sub-02", "sub-03"]},
    }

    query = eegdash_instance._build_query_from_kwargs(**kwargs)

    assert query == expected_query


def test_build_query_with_invalid_field(eegdash_instance):
    """Test 3: Ensures the query builder raises a ValueError when an unsupported
    query field is provided.
    """
    kwargs = {"dataset": "ds003", "invalid_field": "some_value"}

    with pytest.raises(
        ValueError, match="Unsupported query field\\(s\\): invalid_field"
    ):
        eegdash_instance._build_query_from_kwargs(**kwargs)


def test_find_method_with_kwargs(eegdash_instance, mock_mongo_connection):
    """Test 4: Verifies that the `find` method correctly uses the query builder
    and calls the underlying database collection with the constructed query.
    """
    # Mock the return value of the collection's find method
    mock_mongo_connection.find.return_value = [{"_id": "123", "dataset": "ds004"}]

    # Call the method with user-friendly kwargs
    results = eegdash_instance.find(dataset="ds004", subject=["sub-05", "sub-06"])

    # Define the query we expect to be built and passed to the DB
    expected_db_query = {"dataset": "ds004", "subject": {"$in": ["sub-05", "sub-06"]}}

    # Assert that the collection's find method was called once with the correct query
    mock_mongo_connection.find.assert_called_once_with(expected_db_query)

    # Assert that the method returned the mocked data
    assert len(results) == 1
    assert results[0]["dataset"] == "ds004"
