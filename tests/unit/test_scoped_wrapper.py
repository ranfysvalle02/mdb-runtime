"""
Unit tests for ScopedMongoWrapper and ScopedCollectionWrapper.

Tests data scoping, query filtering, and isolation logic.
"""

from unittest.mock import MagicMock

import pytest
from motor.motor_asyncio import AsyncIOMotorCollection

from mdb_engine.database.scoped_wrapper import (ScopedCollectionWrapper,
                                                ScopedMongoWrapper)


@pytest.mark.unit
class TestScopedCollectionWrapper:
    """Test ScopedCollectionWrapper functionality."""

    @pytest.mark.asyncio
    async def test_scoped_insert_one_adds_app_id(self, mock_mongo_collection):
        """Test that insert_one automatically adds app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        document = {"name": "Test", "value": 100}
        await wrapper.insert_one(document)

        # Verify insert_one was called with app_id added
        call_args = mock_mongo_collection.insert_one.call_args
        assert call_args is not None
        inserted_doc = call_args[0][0]
        assert inserted_doc["app_id"] == "test_app"
        assert inserted_doc["name"] == "Test"
        assert inserted_doc["value"] == 100

    @pytest.mark.asyncio
    async def test_scoped_insert_one_preserves_original_document(
        self, mock_mongo_collection
    ):
        """Test that insert_one doesn't mutate the original document."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        original_doc = {"name": "Test", "value": 100}
        await wrapper.insert_one(original_doc)

        # Original document should not have app_id
        assert "app_id" not in original_doc

    @pytest.mark.asyncio
    async def test_scoped_insert_many_adds_app_id(self, mock_mongo_collection):
        """Test that insert_many adds app_id to all documents."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        documents = [{"name": "Test 1", "value": 100}, {"name": "Test 2", "value": 200}]
        await wrapper.insert_many(documents)

        # Verify insert_many was called
        call_args = mock_mongo_collection.insert_many.call_args
        assert call_args is not None
        inserted_docs = call_args[0][0]
        assert len(inserted_docs) == 2
        assert all(doc["app_id"] == "test_app" for doc in inserted_docs)

    @pytest.mark.asyncio
    async def test_scoped_find_one_filters_by_app_id(self, mock_mongo_collection):
        """Test that find_one filters by app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"name": "Test"}
        await wrapper.find_one(user_filter)

        # Verify find_one was called with scoped filter
        call_args = mock_mongo_collection.find_one.call_args
        assert call_args is not None
        scoped_filter = call_args[0][0]

        # Should have $and with user filter and app_id scope
        assert "$and" in scoped_filter
        and_conditions = scoped_filter["$and"]
        assert len(and_conditions) == 2
        assert {"name": "Test"} in and_conditions
        assert {"app_id": {"$in": ["test_app"]}} in and_conditions

    @pytest.mark.asyncio
    async def test_scoped_find_one_empty_filter(self, mock_mongo_collection):
        """Test that find_one with no filter still applies app_id scope."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        await wrapper.find_one()

        # Verify find_one was called with app_id filter
        call_args = mock_mongo_collection.find_one.call_args
        assert call_args is not None
        scoped_filter = call_args[0][0]
        assert scoped_filter == {"app_id": {"$in": ["test_app"]}}

    @pytest.mark.asyncio
    async def test_scoped_find_filters_by_app_id(self, mock_mongo_collection):
        """Test that find filters by app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"status": "active"}
        wrapper.find(user_filter)

        # Verify find was called with scoped filter
        # Note: find returns a cursor, so we check the underlying collection call
        # The actual call happens when cursor is used, but wrapper prepares the filter
        # We can verify by checking the wrapper's filter injection logic
        scoped_filter = wrapper._inject_read_filter(user_filter)
        assert "$and" in scoped_filter
        assert {"status": "active"} in scoped_filter["$and"]
        assert {"app_id": {"$in": ["test_app"]}} in scoped_filter["$and"]

    @pytest.mark.asyncio
    async def test_scoped_find_multiple_read_scopes(self, mock_mongo_collection):
        """Test that find supports multiple read scopes."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["app1", "app2"],
            write_scope="app1",
        )

        scoped_filter = wrapper._inject_read_filter({"name": "Test"})
        assert "$and" in scoped_filter
        assert {"app_id": {"$in": ["app1", "app2"]}} in scoped_filter["$and"]

    @pytest.mark.asyncio
    async def test_scoped_update_one_filters_by_app_id(self, mock_mongo_collection):
        """Test that update_one filters by app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"name": "Test"}
        update = {"$set": {"status": "updated"}}
        await wrapper.update_one(user_filter, update)

        # Verify update_one was called with scoped filter
        call_args = mock_mongo_collection.update_one.call_args
        assert call_args is not None
        scoped_filter = call_args[0][0]

        # Should have $and with user filter and app_id scope
        assert "$and" in scoped_filter
        and_conditions = scoped_filter["$and"]
        assert {"name": "Test"} in and_conditions
        assert {"app_id": {"$in": ["test_app"]}} in and_conditions

    @pytest.mark.asyncio
    async def test_scoped_delete_one_filters_by_app_id(self, mock_mongo_collection):
        """Test that delete_one filters by app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"name": "Test"}
        await wrapper.delete_one(user_filter)

        # Verify delete_one was called with scoped filter
        call_args = mock_mongo_collection.delete_one.call_args
        assert call_args is not None
        scoped_filter = call_args[0][0]

        # Should have $and with user filter and app_id scope
        assert "$and" in scoped_filter
        and_conditions = scoped_filter["$and"]
        assert {"name": "Test"} in and_conditions
        assert {"app_id": {"$in": ["test_app"]}} in and_conditions

    @pytest.mark.asyncio
    async def test_scoped_count_documents_filters_by_app_id(
        self, mock_mongo_collection
    ):
        """Test that count_documents filters by app_id."""
        wrapper = ScopedCollectionWrapper(
            real_collection=mock_mongo_collection,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"status": "active"}
        await wrapper.count_documents(user_filter)

        # Verify count_documents was called with scoped filter
        call_args = mock_mongo_collection.count_documents.call_args
        assert call_args is not None
        scoped_filter = call_args[0][0]

        # Should have $and with user filter and app_id scope
        assert "$and" in scoped_filter
        and_conditions = scoped_filter["$and"]
        assert {"status": "active"} in and_conditions
        assert {"app_id": {"$in": ["test_app"]}} in and_conditions

    def test_inject_read_filter_empty_filter(self):
        """Test _inject_read_filter with empty filter."""
        wrapper = ScopedCollectionWrapper(
            real_collection=MagicMock(spec=AsyncIOMotorCollection),
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        result = wrapper._inject_read_filter(None)
        assert result == {"app_id": {"$in": ["test_app"]}}

        result = wrapper._inject_read_filter({})
        assert result == {"app_id": {"$in": ["test_app"]}}

    def test_inject_read_filter_with_user_filter(self):
        """Test _inject_read_filter with user filter."""
        wrapper = ScopedCollectionWrapper(
            real_collection=MagicMock(spec=AsyncIOMotorCollection),
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        user_filter = {"name": "Test", "value": {"$gt": 100}}
        result = wrapper._inject_read_filter(user_filter)

        assert "$and" in result
        assert user_filter in result["$and"]
        assert {"app_id": {"$in": ["test_app"]}} in result["$and"]


@pytest.mark.unit
class TestScopedMongoWrapper:
    """Test ScopedMongoWrapper functionality."""

    def test_get_collection_returns_scoped_wrapper(self, mock_mongo_database):
        """Test that accessing a collection returns ScopedCollectionWrapper."""
        wrapper = ScopedMongoWrapper(
            real_db=mock_mongo_database,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        collection = wrapper.test_collection
        assert isinstance(collection, ScopedCollectionWrapper)
        assert collection._write_scope == "test_app"
        assert collection._read_scopes == ["test_app"]

    def test_get_collection_caching(self, mock_mongo_database):
        """Test that collection wrappers are cached."""
        wrapper = ScopedMongoWrapper(
            real_db=mock_mongo_database,
            read_scopes=["test_app"],
            write_scope="test_app",
        )

        collection1 = wrapper.test_collection
        collection2 = wrapper.test_collection

        # Should be the same instance (cached)
        assert collection1 is collection2
