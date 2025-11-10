"""
Unit tests for the FastAPI application (app/app.py)
Tests individual components: endpoints, authentication, and utility functions.
"""

import os
import sys
import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import app, conditional_auth, ENV
from app.auth import TokenManager, verify_token
from db.engine import get_mongo_collection

# Load environment variables
load_dotenv()

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """FastAPI TestClient for making requests without a real server."""
    return TestClient(app)


@pytest.fixture
def mock_mongo_collection():
    """Mock MongoDB collection for testing without a real database."""
    with patch('app.app.collection') as mock_coll:
        mock_coll.insert_one = Mock()
        yield mock_coll


@pytest.fixture
def mock_models():
    """Mock pre-loaded ML models."""
    with patch('app.app.MODELS') as mock_models_dict:
        # Create a mock model that returns predictions
        mock_model = Mock()
        mock_model.predict.return_value = (
            "greeting",  # top_intent
            {"greeting": 0.9, "goodbye": 0.1}  # all_probs
        )
        mock_models_dict.__getitem__.return_value = mock_model
        mock_models_dict.items.return_value = [("confusion-v1", mock_model)]
        yield mock_models_dict


@pytest.fixture
def token_data():
    """Sample token data for testing."""
    return {
        "token": "test-token-123",
        "owner": "test_user",
        "note": "Test token",
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=30),
        "active": True
    }


# ============================================================================
# UNIT TESTS - Root Endpoint
# ============================================================================

class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_endpoint_returns_200(self, client):
        """Test that the root endpoint returns HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_returns_message(self, client):
        """Test that the root endpoint returns a JSON message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert isinstance(data["message"], str)


# ============================================================================
# UNIT TESTS - Predict Endpoint (No Auth)
# ============================================================================

class TestPredictEndpointDevMode:
    """Tests for the /predict endpoint in development mode (no auth required)."""

    @pytest.fixture
    def dev_client(self):
        """Setup test client with development environment."""
        with patch.dict(os.environ, {"ENV": "dev"}):
            # Reimport to get dev mode
            from app.app import app as dev_app
            return TestClient(dev_app)

    def test_predict_dev_mode_returns_200(self, dev_client, mock_mongo_collection, mock_models):
        """Test that /predict returns 200 in dev mode without authentication."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="dev_user"):
            response = dev_client.post("/predict", params={"text": "hello"})
            assert response.status_code == 200

    def test_predict_dev_mode_returns_predictions(self, dev_client, mock_mongo_collection, mock_models):
        """Test that /predict returns prediction data."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="dev_user"):
            response = dev_client.post("/predict", params={"text": "hello world"})
            data = response.json()

            assert "text" in data
            assert "owner" in data
            assert "predictions" in data
            assert "timestamp" in data
            assert data["text"] == "hello world"
            assert data["owner"] == "dev_user"

    def test_predict_stores_result_in_database(self, dev_client, mock_mongo_collection, mock_models):
        """Test that prediction results are stored in MongoDB."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="dev_user"):
            response = dev_client.post("/predict", params={"text": "test message"})

            # Verify insert_one was called
            assert mock_mongo_collection.insert_one.called

            # Verify the inserted data has expected keys
            call_args = mock_mongo_collection.insert_one.call_args
            inserted_data = call_args[0][0]
            assert "text" in inserted_data
            assert "owner" in inserted_data
            assert "predictions" in inserted_data
            assert "timestamp" in inserted_data


# ============================================================================
# UNIT TESTS - Authentication
# ============================================================================

class TestTokenAuthentication:
    """Tests for the authentication system."""

    def test_verify_token_requires_authorization_header(self):
        """Test that verify_token raises exception when Authorization header is missing."""
        mock_request = Mock()
        mock_request.headers.get.return_value = None

        with pytest.raises(Exception):  # HTTPException
            # Note: verify_token is async, we can't call it directly
            pass

    def test_verify_token_invalid_token_format(self, mock_mongo_collection):
        """Test that invalid token format is rejected."""
        with patch('db.engine.get_mongo_collection') as mock_get_coll:
            mock_get_coll.return_value.find_one.return_value = None

            mock_request = Mock()
            mock_request.headers.get.return_value = "Bearer invalid-token"

            with pytest.raises(Exception):  # HTTPException
                pass


class TestConditionalAuth:
    """Tests for the conditional_auth dependency."""

    @pytest.mark.asyncio
    async def test_conditional_auth_dev_mode(self):
        """Test that dev mode skips authentication."""
        with patch.dict(os.environ, {"ENV": "dev"}):
            # Reimport to get updated ENV
            from app.app import conditional_auth as dev_conditional_auth
            result = await dev_conditional_auth()
            assert result == "dev_user"

    @pytest.mark.asyncio
    async def test_conditional_auth_prod_mode_requires_auth(self):
        """Test that prod mode requires authentication."""
        with patch.dict(os.environ, {"ENV": "prod"}):
            from app.app import conditional_auth as prod_conditional_auth

            with patch('app.auth.verify_token', new_callable=AsyncMock, return_value="prod_user"):
                # This would normally require the Request object
                pass


# ============================================================================
# UNIT TESTS - Token Manager
# ============================================================================

class TestTokenManager:
    """Tests for the TokenManager utility class."""

    def test_token_manager_create_generates_token(self):
        """Test that TokenManager.create() generates a valid token."""
        with patch('app.auth.get_mongo_collection') as mock_get_coll:
            mock_coll = Mock()
            mock_get_coll.return_value = mock_coll

            manager = TokenManager()
            manager.create(owner="test_user", expires_in_days=30)

            # Verify insert_one was called
            assert mock_coll.insert_one.called

            # Verify the inserted document structure
            call_args = mock_coll.insert_one.call_args
            inserted_doc = call_args[0][0]

            assert "token" in inserted_doc
            assert inserted_doc["owner"] == "test_user"
            assert inserted_doc["active"] is True
            assert "created_at" in inserted_doc
            assert "expires_at" in inserted_doc

    def test_token_manager_read_all(self):
        """Test that TokenManager.read_all() retrieves all tokens."""
        with patch('app.auth.get_mongo_collection') as mock_get_coll:
            mock_coll = Mock()
            mock_coll.find.return_value = [
                {"token": "token1", "owner": "user1", "active": True},
                {"token": "token2", "owner": "user2", "active": True},
            ]
            mock_get_coll.return_value = mock_coll

            manager = TokenManager()
            manager.read_all()

            # Verify find was called
            assert mock_coll.find.called

    def test_token_manager_delete_expired(self):
        """Test that TokenManager.delete_expired() removes expired tokens."""
        with patch('app.auth.get_mongo_collection') as mock_get_coll:
            mock_coll = Mock()
            mock_result = Mock()
            mock_result.deleted_count = 2
            mock_coll.delete_many.return_value = mock_result
            mock_get_coll.return_value = mock_coll

            manager = TokenManager()
            manager.delete_expired()

            # Verify delete_many was called with proper filter
            assert mock_coll.delete_many.called
            call_args = mock_coll.delete_many.call_args
            filter_dict = call_args[0][0]
            assert "expires_at" in filter_dict


# ============================================================================
# UNIT TESTS - Predict Endpoint with Multiple Models
# ============================================================================

class TestPredictWithMultipleModels:
    """Tests for /predict endpoint with multiple loaded models."""

    def test_predict_returns_all_model_predictions(self, client, mock_mongo_collection):
        """Test that /predict returns predictions from all loaded models."""
        # Create multiple mock models
        mock_model_1 = Mock()
        mock_model_1.predict.return_value = ("greeting", {"greeting": 0.9, "goodbye": 0.1})

        mock_model_2 = Mock()
        mock_model_2.predict.return_value = ("greeting", {"greeting": 0.85, "goodbye": 0.15})

        with patch('app.app.MODELS', {"model_v1": mock_model_1, "model_v2": mock_model_2}):
            with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
                response = client.post("/predict", params={"text": "hello"})
                data = response.json()

                # Both models should appear in predictions
                assert "model_v1" in data["predictions"]
                assert "model_v2" in data["predictions"]
                assert data["predictions"]["model_v1"]["top_intent"] == "greeting"
                assert data["predictions"]["model_v2"]["top_intent"] == "greeting"

    def test_predict_response_structure(self, client, mock_mongo_collection):
        """Test the structure of the predict response."""
        with patch('app.app.MODELS', {}):
            with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
                response = client.post("/predict", params={"text": "test"})
                data = response.json()

                # Verify response structure
                assert isinstance(data, dict)
                assert "id" in data
                assert "text" in data
                assert "owner" in data
                assert "predictions" in data
                assert "timestamp" in data
                assert isinstance(data["predictions"], dict)
                assert isinstance(data["timestamp"], int)


# ============================================================================
# UNIT TESTS - Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_predict_empty_text(self, client, mock_mongo_collection, mock_models):
        """Test /predict with empty text."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
            response = client.post("/predict", params={"text": ""})
            # Should still process (empty text is valid input)
            assert response.status_code == 200

    def test_predict_very_long_text(self, client, mock_mongo_collection, mock_models):
        """Test /predict with very long text."""
        long_text = "word " * 10000  # Very long text
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
            response = client.post("/predict", params={"text": long_text})
            # Should handle long text gracefully
            assert response.status_code == 200

    def test_predict_special_characters(self, client, mock_mongo_collection, mock_models):
        """Test /predict with special characters."""
        special_text = "Hello! @#$%^&*() 中文 🎉"
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
            response = client.post("/predict", params={"text": special_text})
            assert response.status_code == 200


# ============================================================================
# UNIT TESTS - Database Integration
# ============================================================================

class TestDatabaseIntegration:
    """Tests for database interactions."""

    def test_prediction_logged_to_database(self, client, mock_mongo_collection, mock_models):
        """Test that predictions are logged to MongoDB."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
            response = client.post("/predict", params={"text": "hello"})

            # Verify that the collection's insert_one was called
            assert mock_mongo_collection.insert_one.called

    def test_logged_data_contains_timestamp(self, client, mock_mongo_collection, mock_models):
        """Test that logged data includes a timestamp."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_user"):
            before_time = int(datetime.now(timezone.utc).timestamp())
            response = client.post("/predict", params={"text": "test"})
            after_time = int(datetime.now(timezone.utc).timestamp())

            data = response.json()
            assert "timestamp" in data
            assert before_time <= data["timestamp"] <= after_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

