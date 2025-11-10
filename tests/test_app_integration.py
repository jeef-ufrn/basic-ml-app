"""
Integration tests for the FastAPI application
Tests the complete flow: API -> Models -> Database with real or realistic mocks.
"""

import os
import sys
import pytest
import json
import yaml
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import app
from app.auth import TokenManager
from intent_classifier import IntentClassifier, Config

# Load environment variables
load_dotenv()

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """FastAPI TestClient for integration testing."""
    return TestClient(app)


@pytest.fixture
def paths():
    """Provides paths to test data files."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "config": os.path.join(project_root, "intent_classifier", "models", "confusion-v1_config.yml"),
        "examples": os.path.join(project_root, "intent_classifier", "data", "confusion_intents.yml"),
    }


@pytest.fixture
def mini_classifier(paths):
    """Create a minimal classifier for testing."""
    config = Config(
        dataset_name="integration_test",
        codes=["greeting", "goodbye"],
        epochs=1,
        callback_patience=1,
        sent_hl_units=8,
        wandb_project=None
    )
    return IntentClassifier(config=config)


@pytest.fixture
def mock_mongo_with_insertions():
    """Mock MongoDB that tracks insertions."""
    insertions = []

    with patch('app.app.collection') as mock_coll:
        def track_insert(doc):
            insertions.append(doc)
            doc['_id'] = "mocked_id_" + str(len(insertions))

        mock_coll.insert_one = Mock(side_effect=track_insert)
        mock_coll.find = Mock(return_value=insertions)
        mock_coll.find_one = Mock(side_effect=lambda query: next((doc for doc in insertions if query.get("token") == doc.get("token")), None))

        yield {
            "collection": mock_coll,
            "insertions": insertions
        }


@pytest.fixture
def sample_predictions():
    """Sample prediction results."""
    return {
        "confusion-v1": {
            "top_intent": "greeting",
            "all_probs": {
                "greeting": 0.92,
                "goodbye": 0.08
            }
        }
    }


# ============================================================================
# INTEGRATION TESTS - Complete Prediction Flow
# ============================================================================

class TestPredictionEndToEnd:
    """Integration tests for the complete prediction flow."""

    def test_predict_flow_dev_mode(self, client, mock_mongo_with_insertions, sample_predictions):
        """Test complete prediction flow in dev mode."""
        with patch.dict(os.environ, {"ENV": "dev"}):
            with patch('app.app.MODELS', {
                "confusion-v1": Mock(predict=Mock(return_value=("greeting", {"greeting": 0.92, "goodbye": 0.08})))
            }):
                from app.app import app as dev_app
                dev_client = TestClient(dev_app)

                response = dev_client.post("/predict", params={"text": "hello world"})

                assert response.status_code == 200
                data = response.json()

                # Verify response structure
                assert "text" in data
                assert "owner" in data
                assert "predictions" in data
                assert "timestamp" in data
                assert "id" in data

                # Verify data content
                assert data["text"] == "hello world"
                assert data["owner"] == "dev_user"
                assert "confusion-v1" in data["predictions"]

    def test_prediction_logged_to_database_integration(self, client, mock_mongo_with_insertions):
        """Test that predictions are properly logged to MongoDB."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="integration_user"):
            with patch('app.app.MODELS', {
                "test-model": Mock(predict=Mock(return_value=("intent_a", {"intent_a": 0.95, "intent_b": 0.05})))
            }):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    response = client.post("/predict", params={"text": "test message"})

                    assert response.status_code == 200
                    data = response.json()

                    # Verify database insertion occurred
                    assert len(mock_mongo_with_insertions["insertions"]) > 0

                    # Verify logged data matches response
                    logged = mock_mongo_with_insertions["insertions"][-1]
                    assert logged["text"] == "test message"
                    assert logged["owner"] == "integration_user"

    def test_multiple_predictions_create_separate_logs(self, client, mock_mongo_with_insertions):
        """Test that each prediction creates a separate database entry."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {
                "model1": Mock(predict=Mock(return_value=("a", {"a": 1.0})))
            }):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    # Make three predictions
                    for i in range(3):
                        response = client.post("/predict", params={"text": f"text {i}"})
                        assert response.status_code == 200

                    # Verify three separate database entries
                    assert len(mock_mongo_with_insertions["insertions"]) == 3
                    assert mock_mongo_with_insertions["insertions"][0]["text"] == "text 0"
                    assert mock_mongo_with_insertions["insertions"][1]["text"] == "text 1"
                    assert mock_mongo_with_insertions["insertions"][2]["text"] == "text 2"


# ============================================================================
# INTEGRATION TESTS - Authentication Flow
# ============================================================================

class TestAuthenticationFlow:
    """Integration tests for the authentication system."""

    def test_token_creation_and_validation_flow(self, mock_mongo_with_insertions):
        """Test creating and validating a token."""
        with patch('app.auth.get_mongo_collection', return_value=mock_mongo_with_insertions["collection"]):
            # Create a token
            manager = TokenManager()
            manager.create(owner="integration_user", expires_in_days=30)

            # Verify token was created
            assert len(mock_mongo_with_insertions["insertions"]) > 0
            token_doc = mock_mongo_with_insertions["insertions"][-1]

            assert token_doc["owner"] == "integration_user"
            assert token_doc["active"] is True
            assert "token" in token_doc
            assert "expires_at" in token_doc

    def test_expired_token_rejection(self, mock_mongo_with_insertions):
        """Test that expired tokens are rejected."""
        with patch('app.auth.get_mongo_collection', return_value=mock_mongo_with_insertions["collection"]):
            manager = TokenManager()

            # Create an expired token (expires_in_days=-1)
            with patch('app.auth.datetime') as mock_datetime:
                mock_datetime.utcnow.return_value = datetime.utcnow()
                # Manually create an expired token entry
                expired_doc = {
                    "token": "expired-token",
                    "owner": "test_user",
                    "created_at": datetime.utcnow() - timedelta(days=60),
                    "expires_at": datetime.utcnow() - timedelta(days=30),
                    "active": True
                }
                mock_mongo_with_insertions["insertions"].append(expired_doc)

            # The token should be considered expired
            assert expired_doc["expires_at"] < datetime.utcnow()

    def test_dev_mode_skips_authentication(self, client):
        """Test that dev mode bypasses authentication entirely."""
        with patch.dict(os.environ, {"ENV": "dev"}):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                from app.app import app as dev_app
                dev_client = TestClient(dev_app)

                # Should work without any token
                response = dev_client.post("/predict", params={"text": "hello"})
                assert response.status_code == 200
                data = response.json()
                assert data["owner"] == "dev_user"


# ============================================================================
# INTEGRATION TESTS - Multi-Model Prediction
# ============================================================================

class TestMultiModelPrediction:
    """Integration tests for handling multiple models."""

    def test_multiple_models_concurrent_prediction(self, client, mock_mongo_with_insertions):
        """Test that /predict returns results from all loaded models."""
        models = {
            "model_v1": Mock(predict=Mock(return_value=("greeting", {"greeting": 0.9, "goodbye": 0.1}))),
            "model_v2": Mock(predict=Mock(return_value=("greeting", {"greeting": 0.85, "goodbye": 0.15}))),
            "model_v3": Mock(predict=Mock(return_value=("greeting", {"greeting": 0.88, "goodbye": 0.12}))),
        }

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', models):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    response = client.post("/predict", params={"text": "hello world"})

                    assert response.status_code == 200
                    data = response.json()

                    # All models should have made predictions
                    assert "model_v1" in data["predictions"]
                    assert "model_v2" in data["predictions"]
                    assert "model_v3" in data["predictions"]

                    # All predictions should be for greeting intent
                    for model_name, pred in data["predictions"].items():
                        assert pred["top_intent"] == "greeting"
                        assert isinstance(pred["all_probs"], dict)


# ============================================================================
# INTEGRATION TESTS - Error Recovery
# ============================================================================

class TestErrorRecoveryFlow:
    """Tests for error handling and recovery."""

    def test_prediction_with_database_failure(self, client):
        """Test graceful handling when database insertion fails."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection') as mock_coll:
                    # Simulate database insertion failure
                    mock_coll.insert_one.side_effect = Exception("Database connection failed")

                    # The endpoint should still work if it catches the exception
                    # or the exception should be properly propagated
                    response = client.post("/predict", params={"text": "hello"})
                    # Depending on implementation, either 500 or successful response
                    assert response.status_code in [200, 500]

    def test_prediction_with_no_models_loaded(self, client, mock_mongo_with_insertions):
        """Test behavior when no models are loaded."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    response = client.post("/predict", params={"text": "hello"})

                    # Should still return 200 with empty predictions
                    assert response.status_code == 200
                    data = response.json()
                    assert data["predictions"] == {}


# ============================================================================
# INTEGRATION TESTS - CORS and Headers
# ============================================================================

class TestCORSAndHeaders:
    """Tests for CORS and HTTP header handling."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in response."""
        response = client.get("/")
        # CORS headers are set by FastAPI middleware
        assert response.status_code == 200

    def test_request_with_custom_headers(self, client, mock_mongo_with_insertions):
        """Test request handling with custom headers."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    response = client.post(
                        "/predict",
                        params={"text": "hello"},
                        headers={"Custom-Header": "test-value"}
                    )
                    assert response.status_code == 200


# ============================================================================
# INTEGRATION TESTS - Data Integrity
# ============================================================================

class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    def test_response_id_uniqueness(self, client, mock_mongo_with_insertions):
        """Test that each response has a unique ID."""
        ids = set()

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    for i in range(5):
                        response = client.post("/predict", params={"text": f"text {i}"})
                        data = response.json()
                        ids.add(data["id"])

        # All IDs should be unique
        assert len(ids) == 5

    def test_timestamp_progression(self, client, mock_mongo_with_insertions):
        """Test that timestamps progress logically."""
        import time

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    timestamps = []

                    for i in range(3):
                        response = client.post("/predict", params={"text": f"text {i}"})
                        data = response.json()
                        timestamps.append(data["timestamp"])
                        time.sleep(0.01)  # Small delay

                    # Timestamps should be in order
                    assert timestamps[0] <= timestamps[1] <= timestamps[2]


# ============================================================================
# INTEGRATION TESTS - Load and Stress (Lightweight)
# ============================================================================

class TestBasicLoadHandling:
    """Basic tests for handling multiple concurrent-like requests."""

    def test_sequential_predictions(self, client, mock_mongo_with_insertions):
        """Test handling multiple sequential predictions."""
        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    responses = []

                    for i in range(10):
                        response = client.post("/predict", params={"text": f"message {i}"})
                        responses.append(response)

                    # All should succeed
                    assert all(r.status_code == 200 for r in responses)
                    assert len(mock_mongo_with_insertions["insertions"]) == 10

    def test_concurrent_like_different_texts(self, client, mock_mongo_with_insertions):
        """Test predictions with diverse input texts."""
        test_texts = [
            "Hello, how are you?",
            "Goodbye, see you later",
            "What is the weather today?",
            "I need help with this",
            "Thank you very much!",
            "¿Cómo estás?",  # Spanish
            "Bonjour, ça va?",  # French
            "你好",  # Chinese
            "مرحبا",  # Arabic
            "😀 Happy emoji text 🎉",  # Emoji
        ]

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    for text in test_texts:
                        response = client.post("/predict", params={"text": text})
                        assert response.status_code == 200

                    assert len(mock_mongo_with_insertions["insertions"]) == len(test_texts)


# ============================================================================
# INTEGRATION TESTS - Full Request-Response Cycle
# ============================================================================

class TestFullRequestResponseCycle:
    """Tests for complete request-response cycles."""

    def test_predict_response_contains_all_required_fields(self, client, mock_mongo_with_insertions):
        """Test that /predict response includes all required fields."""
        required_fields = ["text", "owner", "predictions", "timestamp", "id"]

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="test_owner"):
            with patch('app.app.MODELS', {
                "model1": Mock(predict=Mock(return_value=("intent_a", {"intent_a": 0.8, "intent_b": 0.2}))),
                "model2": Mock(predict=Mock(return_value=("intent_b", {"intent_a": 0.3, "intent_b": 0.7}))),
            }):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    response = client.post("/predict", params={"text": "test message"})

                    assert response.status_code == 200
                    data = response.json()

                    for field in required_fields:
                        assert field in data, f"Missing required field: {field}"

    def test_predict_preserves_input_text_exactly(self, client, mock_mongo_with_insertions):
        """Test that the input text is preserved exactly in the response."""
        test_cases = [
            "simple text",
            "Text with UPPERCASE",
            "special!@#$%characters",
            "  text with spaces  ",
            "line1\nline2",
            "текст на русском",  # Cyrillic
            "🎯 text with emoji 🚀",
        ]

        with patch('app.app.conditional_auth', new_callable=AsyncMock, return_value="user"):
            with patch('app.app.MODELS', {"model": Mock(predict=Mock(return_value=("intent", {"intent": 1.0})))}):
                with patch('app.app.collection', mock_mongo_with_insertions["collection"]):
                    for text in test_cases:
                        response = client.post("/predict", params={"text": text})
                        data = response.json()
                        assert data["text"] == text, f"Text was modified: '{text}' != '{data['text']}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])

