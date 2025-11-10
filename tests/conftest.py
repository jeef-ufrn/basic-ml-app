"""
Shared pytest configuration and fixtures for all tests.
This file is automatically discovered and loaded by pytest.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def project_root_fixture():
    """Provides the project root directory."""
    return str(project_root)


@pytest.fixture
def disable_mongodb_on_startup(monkeypatch):
    """Fixture to disable MongoDB connection on startup for unit tests."""
    # This prevents connection errors during unit tests
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/test")
    monkeypatch.setenv("MONGO_DB", "test_db")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: integration test marker")
    config.addinivalue_line("markers", "unit: unit test marker")
    config.addinivalue_line("markers", "slow: slow test marker")

