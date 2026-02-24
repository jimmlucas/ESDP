"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require trained model)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def model_path():
    """Path to trained model"""
    return Path("models/best_model_pipeline.pkl")


@pytest.fixture(scope="session")
def model_exists(model_path):
    """Check if trained model exists"""
    return model_path.exists()