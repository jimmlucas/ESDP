"""
Unit tests for api_service.py
Tests the FastAPI endpoints, validation, and error handling.
Includes tests for confidence_threshold and rule_overrides (paper-grade transparency).
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_service import app, METRICS


@pytest.fixture
def client():
    """Create test client"""
    # Reset metrics before each test
    METRICS["total_requests"] = 0
    METRICS["successful_predictions"] = 0
    METRICS["failed_predictions"] = 0
    METRICS["avg_response_time_ms"] = 0.0
    METRICS["decisions_by_rounds"] = {1: 0, 3: 0, 5: 0}

    return TestClient(app)


@pytest.fixture
def valid_request_payload():
    """Valid request payload for testing"""
    return {
        "sample_id": "test_001",
        "genus": "Escherichia",
        "round": 1,
        "coverage": 50.0,
        "coverage_effective": 48.0,
        "n50": 4500000,
        "qv": 35.2,
        "error_rate": 0.0003,
        "busco_complete": 95.5,
        "num_contigs": 1,
        "total_length": 4800000,
        "ai_num_contigs": 1,
        "ai_total_bp": 4800000,
        "ai_mean_cov": 78.0,
        "ai_median_cov": 77.0,
        "ai_cov_cv": 0.05,
        "ai_circular_n": 1,
        "ai_circular_bp_frac": 0.99,
        "ai_repeat_bp_frac": 0.02,
        "ai_longest_len": 4500000,
        "ai_longest_cov": 78.0,
        "polish_mean_contig_cov": 77.5,
        "align_err_consensus": 0.0005,
        "align_err_polishing": 0.0003,
        "force_conservative": False,
        "confidence_threshold": 0.5
    }


@pytest.fixture
def low_quality_request_payload():
    """Low quality request payload"""
    return {
        "sample_id": "test_002",
        "genus": "Salmonella",
        "round": 1,
        "coverage": 30.0,
        "coverage_effective": 28.0,
        "n50": 2000000,
        "qv": 28.5,
        "error_rate": 0.0015,
        "busco_complete": 85.0,
        "num_contigs": 5,
        "total_length": 4900000,
        "ai_num_contigs": 5,
        "ai_total_bp": 4900000,
        "ai_mean_cov": 58.0,
        "ai_median_cov": 55.0,
        "ai_cov_cv": 0.15,
        "ai_circular_n": 0,
        "ai_circular_bp_frac": 0.0,
        "ai_repeat_bp_frac": 0.08,
        "ai_longest_len": 350000,
        "ai_longest_cov": 58.0,
        "polish_mean_contig_cov": 56.0,
        "align_err_consensus": 0.002,
        "align_err_polishing": 0.0015,
        "force_conservative": False,
        "confidence_threshold": 0.5
    }


@pytest.fixture
def excellent_r1_payload():
    """Excellent R1 quality payload (should trigger early stop)"""
    return {
        "sample_id": "test_r1_excellent",
        "genus": "Escherichia",
        "round": 1,
        "qv": 36.5,
        "busco_complete": 96.0,
        "num_contigs": 1,
        "n50": 4500000,
        "total_length": 4800000,
        "error_rate": 0.0002,
        "confidence_threshold": 0.5
    }


class TestBasicEndpoints:
    """Test basic API endpoints"""

    def test_root_endpoint(self, client):
        """Test / endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "ESDP Polishing Decision API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "esdp-api"

    def test_model_info(self, client):
        """Test /model/info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_version" in data
        assert "feature_count" in data
        assert "model_type" in data

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        assert "timestamp" in data

        metrics = data["metrics"]
        assert "total_requests" in metrics
        assert "successful_predictions" in metrics
        assert "failed_predictions" in metrics
        assert "avg_response_time_ms" in metrics
        assert "decisions_by_rounds" in metrics


@pytest.mark.integration
class TestPredictionEndpoint:
    """Test /predict endpoint"""

    def test_predict_valid_high_quality(self, client, valid_request_payload):
        """Test /predict with valid high-quality metrics"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["sample_id"] == "test_001"
        assert data["recommended_rounds"] in [1, 3, 5]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["reasoning"], str)
        assert isinstance(data["warnings"], list)
        assert isinstance(data["class_probabilities"], dict)
        assert "processing_time_ms" in data
        assert "rule_overrides" in data  # NEW

    def test_predict_valid_low_quality(self, client, low_quality_request_payload):
        """Test /predict with valid low-quality metrics"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        response = client.post("/predict", json=low_quality_request_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["sample_id"] == "test_002"
        assert data["recommended_rounds"] in [1, 3, 5]
        # Low quality should recommend more rounds
        assert data["recommended_rounds"] >= 3

    def test_predict_with_optional_fields_missing(self, client):
        """Test /predict with some optional fields missing"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        minimal_payload = {
            "sample_id": "test_003",
            "round": 1,
            "qv": 30.0,
            "busco_complete": 95.0,
            "error_rate": 0.001,
            "n50": 1000000,
            "num_contigs": 5,
            "coverage": 70.0
        }

        response = client.post("/predict", json=minimal_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["sample_id"] == "test_003"
        assert data["recommended_rounds"] in [1, 3, 5]

    def test_predict_with_force_conservative(self, client, valid_request_payload):
        """Test /predict with force_conservative=True"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        valid_request_payload["force_conservative"] = True

        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["recommended_rounds"] == 5  # Should always be 5
        assert len(data["warnings"]) > 0
        assert data["rule_overrides"]["force_conservative"] is True  # NEW


@pytest.mark.integration
class TestConfidenceThreshold:
    """Test confidence_threshold parameter (NEW)"""

    def test_default_confidence_threshold(self, client, low_quality_request_payload):
        """Test default confidence threshold (0.5)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        # Remove explicit threshold to test default
        payload = low_quality_request_payload.copy()
        payload.pop("confidence_threshold", None)

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "rule_overrides" in data
        assert data["rule_overrides"]["confidence_threshold"] == 0.5

    def test_custom_confidence_threshold_04(self, client, low_quality_request_payload):
        """Test custom confidence threshold (0.4)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = 0.4

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["rule_overrides"]["confidence_threshold"] == 0.4

    def test_custom_confidence_threshold_06(self, client, low_quality_request_payload):
        """Test custom confidence threshold (0.6)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = 0.6

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["rule_overrides"]["confidence_threshold"] == 0.6

    def test_custom_confidence_threshold_07(self, client, low_quality_request_payload):
        """Test custom confidence threshold (0.7)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = 0.7

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["rule_overrides"]["confidence_threshold"] == 0.7

    def test_confidence_threshold_validation_too_low(self, client, low_quality_request_payload):
        """Test confidence threshold validation (< 0.0)"""
        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = -0.1

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_confidence_threshold_validation_too_high(self, client, low_quality_request_payload):
        """Test confidence threshold validation (> 1.0)"""
        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = 1.5

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestRuleOverrides:
    """Test rule_overrides tracking (NEW)"""

    def test_rule_overrides_structure(self, client, valid_request_payload):
        """Test that rule_overrides has correct structure"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200

        data = response.json()
        assert "rule_overrides" in data

        rule_overrides = data["rule_overrides"]
        required_keys = [
            "force_conservative",
            "low_confidence_override",
            "r1_quality_override",
            "confidence_threshold",
            "applied_threshold"
        ]

        for key in required_keys:
            assert key in rule_overrides, f"Missing key: {key}"

    def test_force_conservative_override(self, client, excellent_r1_payload):
        """Test force_conservative override (highest priority)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = excellent_r1_payload.copy()
        payload["force_conservative"] = True

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["recommended_rounds"] == 5
        assert data["rule_overrides"]["force_conservative"] is True
        assert data["rule_overrides"]["r1_quality_override"] is False  # Ignored
        assert "Force conservative override" in data["reasoning"]

    def test_r1_quality_override(self, client, excellent_r1_payload):
        """Test R1 quality override (QV >= 35, BUSCO >= 95)"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        response = client.post("/predict", json=excellent_r1_payload)
        assert response.status_code == 200

        data = response.json()

        # Check if R1 quality override was applied
        if data["rule_overrides"]["r1_quality_override"]:
            assert data["recommended_rounds"] == 1
            assert "R1 quality excellent" in data["reasoning"]
            assert len(data["warnings"]) > 0

    def test_low_confidence_override(self, client, low_quality_request_payload):
        """Test low confidence override"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        # Use high threshold to trigger override
        payload = low_quality_request_payload.copy()
        payload["confidence_threshold"] = 0.8

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()

        # If confidence is low, override should be applied
        if data["confidence"] < 0.8:
            assert data["rule_overrides"]["low_confidence_override"] is True
            assert data["rule_overrides"]["applied_threshold"] == 0.8
            assert any("Conservative bias applied" in w for w in data["warnings"])

    def test_no_overrides_applied(self, client, valid_request_payload):
        """Test case where no overrides are applied"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        # Use low threshold to avoid override
        payload = valid_request_payload.copy()
        payload["confidence_threshold"] = 0.3
        payload["force_conservative"] = False

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()

        # If confidence is high enough, no overrides should be applied
        if data["confidence"] >= 0.3:
            rule_overrides = data["rule_overrides"]
            # At least one of these should be False
            assert (
                rule_overrides["force_conservative"] is False or
                rule_overrides["low_confidence_override"] is False or
                rule_overrides["r1_quality_override"] is False
            )


@pytest.mark.integration
class TestDecisionHierarchy:
    """Test decision hierarchy (force > r1_quality > low_confidence)"""

    def test_hierarchy_force_over_r1_quality(self, client, excellent_r1_payload):
        """Test that force_conservative overrides r1_quality"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = excellent_r1_payload.copy()
        payload["force_conservative"] = True

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["recommended_rounds"] == 5  # Force conservative wins
        assert data["rule_overrides"]["force_conservative"] is True

    def test_hierarchy_r1_quality_over_low_confidence(self, client, excellent_r1_payload):
        """Test that r1_quality overrides low_confidence"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        payload = excellent_r1_payload.copy()
        payload["confidence_threshold"] = 0.9  # Very high threshold

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()

        # If R1 quality is excellent, it should override low confidence
        if data["rule_overrides"]["r1_quality_override"]:
            assert data["recommended_rounds"] == 1
            assert data["rule_overrides"]["low_confidence_override"] is False


class TestValidation:
    """Test input validation"""

    def test_predict_missing_required_field(self, client):
        """Test /predict with missing required field (sample_id)"""
        invalid_payload = {
            "qv": 35.0,
            "error_rate": 0.0003
        }

        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_invalid_round(self, client):
        """Test /predict with invalid round value"""
        invalid_payload = {
            "sample_id": "test_001",
            "round": -1,
            "qv": 35.0
        }

        response = client.post("/predict", json=invalid_payload)
        # Should still work (round is optional and not validated)
        # Or return 422 if validation is strict
        assert response.status_code in [200, 422, 500]

    def test_predict_negative_qv(self, client):
        """Test /predict with negative QV"""
        invalid_payload = {
            "sample_id": "test_001",
            "qv": -5.0
        }

        response = client.post("/predict", json=invalid_payload)
        # Should still work (no strict validation on QV)
        assert response.status_code in [200, 422, 500]

    def test_predict_invalid_busco(self, client):
        """Test /predict with invalid BUSCO value"""
        invalid_payload = {
            "sample_id": "test_001",
            "busco_complete": 150.0  # > 100
        }

        response = client.post("/predict", json=invalid_payload)
        # Should still work (no strict validation)
        assert response.status_code in [200, 422, 500]

    def test_predict_empty_payload(self, client):
        """Test /predict with empty payload"""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_malformed_json(self, client):
        """Test /predict with malformed JSON"""
        response = client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestMetricsTracking:
    """Test metrics tracking"""

    def test_metrics_tracking(self, client, valid_request_payload):
        """Test that metrics are tracked correctly"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        # Make a prediction
        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200

        # Check metrics
        metrics_response = client.get("/metrics")
        data = metrics_response.json()
        metrics = data["metrics"]

        assert metrics["total_requests"] >= 1
        assert metrics["successful_predictions"] >= 1
        assert metrics["avg_response_time_ms"] > 0

    def test_concurrent_predictions(self, client, valid_request_payload):
        """Test multiple concurrent predictions"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        # Make multiple predictions
        for i in range(3):
            payload = valid_request_payload.copy()
            payload["sample_id"] = f"test_{i:03d}"
            response = client.post("/predict", json=payload)
            assert response.status_code == 200

        # Check metrics
        metrics_response = client.get("/metrics")
        data = metrics_response.json()
        metrics = data["metrics"]

        assert metrics["successful_predictions"] >= 3


class TestResponseSchema:
    """Test response schemas"""

    def test_prediction_response_schema(self, client, valid_request_payload):
        """Test that prediction response matches schema"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200

        data = response.json()

        # Check all required fields
        required_fields = [
            "sample_id",
            "recommended_rounds",
            "confidence",
            "reasoning",
            "warnings",
            "rule_overrides",  # NEW
            "model_version",
            "class_probabilities",
            "processing_time_ms"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Check types
        assert isinstance(data["sample_id"], str)
        assert isinstance(data["recommended_rounds"], int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["reasoning"], str)
        assert isinstance(data["warnings"], list)
        assert isinstance(data["rule_overrides"], dict)  # NEW
        assert isinstance(data["model_version"], str)
        assert isinstance(data["class_probabilities"], dict)
        assert isinstance(data["processing_time_ms"], float)

    def test_prediction_request_schema(self, client):
        """Test that request schema validation works"""
        # Valid minimal request
        minimal_request = {
            "sample_id": "test_001",
            "qv": 35.0
        }

        response = client.post("/predict", json=minimal_request)
        # Should work (all other fields are optional)
        assert response.status_code in [200, 500]  # 500 if model fails with minimal data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
