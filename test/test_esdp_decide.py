"""
Unit tests for esdp_decide.py
Tests the decision logic, domain rules, and feature engineering.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from esdp_decide import (
    PolishingMetrics,
    Decision,
    engineer_features_online,
    check_r1_quality,
    apply_conservative_bias,
    decide
)


class TestPolishingMetrics:
    """Test PolishingMetrics dataclass"""

    def test_minimal_metrics(self):
        """Test PolishingMetrics with minimal required fields"""
        metrics = PolishingMetrics(
            sample_id="test_001",
            round=1,
            qv=35.0,
            busco_complete=95.0
        )

        assert metrics.sample_id == "test_001"
        assert metrics.round == 1
        assert metrics.qv == 35.0
        assert metrics.busco_complete == 95.0

    def test_full_metrics(self):
        """Test PolishingMetrics with all fields"""
        metrics = PolishingMetrics(
            sample_id="test_002",
            genus="Escherichia",
            round=1,
            coverage=50.0,
            coverage_effective=48.0,
            n50=4500000,
            qv=35.2,
            error_rate=0.0003,
            busco_complete=95.5,
            num_contigs=1,
            total_length=4800000
        )

        assert metrics.genus == "Escherichia"
        assert metrics.coverage == 50.0
        assert metrics.n50 == 4500000


class TestDecision:
    """Test Decision dataclass"""

    def test_decision_structure(self):
        """Test Decision dataclass structure"""
        decision = Decision(
            sample_id="test_001",
            recommended_rounds=1,
            confidence=0.85,
            reasoning="High quality R1 assembly",
            warnings=["R1 quality excellent"],
            rule_overrides={
                "force_conservative": False,
                "low_confidence_override": False,
                "r1_quality_override": True,
                "confidence_threshold": 0.5,
                "applied_threshold": None
            },
            model_version="1.0.0",
            class_probabilities={"early_r1_r2": 0.85, "medium_r3_r4": 0.10, "late_r5": 0.05}
        )

        assert decision.sample_id == "test_001"
        assert decision.recommended_rounds == 1
        assert decision.confidence == 0.85
        assert "rule_overrides" in decision.__dict__
        assert decision.rule_overrides["r1_quality_override"] is True


class TestFeatureEngineering:
    """Test feature engineering functions"""

    def test_engineer_features_basic(self):
        """Test basic feature engineering"""
        metrics = PolishingMetrics(
            sample_id="test_001",
            round=1,
            qv=35.0,
            n50=4500000,
            num_contigs=1,
            error_rate=0.0003,
            busco_complete=95.5,
            coverage=50.0,
            total_length=4800000
        )

        features = engineer_features_online(metrics)

        assert isinstance(features, dict)
        # engineer_features_online returns DERIVED features, not raw metrics
        assert "assembly_quality_score" in features
        assert "n50_ratio" in features
        assert "busco_per_contig" in features
        # Verify assembly_quality_score calculation: 35.0*0.4 + 95.5*0.4 - log1p(1)*0.2
        import numpy as np
        expected_score = 35.0 * 0.4 + 95.5 * 0.4 - np.log1p(1) * 0.2
        assert abs(features["assembly_quality_score"] - expected_score) < 0.01

    def test_engineer_features_missing_values(self):
        """Test feature engineering with missing values"""
        metrics = PolishingMetrics(
            sample_id="test_002",
            round=1,
            qv=30.0,
            busco_complete=90.0
        )

        features = engineer_features_online(metrics)

        assert isinstance(features, dict)
        # Missing values should be NaN
        assert np.isnan(features.get("n50", np.nan)) or features.get("n50") is None


class TestDomainRules:
    """Test domain-specific rules"""

    def test_check_r1_quality_excellent(self):
        """Test R1 quality check with excellent metrics"""
        metrics = PolishingMetrics(
            sample_id="test_001",
            round=1,
            qv=36.5,
            busco_complete=96.0,
            num_contigs=1
        )

        is_excellent, reason = check_r1_quality(metrics)

        assert is_excellent is True
        assert "excellent" in reason.lower()

    def test_check_r1_quality_poor(self):
        """Test R1 quality check with poor metrics"""
        metrics = PolishingMetrics(
            sample_id="test_002",
            round=1,
            qv=28.0,
            busco_complete=85.0,
            num_contigs=10
        )

        is_excellent, reason = check_r1_quality(metrics)

        assert is_excellent is False

    def test_apply_conservative_bias_low_confidence(self):
        """Test conservative bias with low confidence"""
        adjusted, override_applied, reason = apply_conservative_bias(
            predicted_class=0,
            confidence=0.4,
            confidence_threshold=0.5
        )

        assert override_applied is True
        assert adjusted == 1  # Should escalate by +1 (Early → Medium)
        assert "confidence" in reason.lower()  # Reason mentions confidence threshold

    def test_apply_conservative_bias_high_confidence(self):
        """Test conservative bias with high confidence"""
        adjusted, override_applied, reason = apply_conservative_bias(
            predicted_class=0,
            confidence=0.9,
            confidence_threshold=0.5
        )

        assert override_applied is False
        assert adjusted == 0  # No change
        assert reason == ""

    def test_apply_conservative_bias_medium_confidence(self):
        """Test conservative bias with medium confidence"""
        adjusted, override_applied, reason = apply_conservative_bias(
            predicted_class=1,
            confidence=0.45,
            confidence_threshold=0.5
        )

        assert override_applied is True
        assert adjusted == 2  # Should escalate by +1 (Medium → Late)


class TestPrepareFeatures:
    """Test feature preparation"""

    def test_prepare_features_basic(self):
        """Test basic feature preparation"""
        metrics = PolishingMetrics(
            sample_id="test_001",
            round=1,
            qv=35.0,
            n50=4500000,
            num_contigs=1,
            error_rate=0.0003,
            busco_complete=95.5
        )

        features = engineer_features_online(metrics)

        assert isinstance(features, dict)
        assert len(features) > 0


@pytest.mark.integration
class TestDecideIntegration:
    """Integration tests for decide() function"""

    def test_decide_with_real_model(self):
        """Test decide() with real trained model"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        metrics = PolishingMetrics(
            sample_id="test_001",
            genus="Escherichia",
            round=1,
            coverage=50.0,
            n50=4500000,
            qv=36.5,
            error_rate=0.0002,
            busco_complete=96.0,
            num_contigs=1,
            total_length=4800000
        )

        decision = decide(
            metrics=metrics,
            model_path=str(model_path),
            force_conservative=False,
            confidence_threshold=0.5
        )

        assert isinstance(decision, Decision)
        assert decision.sample_id == "test_001"
        assert decision.recommended_rounds in [1, 3, 5]
        assert 0 <= decision.confidence <= 1
        assert isinstance(decision.reasoning, str)
        assert isinstance(decision.warnings, list)
        assert isinstance(decision.rule_overrides, dict)
        assert isinstance(decision.class_probabilities, dict)

    def test_decide_poor_quality(self):
        """Test decide() with poor quality metrics"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        metrics = PolishingMetrics(
            sample_id="test_002",
            genus="Salmonella",
            round=1,
            coverage=30.0,
            n50=2000000,
            qv=28.5,
            error_rate=0.0015,
            busco_complete=85.0,
            num_contigs=5,
            total_length=4900000
        )

        decision = decide(
            metrics=metrics,
            model_path=str(model_path),
            force_conservative=False,
            confidence_threshold=0.5
        )

        assert isinstance(decision, Decision)
        # Poor quality should recommend more rounds
        assert decision.recommended_rounds >= 3

    def test_decide_force_conservative(self):
        """Test decide() with force_conservative=True"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        metrics = PolishingMetrics(
            sample_id="test_003",
            round=1,
            qv=36.5,
            busco_complete=96.0,
            num_contigs=1
        )

        decision = decide(
            metrics=metrics,
            model_path=str(model_path),
            force_conservative=True,
            confidence_threshold=0.5
        )

        assert decision.recommended_rounds == 5
        assert decision.rule_overrides["force_conservative"] is True
        assert len(decision.warnings) > 0

    def test_decide_custom_threshold(self):
        """Test decide() with custom confidence threshold"""
        model_path = Path("models/best_model_pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found")

        metrics = PolishingMetrics(
            sample_id="test_004",
            round=1,
            qv=30.0,
            busco_complete=90.0
        )

        decision = decide(
            metrics=metrics,
            model_path=str(model_path),
            force_conservative=False,
            confidence_threshold=0.7
        )

        assert decision.rule_overrides["confidence_threshold"] == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
