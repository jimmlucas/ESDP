"""
Integration tests for the full ESDP pipeline
Tests end-to-end workflows and model loading.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelPipeline:
    """Test the trained model pipeline"""

    @pytest.fixture
    def model_path(self):
        """Get the model path"""
        return Path("models/best_model_pipeline.pkl")

    @pytest.fixture
    def model_pipeline(self, model_path):
        """Load the model pipeline"""
        if not model_path.exists():
            pytest.skip("Model not found. Run training first.")

        return joblib.load(model_path)

    def test_model_exists(self, model_path):
        """Test that the model file exists"""
        assert model_path.exists(), "Model not found. Run: bash run_pipeline.sh"

    def test_model_is_pipeline(self, model_pipeline):
        """Test that the model is a sklearn Pipeline"""
        from sklearn.pipeline import Pipeline
        assert isinstance(model_pipeline, Pipeline)

    def test_pipeline_has_steps(self, model_pipeline):
        """Test that pipeline has expected steps"""
        step_names = [name for name, _ in model_pipeline.steps]

        # Should have imputer, scaler, and model
        assert 'imputer' in step_names
        assert 'scaler' in step_names
        assert 'model' in step_names

    def test_pipeline_has_metadata(self, model_pipeline):
        """Test that pipeline has metadata"""
        assert hasattr(model_pipeline, 'feature_names')
        assert hasattr(model_pipeline, 'model_version')

        # Check metadata content
        assert isinstance(model_pipeline.feature_names, list)
        assert len(model_pipeline.feature_names) > 0

    def test_pipeline_predict_with_nans(self, model_pipeline):
        """Test that pipeline handles NaN values"""
        # Create a sample with some NaN values
        feature_names = model_pipeline.feature_names

        # Create DataFrame with NaNs
        sample_data = {name: [np.nan] for name in feature_names}
        # Fill some with actual values
        if 'qv' in sample_data:
            sample_data['qv'] = [35.5]
        if 'error_rate' in sample_data:
            sample_data['error_rate'] = [0.00028]

        X = pd.DataFrame(sample_data)

        # Should not raise an error
        prediction = model_pipeline.predict(X)
        assert prediction is not None
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]

    def test_pipeline_predict_proba(self, model_pipeline):
        """Test that pipeline can predict probabilities"""
        feature_names = model_pipeline.feature_names

        # Create a valid sample
        sample_data = {name: [0.0] for name in feature_names}
        X = pd.DataFrame(sample_data)

        # Should return probabilities
        proba = model_pipeline.predict_proba(X)
        assert proba is not None
        assert proba.shape == (1, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestFeatureNames:
    """Test feature names consistency"""

    @pytest.fixture
    def feature_names_file(self):
        """Get feature names file path"""
        return Path("models/feature_names.txt")

    @pytest.fixture
    def model_path(self):
        """Get model path"""
        return Path("models/best_model_pipeline.pkl")

    def test_feature_names_file_exists(self, feature_names_file):
        """Test that feature names file exists"""
        if not feature_names_file.exists():
            pytest.skip("Feature names file not found")
        assert feature_names_file.exists()

    def test_feature_names_match_model(self, feature_names_file, model_path):
        """Test that feature names in file match model"""
        if not feature_names_file.exists() or not model_path.exists():
            pytest.skip("Files not found")

        # Load feature names from file
        with open(feature_names_file, 'r') as f:
            file_features = [line.strip() for line in f]

        # Load feature names from model
        pipeline = joblib.load(model_path)

        model_features = pipeline.feature_names

        # Should match
        assert file_features == model_features


class TestTrainingDataset:
    """Test the training dataset"""

    @pytest.fixture
    def training_data_path(self):
        """Get training data path"""
        return Path("data/training_dataset_engineered.csv")

    def test_training_data_exists(self, training_data_path):
        """Test that training data exists"""
        assert training_data_path.exists()

    def test_training_data_has_nan_values(self, training_data_path):
        """Test that training data has NaN values (not 0.0 for missing)"""
        df = pd.read_csv(training_data_path)

        # Check that we have actual NaN values
        assert df.isna().sum().sum() > 0, "Expected some NaN values in training data"


class TestEndToEndPrediction:
    """Test end-to-end prediction workflow"""

    @pytest.fixture
    def model_path(self):
        """Get model path"""
        return Path("models/best_model_pipeline.pkl")

    def test_full_prediction_workflow(self, model_path):
        """Test complete prediction workflow"""
        if not model_path.exists():
            pytest.skip("Model not found")

        # Import the decide function
        from esdp_decide import decide, PolishingMetrics

        # Create test metrics
        metrics = PolishingMetrics(
            sample_id="test_001",
            qv=35.5,
            error_rate=0.00028,
            busco_complete=95.2,
            n50=4500000,
            num_contigs=12,
            coverage=50.0
        )

        # Make prediction
        decision = decide(metrics, str(model_path))

        # Validate decision
        assert decision.recommended_rounds in [1, 3, 5]
        assert 0 <= decision.confidence <= 1
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0

    def test_prediction_with_minimal_features(self, model_path):
        """Test prediction with only required features"""
        if not model_path.exists():
            pytest.skip("Model not found")

        from esdp_decide import decide, PolishingMetrics

        # Only required fields
        metrics = PolishingMetrics(
            sample_id="test_002",
            qv=32.0,
            error_rate=0.0005,
            busco_complete=92.0,
            n50=3000000,
            num_contigs=15
        )

        decision = decide(metrics, str(model_path))

        # Should still work
        assert decision.recommended_rounds in [1, 3, 5]
        assert decision.confidence > 0


class TestModelPerformance:
    """Test model performance metrics"""

    @pytest.fixture
    def model_comparison_path(self):
        """Get model comparison results"""
        return Path("outputs/model_comparison.csv")

    def test_model_comparison_exists(self, model_comparison_path):
        """Test that model comparison results exist"""
        if not model_comparison_path.exists():
            pytest.skip("Model comparison not found")
        assert model_comparison_path.exists()

    def test_model_meets_minimum_accuracy(self, model_comparison_path):
        """Test that model meets minimum accuracy threshold"""
        if not model_comparison_path.exists():
            pytest.skip("Model comparison not found")

        df = pd.read_csv(model_comparison_path)

        # Get best model accuracy
        best_accuracy = df['accuracy'].max()

        # Should be better than random (33.3% for 3 classes)
        assert best_accuracy > 0.40, f"Model accuracy {best_accuracy} is too low"

    def test_model_better_than_baseline(self, model_comparison_path):
        """Test that model is better than baseline"""
        baseline_path = Path("outputs/baseline_comparison.csv")

        if not model_comparison_path.exists() or not baseline_path.exists():
            pytest.skip("Comparison files not found")

        model_df = pd.read_csv(model_comparison_path)
        baseline_df = pd.read_csv(baseline_path)

        # Get best model accuracy
        best_model_acc = model_df['accuracy'].max()

        # Get baseline accuracy (excluding "Best_Model" row)
        baseline_acc = baseline_df[
            baseline_df['model'] != 'Best_Model'
        ]['accuracy'].max()

        # Model should be better than baseline
        assert best_model_acc > baseline_acc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
