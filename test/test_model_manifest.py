"""Tests for verified ESDP model manifests and cached loading."""

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from pydantic import ValidationError

import esdp_manifest
from esdp_decide import PolishingMetrics, decide
from esdp_manifest import (
    ArtifactIntegrityError,
    ManifestError,
    ModelCompatibilityError,
    ModelManifest,
    clear_model_cache,
    load_verified_model,
    read_manifest,
    sha256_file,
    verify_manifest_files,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY / "models" / "model_manifest.v1.json"


def _copy_manifest_bundle(tmp_path: Path) -> tuple[Path, dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)

    model_path = tmp_path / "model.pkl"
    feature_names_path = tmp_path / "feature_names.txt"
    training_data_path = tmp_path / "training.csv"
    shutil.copyfile(REPOSITORY / "models" / "best_model_pipeline.pkl", model_path)
    shutil.copyfile(REPOSITORY / "models" / "feature_names.txt", feature_names_path)
    shutil.copyfile(
        REPOSITORY / "data" / "training_dataset_with_target.csv",
        training_data_path,
    )

    payload["artifact"]["path"] = model_path.name
    payload["feature_schema"]["feature_names_file"]["path"] = (
        feature_names_path.name
    )
    payload["training_data"]["dataset"]["path"] = training_data_path.name

    copied_manifest = tmp_path / "manifest.json"
    copied_manifest.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return copied_manifest, payload


def _load_batch_inference_module():
    module_path = REPOSITORY / "7_inference_pipeline.py"
    spec = spec_from_file_location("esdp_batch_inference", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def reset_verified_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def test_repository_manifest_verifies_all_recorded_files():
    manifest = read_manifest(MANIFEST_PATH)
    paths = verify_manifest_files(
        manifest,
        MANIFEST_PATH,
        include_training_data=True,
    )

    assert manifest.model_version == "v1.1.0"
    assert manifest.feature_schema.version == "1.0.0"
    assert manifest.feature_schema.prospective is False
    assert len(manifest.feature_schema.names) == 42
    assert manifest.prediction_contract.classes == (0, 1, 2)
    assert manifest.prediction_contract.recommended_rounds == (1, 3, 5)
    assert paths["artifact"].name == "best_model_pipeline.pkl"


def test_verified_loader_checks_model_metadata_and_uses_cache(monkeypatch):
    original_load = esdp_manifest.joblib.load
    load_count = 0

    def counting_load(path):
        nonlocal load_count
        load_count += 1
        return original_load(path)

    monkeypatch.setattr(esdp_manifest.joblib, "load", counting_load)

    first = load_verified_model(MANIFEST_PATH)
    second = load_verified_model(MANIFEST_PATH)

    assert first.model is second.model
    assert load_count == 1


def test_corrupted_model_artifact_is_rejected_before_deserialization(tmp_path):
    manifest_path, _ = _copy_manifest_bundle(tmp_path)
    model_path = tmp_path / "model.pkl"
    with model_path.open("ab") as model_file:
        model_file.write(b"corruption")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_verified_model(manifest_path)


def test_changed_feature_file_is_rejected_even_with_updated_checksum(tmp_path):
    manifest_path, payload = _copy_manifest_bundle(tmp_path)
    feature_path = tmp_path / "feature_names.txt"
    feature_names = feature_path.read_text(encoding="utf-8").splitlines()
    feature_path.write_text(
        "\n".join(reversed(feature_names)) + "\n",
        encoding="utf-8",
    )
    payload["feature_schema"]["feature_names_file"]["sha256"] = sha256_file(
        feature_path
    )
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="feature order"):
        load_verified_model(manifest_path)


def test_manifest_model_version_must_match_serialized_model(tmp_path):
    manifest_path, payload = _copy_manifest_bundle(tmp_path)
    payload["model_version"] = "v9.9.9"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="model version mismatch"):
        load_verified_model(manifest_path)


def test_manifest_prediction_classes_must_match_serialized_model(tmp_path):
    manifest_path, payload = _copy_manifest_bundle(tmp_path)
    payload["prediction_contract"]["classes"] = [0, 2, 1]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="model classes"):
        load_verified_model(manifest_path)


def test_prediction_contract_requires_parallel_output_semantics():
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    payload["prediction_contract"]["labels"].pop()

    with pytest.raises(ValidationError, match="equal length"):
        ModelManifest.model_validate(payload)


def test_legacy_model_cannot_claim_prospective_v2_compatibility():
    with pytest.raises(ModelCompatibilityError, match="feature schema mismatch"):
        load_verified_model(
            MANIFEST_PATH,
            required_feature_schema_version="2.0.0",
        )


def test_manifest_rejects_unknown_fields():
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    payload["untracked_metadata"] = True

    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)


def test_missing_manifest_has_an_explicit_error(tmp_path):
    with pytest.raises(ManifestError, match="manifest not found"):
        load_verified_model(tmp_path / "missing.json")


def test_default_decision_reports_verified_feature_schema():
    decision = decide(
        PolishingMetrics(
            sample_id="manifest_test",
            round=1,
            coverage=40.0,
            qv=35.0,
            busco_complete=95.0,
            n50=4_000_000,
            num_contigs=2,
            error_rate=0.001,
            total_length=4_800_000,
        )
    )

    assert decision.model_version == "v1.1.0"
    assert decision.feature_schema_version == "1.0.0"


def test_default_decision_is_independent_of_working_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    decision = decide(
        PolishingMetrics(
            sample_id="cwd_independent",
            round=1,
            coverage=40.0,
            qv=35.0,
            busco_complete=95.0,
            n50=4_000_000,
            num_contigs=2,
            error_rate=0.001,
            total_length=4_800_000,
        )
    )

    assert decision.model_version == "v1.1.0"
    assert decision.feature_schema_version == "1.0.0"


def test_batch_predictor_uses_the_verified_manifest_by_default():
    inference = _load_batch_inference_module()

    predictor = inference.PolishingPredictor()
    prediction = predictor.predict(
        {
            "qv": 35.0,
            "busco_complete": 95.0,
            "n50": 4_000_000,
            "num_contigs": 2,
            "error_rate": 0.001,
            "total_length": 4_800_000,
        }
    )

    assert predictor.feature_schema_version == "1.0.0"
    assert prediction["model_version"] == "v1.1.0"
    assert prediction["feature_schema_version"] == "1.0.0"


def test_batch_predictor_defaults_are_independent_of_working_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    inference = _load_batch_inference_module()

    predictor = inference.PolishingPredictor()

    assert predictor.feature_schema_version == "1.0.0"


def test_docker_image_contains_the_verified_model_runtime():
    dockerfile = (REPOSITORY / "Dockerfile").read_text(encoding="utf-8")

    for required_source in (
        "esdp_features.py",
        "esdp_manifest.py",
        "models/feature_names.txt",
        "models/model_manifest.v1.json",
    ):
        assert f"COPY --chown=esdp:esdp {required_source} " in dockerfile
