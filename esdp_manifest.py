"""Verified model manifests and cached artifact loading for ESDP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


MANIFEST_VERSION = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class ManifestError(RuntimeError):
    """Base class for manifest and artifact failures."""


class ArtifactIntegrityError(ManifestError):
    """Raised when an artifact does not match its recorded checksum."""


class ModelCompatibilityError(ManifestError):
    """Raised when an artifact and its declared contract disagree."""


class StrictManifestModel(BaseModel):
    """Immutable and non-extensible manifest model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FileReference(StrictManifestModel):
    """Repository-relative file identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("manifest file paths must be relative")
        return value


class ModelArtifact(FileReference):
    """Serialized inference artifact."""

    format: Literal["joblib"]
    python_type: str = Field(min_length=1)


class FeatureSchema(StrictManifestModel):
    """Exact ordered feature contract expected by the model."""

    version: str = Field(min_length=1)
    prospective: bool
    feature_names_file: FileReference
    names: tuple[str, ...] = Field(min_length=1)
    notes: str = Field(min_length=1)

    @field_validator("names", mode="before")
    @classmethod
    def parse_json_names(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def require_unique_feature_names(self):
        if len(self.names) != len(set(self.names)):
            raise ValueError("feature names must be unique")
        if any(not name.strip() for name in self.names):
            raise ValueError("feature names cannot be blank")
        return self


class TrainingData(StrictManifestModel):
    """Training dataset identity and split semantics."""

    dataset: FileReference
    split_group_level: str = Field(min_length=1)
    label_unit: str = Field(min_length=1)


class SourceIdentity(StrictManifestModel):
    """Source repository state associated with the artifact."""

    repository: str = Field(min_length=1)
    commit: str = Field(pattern=COMMIT_PATTERN)


class PipelineStep(StrictManifestModel):
    """Expected pipeline step and implementation type."""

    name: str = Field(min_length=1)
    python_type: str = Field(min_length=1)


class Compatibility(StrictManifestModel):
    """Scientifically supported operating domain."""

    sequencing_platforms: tuple[str, ...] = Field(min_length=1)
    organism_scope: tuple[str, ...] = Field(min_length=1)
    assemblers: tuple[str, ...] = Field(min_length=1)
    polishers: tuple[str, ...] = Field(min_length=1)
    supported_rounds: tuple[int, ...] = Field(min_length=1)
    training_software: dict[str, str] = Field(min_length=1)

    @field_validator(
        "sequencing_platforms",
        "organism_scope",
        "assemblers",
        "polishers",
        "supported_rounds",
        mode="before",
    )
    @classmethod
    def parse_json_arrays(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_supported_rounds(self):
        if tuple(sorted(set(self.supported_rounds))) != self.supported_rounds:
            raise ValueError("supported_rounds must be sorted and unique")
        outside_supported_range = any(
            round_number < 1 or round_number > 5
            for round_number in self.supported_rounds
        )
        if outside_supported_range:
            raise ValueError("supported_rounds must be within R1-R5")
        return self


class ModelManifest(StrictManifestModel):
    """Complete identity and compatibility contract for an ESDP model."""

    manifest_version: Literal["1.0.0"] = MANIFEST_VERSION
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    artifact: ModelArtifact
    feature_schema: FeatureSchema
    training_data: TrainingData
    source: SourceIdentity
    pipeline_steps: tuple[PipelineStep, ...] = Field(min_length=1)
    compatibility: Compatibility

    @field_validator("pipeline_steps", mode="before")
    @classmethod
    def parse_json_pipeline_steps(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def require_unique_step_names(self):
        names = [step.name for step in self.pipeline_steps]
        if len(names) != len(set(names)):
            raise ValueError("pipeline step names must be unique")
        return self


@dataclass(frozen=True)
class VerifiedModel:
    """Loaded model paired with the manifest that verified it."""

    model: Any
    manifest: ModelManifest
    manifest_path: Path
    artifact_path: Path


def sha256_file(path: str | Path) -> str:
    """Calculate a file checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> ModelManifest:
    """Read and strictly validate a JSON model manifest."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
        return ModelManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ManifestError(
            f"invalid model manifest {manifest_path}: {error}"
        ) from error


def _referenced_path(manifest_path: Path, reference: FileReference) -> Path:
    return (manifest_path.parent / reference.path).resolve()


def _verify_checksum(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ArtifactIntegrityError(f"{label} file not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"{label} checksum mismatch: expected {expected}, got {actual}"
        )


def verify_manifest_files(
    manifest: ModelManifest,
    manifest_path: str | Path,
    include_training_data: bool = False,
) -> dict[str, Path]:
    """Verify checksums and return resolved referenced paths."""
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    artifact_path = _referenced_path(resolved_manifest, manifest.artifact)
    feature_names_path = _referenced_path(
        resolved_manifest,
        manifest.feature_schema.feature_names_file,
    )
    training_data_path = _referenced_path(
        resolved_manifest,
        manifest.training_data.dataset,
    )

    _verify_checksum(
        artifact_path,
        manifest.artifact.sha256,
        "model artifact",
    )
    _verify_checksum(
        feature_names_path,
        manifest.feature_schema.feature_names_file.sha256,
        "feature names",
    )
    if include_training_data:
        _verify_checksum(
            training_data_path,
            manifest.training_data.dataset.sha256,
            "training data",
        )

    file_names = tuple(
        line.strip()
        for line in feature_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if file_names != manifest.feature_schema.names:
        raise ModelCompatibilityError(
            "feature names file does not match the manifest feature order"
        )

    return {
        "artifact": artifact_path,
        "feature_names": feature_names_path,
        "training_data": training_data_path,
    }


def _python_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _verify_loaded_model(model: Any, manifest: ModelManifest) -> None:
    actual_type = _python_type(model)
    if actual_type != manifest.artifact.python_type:
        raise ModelCompatibilityError(
            f"model type mismatch: expected {manifest.artifact.python_type}, "
            f"got {actual_type}"
        )

    actual_version = getattr(model, "model_version", None)
    if actual_version != manifest.model_version:
        raise ModelCompatibilityError(
            f"model version mismatch: expected {manifest.model_version}, "
            f"got {actual_version!r}"
        )

    actual_features = tuple(getattr(model, "feature_names", ()))
    if actual_features != manifest.feature_schema.names:
        raise ModelCompatibilityError(
            "serialized model feature order does not match the manifest"
        )

    actual_steps = getattr(model, "named_steps", None)
    if actual_steps is None:
        raise ModelCompatibilityError("serialized model has no named_steps")

    expected_steps = tuple(
        (step.name, step.python_type)
        for step in manifest.pipeline_steps
    )
    observed_steps = tuple(
        (name, _python_type(step))
        for name, step in actual_steps.items()
    )
    if observed_steps != expected_steps:
        raise ModelCompatibilityError(
            f"pipeline steps mismatch: expected {expected_steps}, "
            f"got {observed_steps}"
        )

    if getattr(model, "split_group_level", None) != (
        manifest.training_data.split_group_level
    ):
        raise ModelCompatibilityError("split_group_level mismatch")
    if getattr(model, "label_unit", None) != manifest.training_data.label_unit:
        raise ModelCompatibilityError("label_unit mismatch")


@lru_cache(maxsize=8)
def _load_verified_model_cached(
    manifest_path_text: str,
    manifest_checksum: str,
) -> VerifiedModel:
    del manifest_checksum  # The cache key invalidates when the manifest changes.
    manifest_path = Path(manifest_path_text)
    manifest = read_manifest(manifest_path)
    paths = verify_manifest_files(manifest, manifest_path)
    model = joblib.load(paths["artifact"])
    _verify_loaded_model(model, manifest)
    return VerifiedModel(
        model=model,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_path=paths["artifact"],
    )


def load_verified_model(
    manifest_path: str | Path,
    required_feature_schema_version: str | None = None,
) -> VerifiedModel:
    """Load a checksum-verified compatible model, cached per manifest state."""
    resolved = Path(manifest_path).expanduser().resolve()
    if not resolved.is_file():
        raise ManifestError(f"model manifest not found: {resolved}")
    try:
        manifest_checksum = sha256_file(resolved)
    except OSError as error:
        raise ManifestError(
            f"unable to read model manifest {resolved}: {error}"
        ) from error
    verified = _load_verified_model_cached(str(resolved), manifest_checksum)

    if (
        required_feature_schema_version is not None
        and verified.manifest.feature_schema.version
        != required_feature_schema_version
    ):
        raise ModelCompatibilityError(
            "feature schema mismatch: required "
            f"{required_feature_schema_version}, manifest provides "
            f"{verified.manifest.feature_schema.version}"
        )
    return verified


def clear_model_cache() -> None:
    """Clear the process-local verified model cache (primarily for tests)."""
    _load_verified_model_cached.cache_clear()
