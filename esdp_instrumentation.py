"""Immutable prospective instrumentation projects for ESDP development."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from esdp_light_history import LightFeatureHistory, build_light_feature_history
from esdp_light_metrics import (
    LightRoundObservation,
    collect_light_round_observation,
)
from esdp_manifest import sha256_file


INSTRUMENTATION_PROJECT_SCHEMA_VERSION = "1.0.0"
OFFLINE_QC_SCHEMA_VERSION = "1.0.0"
ROUND_RECORD_SCHEMA_VERSION = "1.0.0"
PROJECT_FILE_NAME = "esdp-project.json"
OFFLINE_QC_SCHEMA_FILE_NAME = "offline-qc.schema.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InstrumentationError(ValueError):
    """Raised when a prospective instrumentation transaction is invalid."""


class ToolIdentity(BaseModel):
    """Versioned identity for one workflow tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters: tuple[str, ...] = ()

    @field_validator("name", "version", mode="before")
    @classmethod
    def normalize_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, value):
        return tuple(value) if isinstance(value, list) else value


class LongReadTechnology(BaseModel):
    """Sequencing-domain metadata required for prospective calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    platform: Literal["ont", "pacbio_hifi", "pacbio_clr", "other"]
    chemistry: str = Field(min_length=1)
    basecaller: str | None = None
    basecaller_version: str | None = None
    basecaller_model: str | None = None

    @field_validator(
        "chemistry",
        "basecaller",
        "basecaller_version",
        "basecaller_model",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def require_ont_basecalling_contract(self):
        if self.platform == "ont":
            missing = [
                name
                for name in (
                    "basecaller",
                    "basecaller_version",
                    "basecaller_model",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"ONT projects require basecalling provenance: {missing}"
                )
        return self


class InstrumentationProject(BaseModel):
    """Frozen project-level contract; it can never enable model decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = INSTRUMENTATION_PROJECT_SCHEMA_VERSION
    project_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    mode: Literal["prospective_instrumentation"] = "prospective_instrumentation"
    decision_enabled: Literal[False] = False
    max_rounds: int = Field(default=5, ge=1, le=5)
    technology: LongReadTechnology
    assembler: ToolIdentity
    polisher: ToolIdentity
    online_metric_contract: tuple[str, ...] = (
        "n50",
        "num_contigs",
        "total_length",
        "gc",
        "mapping_error_rate_optional",
        "bases_mapped_cigar_optional",
        "mismatches_optional",
    )
    offline_outcome_contract: tuple[str, ...] = (
        "busco_marker_counts_with_provenance",
        "independent_kmer_qv_and_completeness",
        "substitution_insertion_deletion_counts",
        "homopolymer_indels",
        "frameshifts_and_premature_stops",
    )
    materiality_contract_status: Literal["blocked_pending_validation"] = (
        "blocked_pending_validation"
    )

    @field_validator(
        "online_metric_contract",
        "offline_outcome_contract",
        mode="before",
    )
    @classmethod
    def normalize_contract_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value


class BuscoOutcome(BaseModel):
    """Count-based BUSCO outcome with complete reproducibility metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    complete_count: int = Field(ge=0)
    single_copy_count: int = Field(ge=0)
    duplicated_count: int = Field(ge=0)
    fragmented_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    marker_count: int = Field(gt=0)
    version: str = Field(min_length=1)
    lineage_dataset: str = Field(min_length=1)
    lineage_creation_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    mode: str = Field(min_length=1)
    options: tuple[str, ...] = ()

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_marker_partition(self):
        if self.single_copy_count + self.duplicated_count != self.complete_count:
            raise ValueError(
                "BUSCO single-copy + duplicated counts must equal complete_count"
            )
        if (
            self.complete_count + self.fragmented_count + self.missing_count
            != self.marker_count
        ):
            raise ValueError(
                "BUSCO complete + fragmented + missing must equal marker_count"
            )
        return self


class KmerOutcome(BaseModel):
    """Independent-read k-mer consensus outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    qv: float = Field(ge=0)
    completeness_percent: float = Field(ge=0, le=100)
    tool: ToolIdentity
    read_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kmer_size: int = Field(gt=0)


class ConsensusErrorOutcome(BaseModel):
    """Separated consensus error counts from an independent evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evaluated_bases: int = Field(gt=0)
    substitutions: int = Field(ge=0)
    insertions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    homopolymer_indels: int | None = Field(default=None, ge=0)
    tool: ToolIdentity
    method: str = Field(min_length=1)


class CodingIntegrityOutcome(BaseModel):
    """Functional damage indicators sensitive to long-read indels."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    predicted_cds: int = Field(ge=0)
    frameshifts: int = Field(ge=0)
    premature_stops: int = Field(ge=0)
    truncated_cds: int = Field(ge=0)
    tool: ToolIdentity


class OfflineQualityOutcome(BaseModel):
    """Optional expensive outcomes recorded without entering light features."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = OFFLINE_QC_SCHEMA_VERSION
    assembly_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    busco: BuscoOutcome | None = None
    kmer: KmerOutcome | None = None
    consensus_errors: ConsensusErrorOutcome | None = None
    coding_integrity: CodingIntegrityOutcome | None = None

    @model_validator(mode="after")
    def require_at_least_one_outcome(self):
        if not any(
            (
                self.busco,
                self.kmer,
                self.consensus_errors,
                self.coding_integrity,
            )
        ):
            raise ValueError("offline QC must contain at least one outcome family")
        return self


class ReadArtifact(BaseModel):
    """Content identity for an input read artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    file_name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProspectiveRoundRecord(BaseModel):
    """Immutable commit marker for one successfully recorded round."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = ROUND_RECORD_SCHEMA_VERSION
    project_id: str
    project_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str
    coverage_effective: float = Field(gt=0)
    round: int = Field(ge=1)
    decision_enabled: Literal[False] = False
    observation_file: Literal["observation.json"] = "observation.json"
    history_file: Literal["history.json"] = "history.json"
    offline_qc_file: Literal["offline-qc.json"] | None = None
    read_artifacts: tuple[ReadArtifact, ...] = ()

    @field_validator("read_artifacts", mode="before")
    @classmethod
    def normalize_read_artifacts(cls, value):
        return tuple(value) if isinstance(value, list) else value


class ProjectInitReceipt(BaseModel):
    """Stable receipt returned by the Python API and CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project_id: str
    project_directory: str
    project_file: str
    project_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    offline_qc_schema_file: str
    offline_qc_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_enabled: Literal[False] = False


class RoundRecordReceipt(BaseModel):
    """Paths and identities committed by one record-round transaction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project_id: str
    project_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str
    coverage_effective: float
    round: int
    round_directory: str
    observation_file: str
    history_file: str
    record_file: str
    offline_qc_file: str | None
    decision_enabled: Literal[False] = False


def _json_bytes(payload: Any) -> bytes:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise InstrumentationError(f"unable to write {path}: {error}") from error


def _validate_safe_id(value: str, label: str) -> str:
    normalized = value.strip()
    if not SAFE_ID.fullmatch(normalized):
        raise InstrumentationError(
            f"{label} must match {SAFE_ID.pattern}: {value!r}"
        )
    return normalized


def _coverage_segment(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise InstrumentationError("coverage_effective must be finite and positive")
    return f"coverage-{format(value, '.10g').replace('.', 'p')}"


def init_project(
    project_directory: str | Path,
    *,
    project_id: str,
    technology: LongReadTechnology,
    assembler: ToolIdentity,
    polisher: ToolIdentity,
    max_rounds: int = 5,
) -> ProjectInitReceipt:
    """Create an empty prospective project with decisions permanently disabled."""
    root = Path(project_directory).expanduser().resolve()
    project_id = _validate_safe_id(project_id, "project_id")
    project = InstrumentationProject(
        project_id=project_id,
        technology=technology,
        assembler=assembler,
        polisher=polisher,
        max_rounds=max_rounds,
    )
    if root.exists() and any(root.iterdir()):
        raise InstrumentationError(
            f"project directory must be absent or empty: {root}"
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir()
        (root / "schemas").mkdir()
    except OSError as error:
        raise InstrumentationError(
            f"unable to initialize project directory {root}: {error}"
        ) from error

    project_path = root / PROJECT_FILE_NAME
    schema_path = root / "schemas" / OFFLINE_QC_SCHEMA_FILE_NAME
    _atomic_json_file(project_path, project)
    _atomic_json_file(schema_path, OfflineQualityOutcome.model_json_schema())
    return ProjectInitReceipt(
        project_id=project.project_id,
        project_directory=str(root),
        project_file=str(project_path),
        project_contract_sha256=sha256_file(project_path),
        offline_qc_schema_file=str(schema_path),
        offline_qc_schema_sha256=sha256_file(schema_path),
    )


def load_project(project_directory: str | Path) -> InstrumentationProject:
    root = Path(project_directory).expanduser().resolve()
    path = root / PROJECT_FILE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return InstrumentationProject.model_validate(payload)
    except OSError as error:
        raise InstrumentationError(f"unable to read project contract {path}: {error}")
    except (json.JSONDecodeError, ValueError) as error:
        raise InstrumentationError(f"invalid project contract {path}: {error}")


def load_offline_qc(path: str | Path) -> OfflineQualityOutcome:
    qc_path = Path(path).expanduser().resolve()
    try:
        return OfflineQualityOutcome.model_validate_json(
            qc_path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise InstrumentationError(f"unable to read offline QC {qc_path}: {error}")
    except ValueError as error:
        raise InstrumentationError(f"invalid offline QC {qc_path}: {error}")


def _read_observation(path: Path) -> LightRoundObservation:
    try:
        return LightRoundObservation.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise InstrumentationError(f"invalid prior observation {path}: {error}")


def _read_round_record(path: Path) -> ProspectiveRoundRecord:
    try:
        return ProspectiveRoundRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise InstrumentationError(f"invalid prior round record {path}: {error}")


def _read_artifacts(paths: Sequence[str | Path]) -> tuple[ReadArtifact, ...]:
    artifacts = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise InstrumentationError(f"read artifact not found: {path}")
        try:
            digest = sha256_file(path)
        except OSError as error:
            raise InstrumentationError(
                f"unable to hash read artifact {path}: {error}"
            ) from error
        artifacts.append(ReadArtifact(file_name=path.name, sha256=digest))
    if len({item.sha256 for item in artifacts}) != len(artifacts):
        raise InstrumentationError("duplicate read artifacts are not allowed")
    return tuple(artifacts)


def record_round(
    project_directory: str | Path,
    *,
    sample_id: str,
    coverage_effective: float,
    round_number: int,
    assembly_path: str | Path,
    samtools_stats_path: str | Path | None = None,
    alignment_reference_path: str | Path | None = None,
    offline_qc_path: str | Path | None = None,
    read_paths: Sequence[str | Path] = (),
) -> RoundRecordReceipt:
    """Atomically commit one immutable observation and cumulative history."""
    root = Path(project_directory).expanduser().resolve()
    project = load_project(root)
    sample_id = _validate_safe_id(sample_id, "sample_id")
    if round_number < 1 or round_number > project.max_rounds:
        raise InstrumentationError(
            f"round must be between 1 and project max_rounds={project.max_rounds}"
        )
    trajectory = root / "trajectories" / sample_id / _coverage_segment(
        coverage_effective
    )
    target = trajectory / f"R{round_number}"
    if target.exists():
        raise InstrumentationError(f"round is immutable and already exists: {target}")

    prior_directories = sorted(
        (path for path in trajectory.glob("R*") if path.is_dir()),
        key=lambda path: int(path.name[1:]),
    ) if trajectory.exists() else []
    prior_rounds = [int(path.name[1:]) for path in prior_directories]
    expected_prior = list(range(1, round_number))
    if prior_rounds != expected_prior:
        raise InstrumentationError(
            f"rounds must be recorded contiguously; expected prior rounds "
            f"{expected_prior}, found {prior_rounds}"
        )
    project_contract_sha256 = sha256_file(root / PROJECT_FILE_NAME)
    prior_records = [
        _read_round_record(path / "record.json") for path in prior_directories
    ]
    drifted = [
        record.round
        for record in prior_records
        if record.project_contract_sha256 != project_contract_sha256
    ]
    if drifted:
        raise InstrumentationError(
            "project contract differs from previously recorded rounds: "
            f"{drifted}"
        )

    observation = collect_light_round_observation(
        sample_id=sample_id,
        round_number=round_number,
        assembly_path=assembly_path,
        samtools_stats_path=samtools_stats_path,
        alignment_reference_path=alignment_reference_path,
    )
    observations = [
        _read_observation(path / "observation.json")
        for path in prior_directories
    ] + [observation]
    history: LightFeatureHistory = build_light_feature_history(
        observations,
        coverage_effective=coverage_effective,
    )

    offline_qc = load_offline_qc(offline_qc_path) if offline_qc_path else None
    if (
        offline_qc is not None
        and offline_qc.assembly_sha256 != observation.assembly_sha256
    ):
        raise InstrumentationError(
            "offline QC assembly_sha256 does not match the current assembly"
        )
    read_artifacts = _read_artifacts(read_paths)
    record = ProspectiveRoundRecord(
        project_id=project.project_id,
        project_contract_sha256=project_contract_sha256,
        sample_id=sample_id,
        coverage_effective=float(coverage_effective),
        round=round_number,
        offline_qc_file="offline-qc.json" if offline_qc else None,
        read_artifacts=read_artifacts,
    )

    trajectory.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".R{round_number}.", dir=trajectory)
    )
    try:
        _write_bytes(temporary / "observation.json", _json_bytes(observation))
        _write_bytes(temporary / "history.json", _json_bytes(history))
        if offline_qc is not None:
            _write_bytes(temporary / "offline-qc.json", _json_bytes(offline_qc))
        _write_bytes(temporary / "record.json", _json_bytes(record))
        os.replace(temporary, target)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise InstrumentationError(
            f"unable to commit round transaction {target}: {error}"
        ) from error

    return RoundRecordReceipt(
        project_id=project.project_id,
        project_contract_sha256=project_contract_sha256,
        sample_id=sample_id,
        coverage_effective=float(coverage_effective),
        round=round_number,
        round_directory=str(target),
        observation_file=str(target / "observation.json"),
        history_file=str(target / "history.json"),
        record_file=str(target / "record.json"),
        offline_qc_file=(str(target / "offline-qc.json") if offline_qc else None),
    )
