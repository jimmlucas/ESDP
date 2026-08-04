"""Low-cost, provenance-checked metric extraction for ESDP-light."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from esdp_manifest import sha256_file


LIGHT_OBSERVATION_SCHEMA_VERSION = "1.0.0"
IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")


class LightMetricError(ValueError):
    """Base error for invalid or temporally ambiguous light metrics."""


class AlignmentReferenceMismatchError(LightMetricError):
    """Raised when the supplied alignment reference differs from the assembly."""


class FastaMetrics(BaseModel):
    """Reference-free statistics computed directly from one assembly FASTA."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    num_contigs: int = Field(ge=1)
    total_length: int = Field(ge=1)
    n50: int = Field(ge=1)
    gc_percent: float = Field(ge=0, le=100)
    acgt_bases: int = Field(ge=0)
    ambiguous_bases: int = Field(ge=0)


class SamtoolsAlignmentMetrics(BaseModel):
    """Mapping metrics parsed from samtools stats for the current assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mapping_error_rate: float = Field(ge=0, le=1)
    bases_mapped_cigar: int = Field(ge=0)
    mismatches: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.mismatches > self.bases_mapped_cigar:
            raise ValueError("mismatches cannot exceed bases_mapped_cigar")
        return self


class LightRoundObservation(BaseModel):
    """One round of light metrics tied to exact assembly identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = LIGHT_OBSERVATION_SCHEMA_VERSION
    sample_id: str = Field(min_length=1)
    round: int = Field(ge=1, le=5)
    assembly_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fasta: FastaMetrics
    alignment: SamtoolsAlignmentMetrics | None = None
    alignment_reference_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    samtools_stats_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_alignment_provenance(self):
        alignment_fields = (
            self.alignment,
            self.alignment_reference_sha256,
            self.samtools_stats_sha256,
        )
        if any(value is not None for value in alignment_fields) and any(
            value is None for value in alignment_fields
        ):
            raise ValueError(
                "alignment metrics, reference identity, and stats identity "
                "must be provided together"
            )
        if (
            self.alignment_reference_sha256 is not None
            and self.alignment_reference_sha256 != self.assembly_sha256
        ):
            raise ValueError(
                "alignment reference must be the polished assembly for this round"
            )
        return self


def _open_fasta(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def calculate_fasta_metrics(path: str | Path) -> FastaMetrics:
    """Calculate N50, contig count, length, and GC without QUAST."""
    fasta_path = Path(path).expanduser().resolve()
    if not fasta_path.is_file():
        raise LightMetricError(f"assembly FASTA not found: {fasta_path}")

    lengths: list[int] = []
    current_length = 0
    acgt_bases = 0
    gc_bases = 0
    ambiguous_bases = 0
    seen_header = False

    try:
        with _open_fasta(fasta_path) as fasta_file:
            for line_number, raw_line in enumerate(fasta_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seen_header:
                        if current_length == 0:
                            raise LightMetricError(
                                f"empty FASTA record before line {line_number}"
                            )
                        lengths.append(current_length)
                    seen_header = True
                    current_length = 0
                    continue
                if not seen_header:
                    raise LightMetricError(
                        "FASTA sequence appears before the first header at "
                        f"line {line_number}"
                    )

                sequence = "".join(line.split()).upper()
                invalid = sorted(set(sequence) - IUPAC_DNA)
                if invalid:
                    raise LightMetricError(
                        f"invalid FASTA symbols at line {line_number}: {invalid}"
                    )
                current_length += len(sequence)
                acgt_bases += sum(sequence.count(base) for base in "ACGT")
                gc_bases += sequence.count("G") + sequence.count("C")
                ambiguous_bases += sum(
                    sequence.count(base) for base in IUPAC_DNA - set("ACGT")
                )
    except OSError as error:
        raise LightMetricError(f"unable to read FASTA {fasta_path}: {error}") from error

    if not seen_header:
        raise LightMetricError("assembly FASTA contains no records")
    if current_length == 0:
        raise LightMetricError("final FASTA record is empty")
    lengths.append(current_length)

    total_length = sum(lengths)
    halfway = total_length / 2
    cumulative = 0
    n50 = 0
    for length in sorted(lengths, reverse=True):
        cumulative += length
        if cumulative >= halfway:
            n50 = length
            break

    gc_percent = (100 * gc_bases / acgt_bases) if acgt_bases else 0.0
    return FastaMetrics(
        num_contigs=len(lengths),
        total_length=total_length,
        n50=n50,
        gc_percent=gc_percent,
        acgt_bases=acgt_bases,
        ambiguous_bases=ambiguous_bases,
    )


def parse_samtools_stats(path: str | Path) -> SamtoolsAlignmentMetrics:
    """Parse required SN records without inventing fallback values."""
    stats_path = Path(path).expanduser().resolve()
    if not stats_path.is_file():
        raise LightMetricError(f"samtools stats file not found: {stats_path}")

    values: dict[str, str] = {}
    try:
        with stats_path.open(encoding="utf-8", errors="strict") as stats_file:
            for raw_line in stats_file:
                if not raw_line.startswith("SN"):
                    continue
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                values[parts[1].strip().rstrip(":").lower()] = parts[2].strip()
    except OSError as error:
        raise LightMetricError(
            f"unable to read samtools stats {stats_path}: {error}"
        ) from error

    required = ("error rate", "bases mapped (cigar)", "mismatches")
    missing = [name for name in required if name not in values]
    if missing:
        raise LightMetricError(f"samtools stats missing required SN fields: {missing}")

    try:
        return SamtoolsAlignmentMetrics(
            mapping_error_rate=float(values["error rate"].replace(",", "")),
            bases_mapped_cigar=int(
                values["bases mapped (cigar)"].replace(",", "")
            ),
            mismatches=int(values["mismatches"].replace(",", "")),
        )
    except (TypeError, ValueError) as error:
        raise LightMetricError(
            f"invalid numeric value in samtools stats: {error}"
        ) from error


def collect_light_round_observation(
    *,
    sample_id: str,
    round_number: int,
    assembly_path: str | Path,
    samtools_stats_path: str | Path | None = None,
    alignment_reference_path: str | Path | None = None,
) -> LightRoundObservation:
    """Collect one round and reject pre-polish/mismatched alignment metrics."""
    if bool(samtools_stats_path) != bool(alignment_reference_path):
        raise LightMetricError(
            "samtools_stats_path and alignment_reference_path are required together"
        )

    assembly = Path(assembly_path).expanduser().resolve()
    fasta_metrics = calculate_fasta_metrics(assembly)
    try:
        assembly_sha256 = sha256_file(assembly)
    except OSError as error:
        raise LightMetricError(
            f"unable to hash assembly FASTA {assembly}: {error}"
        ) from error

    alignment = None
    reference_sha256 = None
    stats_sha256 = None
    if samtools_stats_path is not None and alignment_reference_path is not None:
        reference = Path(alignment_reference_path).expanduser().resolve()
        if not reference.is_file():
            raise LightMetricError(
                f"alignment reference not found: {reference}"
            )
        try:
            reference_sha256 = sha256_file(reference)
        except OSError as error:
            raise LightMetricError(
                f"unable to hash alignment reference {reference}: {error}"
            ) from error
        if reference_sha256 != assembly_sha256:
            raise AlignmentReferenceMismatchError(
                "samtools alignment reference is not the polished assembly "
                f"for round {round_number}"
            )
        alignment = parse_samtools_stats(samtools_stats_path)
        try:
            stats_sha256 = sha256_file(samtools_stats_path)
        except OSError as error:
            raise LightMetricError(
                f"unable to hash samtools stats {samtools_stats_path}: {error}"
            ) from error

    try:
        return LightRoundObservation(
            sample_id=sample_id.strip(),
            round=round_number,
            assembly_sha256=assembly_sha256,
            fasta=fasta_metrics,
            alignment=alignment,
            alignment_reference_sha256=reference_sha256,
            samtools_stats_sha256=stats_sha256,
        )
    except ValueError as error:
        raise LightMetricError(str(error)) from error
