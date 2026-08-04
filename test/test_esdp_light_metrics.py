"""Tests for low-cost ESDP-light metric extraction and provenance."""

import gzip
from pathlib import Path

import pytest

from esdp_light_metrics import (
    AlignmentReferenceMismatchError,
    LightMetricError,
    calculate_fasta_metrics,
    collect_light_round_observation,
    parse_samtools_stats,
)
from esdp_manifest import sha256_file


SAMTOOLS_STATS = """# This file was produced by samtools stats
SN\traw total sequences:\t100
SN\terror rate:\t1.250000e-02
SN\tbases mapped (cigar):\t1,000
SN\tmismatches:\t12
"""


def test_calculate_fasta_metrics_without_external_qc(tmp_path: Path):
    assembly = tmp_path / "assembly.fasta"
    assembly.write_text(">long\nGGCCAAAANN\n>short\nATGC\n", encoding="utf-8")

    metrics = calculate_fasta_metrics(assembly)

    assert metrics.num_contigs == 2
    assert metrics.total_length == 14
    assert metrics.n50 == 10
    assert metrics.acgt_bases == 12
    assert metrics.ambiguous_bases == 2
    assert metrics.gc_percent == pytest.approx(50.0)


def test_calculate_fasta_metrics_accepts_gzip(tmp_path: Path):
    assembly = tmp_path / "assembly.fa.gz"
    with gzip.open(assembly, "wt", encoding="utf-8") as output:
        output.write(">contig\nACGTN\n")

    metrics = calculate_fasta_metrics(assembly)

    assert metrics.total_length == 5
    assert metrics.gc_percent == pytest.approx(50.0)
    assert metrics.ambiguous_bases == 1


@pytest.mark.parametrize(
    "content, message",
    [
        ("ACGT\n", "before the first header"),
        (">empty\n", "final FASTA record is empty"),
        (">contig\nACGTZ\n", "invalid FASTA symbols"),
    ],
)
def test_calculate_fasta_metrics_rejects_invalid_fasta(
    tmp_path: Path,
    content: str,
    message: str,
):
    assembly = tmp_path / "invalid.fasta"
    assembly.write_text(content, encoding="utf-8")

    with pytest.raises(LightMetricError, match=message):
        calculate_fasta_metrics(assembly)


def test_parse_samtools_stats_requires_explicit_fields(tmp_path: Path):
    complete = tmp_path / "complete.stats"
    complete.write_text(SAMTOOLS_STATS, encoding="utf-8")

    metrics = parse_samtools_stats(complete)

    assert metrics.mapping_error_rate == pytest.approx(0.0125)
    assert metrics.bases_mapped_cigar == 1000
    assert metrics.mismatches == 12

    incomplete = tmp_path / "incomplete.stats"
    incomplete.write_text("SN\terror rate:\t0.01\n", encoding="utf-8")
    with pytest.raises(LightMetricError, match="missing required SN fields"):
        parse_samtools_stats(incomplete)


def test_observation_verifies_supplied_alignment_reference(tmp_path: Path):
    assembly = tmp_path / "polished.fasta"
    assembly.write_text(">contig\nACGTACGT\n", encoding="utf-8")
    reference = tmp_path / "alignment-reference.fasta"
    reference.write_bytes(assembly.read_bytes())
    stats = tmp_path / "alignment.stats"
    stats.write_text(SAMTOOLS_STATS, encoding="utf-8")

    observation = collect_light_round_observation(
        sample_id="sample-1",
        round_number=2,
        assembly_path=assembly,
        samtools_stats_path=stats,
        alignment_reference_path=reference,
    )

    assert observation.assembly_sha256 == sha256_file(assembly)
    assert observation.alignment_reference_sha256 == observation.assembly_sha256
    assert observation.samtools_stats_sha256 == sha256_file(stats)
    assert observation.alignment is not None
    assert observation.alignment.mapping_error_rate == pytest.approx(0.0125)


def test_observation_rejects_pre_polish_or_different_reference(tmp_path: Path):
    assembly = tmp_path / "polished.fasta"
    assembly.write_text(">contig\nACGTACGT\n", encoding="utf-8")
    pre_polish = tmp_path / "pre-polish.fasta"
    pre_polish.write_text(">contig\nACGTTCGT\n", encoding="utf-8")
    stats = tmp_path / "alignment.stats"
    stats.write_text(SAMTOOLS_STATS, encoding="utf-8")

    with pytest.raises(
        AlignmentReferenceMismatchError,
        match="not the polished assembly",
    ):
        collect_light_round_observation(
            sample_id="sample-1",
            round_number=2,
            assembly_path=assembly,
            samtools_stats_path=stats,
            alignment_reference_path=pre_polish,
        )


def test_observation_requires_stats_and_reference_as_a_pair(tmp_path: Path):
    assembly = tmp_path / "assembly.fasta"
    assembly.write_text(">contig\nACGT\n", encoding="utf-8")
    stats = tmp_path / "alignment.stats"
    stats.write_text(SAMTOOLS_STATS, encoding="utf-8")

    with pytest.raises(LightMetricError, match="required together"):
        collect_light_round_observation(
            sample_id="sample-1",
            round_number=1,
            assembly_path=assembly,
            samtools_stats_path=stats,
        )


def test_observation_without_alignment_is_valid(tmp_path: Path):
    assembly = tmp_path / "assembly.fasta"
    assembly.write_text(">contig\nACGT\n", encoding="utf-8")

    observation = collect_light_round_observation(
        sample_id="sample-1",
        round_number=1,
        assembly_path=assembly,
    )

    assert observation.alignment is None
    assert observation.alignment_reference_sha256 is None
    assert observation.samtools_stats_sha256 is None
