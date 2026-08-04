# ESDP production CLI

The v2 development line provides a workflow-oriented `esdp` command without
starting the REST API. It is intended for local execution, containers, and
workflow engines such as Nextflow.

## Installation

Core decision runtime:

```bash
python -m pip install .
```

Optional dependency groups are separate:

```bash
python -m pip install '.[api]'
python -m pip install '.[training]'
python -m pip install '.[dev]'
```

The CLI and core runtime require Python 3.10 or newer. The manuscript release
and its model remain versioned independently:

- CLI development version: `2.0.0.dev0`;
- bundled model: `v1.1.0`;
- bundled legacy feature schema: `1.0.0`.

## Commands

### `esdp decide`

Make one decision from a strictly validated single-round metrics object:

```bash
esdp decide \
  --input metrics.json \
  --output decision.json
```

The required decision metrics are `qv`, `busco_complete`, `n50`,
`num_contigs`, `error_rate`, and `total_length`. Missing essential metrics
produce exit code `3`; the surrounding workflow can then apply its
conservative continuation policy instead of accepting a prediction based
entirely on imputation.

Both paths may be `-` for standard input or standard output:

```bash
esdp decide --input - --output - < metrics.json
```

The command writes only deterministic JSON to standard output. To remove
irrelevant thread-level floating-point reduction noise, floating values are
serialized to at most ten decimal places. Operational messages and warnings
use standard error. File output is written to a temporary file in the
destination directory and atomically renamed only after the complete JSON
document is flushed.

The output reports:

- decision contract version;
- model and feature-schema versions;
- recommendation, confidence, probabilities, and rule overrides;
- current round, next round, selected final round, and `STOP` or `CONTINUE`.

The bundled model uses the legacy feature schema `1.0.0`. The CLI does not
claim that this artifact is a prospective v2 model. A future v2 model must be
retrained and published with feature schema `2.0.0`.

### `esdp validate`

Validate a legacy metrics document or a v2 trajectory:

```bash
esdp validate --input trajectory.json
esdp validate --kind metrics --input metrics.json
```

The default `--kind auto` selects a trajectory when the document contains
`schema_version: "2.0.0"` or a `rounds` array. Success produces a small JSON
validation receipt with `complete` and `missing_metrics`. An incomplete
trajectory configured with `incomplete_history_policy: "ERROR"` exits with
code `3`; the conservative policy remains structurally valid and reports its
missing metrics.

### `esdp model-info`

Verify the bundled model and report its complete manifest:

```bash
esdp model-info --output model-info.json
```

Repository audits can also verify the training dataset:

```bash
esdp model-info --verify-training-data
```

That option is expected to fail in a minimal installed wheel or runtime
container because training data are deliberately excluded from the core
package.

### `esdp light-observe`

Collect one versioned ESDP-light observation directly from the polished
assembly, without running the legacy model:

```bash
esdp light-observe \
  --sample-id sample_001 \
  --round 2 \
  --assembly polished_r2.fasta \
  --output observation.r2.json
```

The command calculates reference-free N50, contig count, total length, GC,
ACGT bases, and ambiguous bases. It records the SHA-256 identity of the FASTA
and accepts plain or gzip-compressed input.

Optional post-polish alignment statistics require both inputs:

```bash
esdp light-observe \
  --sample-id sample_001 \
  --round 2 \
  --assembly polished_r2.fasta \
  --samtools-stats polished_r2.stats \
  --alignment-reference polished_r2.fasta \
  --output observation.r2.json
```

The supplied alignment reference must be byte-identical to the current
polished assembly. The JSON records hashes for the assembly, supplied
reference, and Samtools artifact. The stats parser requires explicit `error
rate`, `bases mapped (cigar)`, and `mismatches` summary fields and never fills
missing values with defaults.

This command does not make a stopping decision. `mapping_error_rate` remains
an experimental candidate until the complete Minimap2/Samtools process,
software versions, and incremental cost are frozen and evaluated.

### `esdp light-history`

Aggregate all observations available for one sample and effective-coverage
trajectory:

```bash
esdp light-history \
  --observation observation.r1.json \
  --observation observation.r2.json \
  --coverage-effective 40 \
  --output history.r2.json
```

Observation arguments may arrive in any order. The command validates that
they use observation schema `1.0.0`, belong to one sample, start at R1, and
form a contiguous sequence without duplicate rounds. It then emits history
schema `1.0.0` with:

- exact assembly hashes for every source round;
- the deployment-ready N50, contig count, total length, and GC values;
- round-to-round deltas;
- changes relative to R1;
- changes in the deltas between consecutive rounds;
- an ordered feature-name contract.

Unavailable causal quantities are JSON `null`: R1 has no round delta, and R1
and R2 have no delta trend. The output never encodes these states as nonstandard
`NaN` values. Future observations cannot modify previously generated rows.

The aggregator deliberately excludes provisional alignment values even when
an input observation contains them. This prevents `mapping_error_rate` from
entering an inference contract before its workflow and cost validation gates
are complete.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Command completed successfully |
| `2` | Invalid command-line syntax (`argparse`) |
| `3` | Invalid, missing, or unreadable input |
| `4` | Model integrity, compatibility, or inference failure |
| `5` | Output serialization or write failure |

On every nonzero result, no decision JSON is written to standard output.

## Container execution

The Docker image exposes the same command:

```bash
docker run --rm \
  -v "$PWD:/work" \
  jimmlucas/esdp \
  esdp decide --input /work/metrics.json --output /work/decision.json
```

The entrypoint bypasses API startup messages when the first command is
`esdp`, preserving clean JSON output. The same image layout is suitable for
Apptainer execution.

## Nextflow contract

A Nextflow process should use explicit staged files:

```nextflow
script:
"""
esdp decide \
  --input ${metrics_json} \
  --output decision.json
"""
```

The process should treat exit codes `3`–`5` conservatively and continue
polishing unless the workflow policy explicitly chooses to fail.

An ESDP-light collection process can remain independent of model inference:

```nextflow
process ESDP_LIGHT_OBSERVE {
    tag "${meta.id}:R${round}"

    input:
    tuple val(meta), val(round), path(assembly)

    output:
    tuple val(meta), val(round), path("*.esdp-light.json"), emit: observation

    script:
    """
    esdp light-observe \\
      --sample-id '${meta.id}' \\
      --round ${round} \\
      --assembly ${assembly} \\
      --output '${meta.id}.R${round}.esdp-light.json'
    """
}
```

Alignment generation and `light-observe` should eventually run under one
versioned workflow contract so the supplied reference cannot be separated
from the process that created the statistics.

After collecting the rounds currently available, a separate process can
produce the causal history artifact:

```nextflow
process ESDP_LIGHT_HISTORY {
    tag "${meta.id}:R${round}"

    input:
    tuple val(meta), val(round), val(coverage_effective), path(observations)

    output:
    tuple val(meta), val(round), path("*.esdp-light.history.json"), emit: history

    script:
    def observation_args = observations.collect { "--observation ${it}" }.join(' ')
    """
    esdp light-history \\
      ${observation_args} \\
      --coverage-effective ${coverage_effective} \\
      --output '${meta.id}.R${round}.esdp-light.history.json'
    """
}
```

This process still performs feature construction only. A later model process
must consume a frozen feature schema and verified ESDP-light manifest.
