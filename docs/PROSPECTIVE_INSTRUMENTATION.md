# Native prospective instrumentation

## Purpose

ESDP can collect a prospective polishing dataset directly, without Nextflow
and without activating an experimental stopping model. The instrumentation
layer is available through both the `esdp` command and a Python API.

The project contract permanently records:

- sequencing platform, chemistry, and ONT basecalling provenance;
- assembler and polisher identities, versions, and frozen parameters;
- online low-cost metrics and optional same-round alignment statistics;
- expected offline outcome families;
- an explicit `decision_enabled: false` gate.

Initialization does not run BUSCO, Merqury, an aligner, or a gene predictor.
Those tools have different resource and environment requirements. Instead,
ESDP generates a strict JSON Schema and validates their results when a round is
recorded. This keeps collection reproducible without silently making expensive
tools runtime dependencies.

## Initialize a project

For Oxford Nanopore data:

```bash
esdp init \
  --project-directory esdp-study \
  --project-id bacteria-ont-001 \
  --platform ont \
  --chemistry R10.4.1 \
  --basecaller Dorado \
  --basecaller-version 0.9.0 \
  --basecaller-model sup@v5.0.0 \
  --assembler Flye \
  --assembler-version 2.9.6 \
  --assembler-parameter=--nano-hq \
  --polisher Racon \
  --polisher-version 1.5.0 \
  --polisher-parameter=-m=8
```

ONT projects require chemistry, basecaller, basecaller version, and model.
PacBio HiFi and CLR projects require chemistry but not an ONT basecaller.

The target directory must be absent or empty. Initialization creates:

```text
esdp-study/
├── esdp-project.json
├── schemas/
│   └── offline-qc.schema.json
└── trajectories/
```

The generated project and schema are deterministic. The contract cannot enable
STOP/CONTINUE decisions.

## Record one polishing round

```bash
esdp record-round \
  --project-directory esdp-study \
  --sample-id isolate-001 \
  --coverage-effective 40 \
  --round 1 \
  --assembly isolate-001.r1.fasta \
  --reads isolate-001.fastq.gz
```

This performs the existing `light-observe` and `light-history` operations as
one immutable transaction. It calculates FASTA statistics, records assembly
and optional read identities, validates contiguous rounds, and stores a causal
history containing only information available through the current round.

Optional same-round alignment metrics require both artifacts:

```bash
esdp record-round \
  --project-directory esdp-study \
  --sample-id isolate-001 \
  --coverage-effective 40 \
  --round 2 \
  --assembly isolate-001.r2.fasta \
  --samtools-stats isolate-001.r2.stats \
  --alignment-reference isolate-001.r2.fasta
```

The alignment reference must have exactly the same SHA-256 identity as the
polished assembly. This prevents pre-polish statistics from being attributed
to a post-polish round.

## Offline quality outcomes

An optional `--offline-qc` JSON can contain one or more outcome families:

- BUSCO marker counts with version, lineage dataset, creation date, mode, and
  options;
- independent-read k-mer QV and completeness;
- separate substitutions, insertions, deletions, and homopolymer indels;
- predicted CDS, frameshifts, premature stops, and truncations.

The input must validate against `schemas/offline-qc.schema.json` and its
`assembly_sha256` must match the current assembly. These outcomes are archived
for scientific supervision but never inserted into the light feature history.

Example BUSCO fragment:

```json
{
  "schema_version": "1.0.0",
  "assembly_sha256": "<current assembly SHA-256>",
  "busco": {
    "complete_count": 110,
    "single_copy_count": 109,
    "duplicated_count": 1,
    "fragmented_count": 2,
    "missing_count": 4,
    "marker_count": 116,
    "version": "6.1.0",
    "lineage_dataset": "bacteria_odb12.2",
    "lineage_creation_date": "2026-05-22",
    "mode": "genome",
    "options": ["--offline"]
  }
}
```

## Immutable layout

Each recorded round contains its own observation and cumulative history:

```text
trajectories/isolate-001/coverage-40/
├── R1/
│   ├── observation.json
│   ├── history.json
│   └── record.json
└── R2/
    ├── observation.json
    ├── history.json
    ├── offline-qc.json
    └── record.json
```

Round directories are committed by atomic directory rename. Each record stores
the SHA-256 identity of `esdp-project.json`, and a changed project contract
cannot be mixed with previously recorded rounds. Existing rounds
cannot be overwritten, rounds cannot be skipped, and recording R2 does not
modify the R1 history. `record.json` is the transaction commit marker and
always contains `decision_enabled: false`.

## Python API

The same operations can be called without subprocesses:

```python
from esdp_instrumentation import (
    LongReadTechnology,
    ToolIdentity,
    init_project,
    record_round,
)

init_project(
    "esdp-study",
    project_id="bacteria-ont-001",
    technology=LongReadTechnology(
        platform="ont",
        chemistry="R10.4.1",
        basecaller="Dorado",
        basecaller_version="0.9.0",
        basecaller_model="sup@v5.0.0",
    ),
    assembler=ToolIdentity(name="Flye", version="2.9.6"),
    polisher=ToolIdentity(name="Racon", version="1.5.0"),
)

receipt = record_round(
    "esdp-study",
    sample_id="isolate-001",
    coverage_effective=40,
    round_number=1,
    assembly_path="isolate-001.r1.fasta",
    read_paths=("isolate-001.fastq.gz",),
)
```

The returned receipt identifies every committed artifact. Neither function
loads a model or makes a polishing decision.

## What remains external

The instrumentation contract records results but does not yet execute:

- read-to-assembly alignment;
- BUSCO;
- k-mer QV evaluation;
- homopolymer-aware error analysis;
- coding-integrity analysis.

These can initially run as separate commands or scripts and submit validated
JSON. Once their commands, containers, runtime, and portability are frozen,
they can become optional ESDP collectors without changing the project schema.
