# ESDP-light measurement-validity gate

## Decision

The current material-benefit endpoint is not valid enough to authorize another
model-selection cycle. The low-cost observation and history infrastructure
remains valid, but model retraining, opening the held-out test set, production
`light-decide`, and adaptive nf-core/bacass stopping remain blocked.

This is a scientific gate, not a software failure.

## Exact metric dependencies

Across all 635 development rows:

```text
qv = -10 * log10(error_rate)
assembly_error = abs(assembly_frac - 1)
```

Both identities hold to floating-point precision. QV and error rate are not
independent endpoint dimensions, and assembly error is only another
representation of assembly fraction. A future endpoint must use a single
representation from each pair.

The historical error measurement also has ambiguous round provenance because
it can refer to an alignment produced before the corresponding polished
assembly. The replacement must be a versioned same-round
`mapping_error_rate`; QV may be derived from it for presentation, but must not
be counted as a second outcome.

## BUSCO resolution and provenance

Every development trajectory uses `busco_n = 116`. One marker therefore
represents 0.862 percentage points. Because the current materiality test uses
`>1.0` percentage point, a positive BUSCO event requires at least two marker
genes.

This resolves the numerical meaning of the threshold, but not its biological
meaning:

- 38/127 trajectories have some post-R1 BUSCO gain above one point;
- only 26/127 retain a gain above one point at R5;
- 12/127 (9.4%) therefore have only a transient gain;
- adjacent changes are positive in 31.5%, zero in 37.2%, and negative in
  31.3% of round transitions;
- the mean R1-to-R5 change is -0.078 percentage points.

The material-benefit endpoint currently treats every transient gain as a
reason to continue. That choice is not yet biologically justified.

The dataset stores BUSCO percentages and `n`, but not the BUSCO version,
lineage dataset, dataset creation date, mode, or options. These fields are
required for a reproducible endpoint contract because BUSCO scores are defined
against lineage-specific marker sets and classification behavior depends on
the dataset version. This follows the
[official BUSCO user guide](https://busco.ezlab.org/busco_userguide.html),
which also recommends reporting the tool/dependency versions, lineage set and
creation date, options, and assessed assembly version.

## Repeatability limitation

The development table has one observation for each
`Sample+Coverage_effective+round` cell. Coverage levels are experimental
conditions, not repeated measurements of the same assembly artifact.
Consequently, technical repeatability and assay noise cannot be estimated from
the current dataset.

BUSCO may be computationally deterministic under a frozen environment, but
that does not establish a minimal biologically important difference or prove
that a temporary two-marker change should control polishing.

## Versioned development contract

The machine-readable `materiality_contract.json` records:

- QV/error: `invalid_redundant_and_provenance_ambiguous`;
- BUSCO complete: `provisional`;
- assembly fraction/error: `provisional_redundant_representation`;
- overall release gate: `blocked`.

To unblock the next model experiment, new development data must provide:

1. Same-round mapping-error provenance: assembly hash, reference hash,
   aligner/Samtools versions, commands, and an unambiguous error definition.
2. BUSCO complete marker counts, BUSCO/dependency versions, lineage dataset and
   creation date, mode, options, and assembly identity.
3. Expected-genome-size provenance when assembly fraction is used.
4. A preregistered rule stating whether a BUSCO improvement must persist, and
   the maximum acceptable cost of missing a transient improvement.
5. Repeated or otherwise independently validated measurements sufficient to
   justify minimal important differences.

Only after these fields and rules are frozen should nested grouped validation
resume. The existing nine held-out samples must remain sealed until that model,
calibration method, threshold, and asymmetric safety gate are fixed.

## Nextflow implication

The safe nf-core integration path remains two-stage:

- now: expose `light-observe` and `light-history` as optional provenance and
  feature-generation modules with no effect on polishing decisions;
- later: add `light-decide` only when a versioned materiality contract and
  independently evaluated release artifact pass the safety gate.

This preserves the useful ESDP core while preventing an experimental endpoint
from silently changing scientific workflow behavior.

## Reproducibility

Run from the repository root:

```bash
python -m experiments.esdp_light_measurement_validity
```

Outputs are stored in `outputs/esdp_light_measurement_validity/` and include
the versioned contract, strict JSON report, adjacent-change audit, and BUSCO
trajectory audit.
