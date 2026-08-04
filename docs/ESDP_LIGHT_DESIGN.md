# ESDP-light scientific design

ESDP-light is a separate research track for reducing the incremental cost of
adaptive polishing decisions. It does not replace or relabel the published v1
model. A light artifact will be released only after prospective, grouped
evaluation and will have its own feature schema, model version, and manifest.

## Dataset audit

The current labeled dataset contains:

- 805 round-level observations;
- 41 biological samples;
- 161 `Sample+Coverage` trajectories;
- exactly 161 observations for each round R1-R5;
- 32 training samples and 9 held-out test samples in the frozen split.

All audited Flye, assembly-statistics, `assembly_info.txt`, and alignment
columns except `raw_mean_read_len` are populated in the current table.
However, several complete columns are not reconstructed by the checked-in
collector, including `raw_n_reads`, Flye coverage thresholds, and extended
`assembly_info.txt` summaries. Dataset completeness therefore does not make
those fields deployment-ready.

Availability in the CSV is not sufficient evidence of reproducible
deployment. Alignment-derived columns contain values, but the checked-in
collector does not currently reconstruct those values. The historical
`error_rate` also uses mapping statistics produced against the assembly before
the current Racon output, creating ambiguous round alignment. These fields
remain provisional until a versioned Minimap2/Samtools extractor measures an
explicit `mapping_error_rate` against the polished assembly for the same
round.

## Cost and provenance tiers

1. `STATIC_ONCE`: read and Flye metrics produced once before adaptive rounds.
2. `ROUND_NATIVE`: FASTA or Flye `assembly_info.txt` metrics available without
   BUSCO, reference-based QUAST, or an additional whole-genome quality run.
3. `ALIGNMENT_REUSE`: metrics that may reuse the Racon alignment but still
   require a reproducible extractor and measured incremental cost.
4. `EXPENSIVE_QC`: BUSCO, QV, reference-derived error, and features derived
   from them. These are forbidden in ESDP-light inference.

`esdp_light.py` encodes these tiers. The default light contract accepts only
deployment-ready metrics. Alignment candidates require an explicit opt-in and
must not be used for a release artifact while their status is provisional.

## Lightweight round observation

`esdp_light_metrics.py` now provides a reference-free FASTA extractor for
`n50`, `num_contigs`, `total_length`, and `gc`. It reads plain or gzip FASTA
directly, rejects empty records and invalid sequence symbols, and requires no
QUAST or BUSCO execution. GC percentage uses A, C, G, and T as its denominator;
ambiguous IUPAC bases are excluded from that denominator and reported
separately. These four assembly metrics are therefore deployment-ready.

The same module defines a prospective alignment contract for
`mapping_error_rate`. Samtools statistics are accepted only when the alignment
reference and the current polished assembly have identical SHA-256 content
identities. The observation records hashes for the assembly, alignment
reference, and statistics file. Missing required Samtools `SN` fields are an
error; the collector does not invent defaults. This closes the pre-polish versus
post-polish ambiguity, but alignment metrics remain provisional until the
Minimap2/Samtools command, versions, runtime, and portability are validated.
Because a standalone Samtools stats file does not embed the reference hash,
this guarantee also depends on the future workflow producing the alignment,
stats, and provenance record in one isolated process. The current collector
verifies the supplied reference identity; it does not claim that an unrelated
stats file is cryptographically self-authenticating.

`esdp_light_features.py` builds round deltas, changes from R1, and delta trends
for the four FASTA metrics. Each `Sample+Coverage` trajectory must begin at R1
and have contiguous rounds. Features for round Rn are calculated only from
R1...Rn; automated truncation tests verify that changing a future round cannot
alter any earlier feature row.

The production CLI exposes this collection contract as `esdp light-observe`.
Its deterministic, atomically written JSON is a model-independent workflow
artifact: producing it does not imply that an ESDP-light classifier has been
trained, selected, calibrated, or released.

`esdp light-history` aggregates those observations under a second versioned
schema. It requires exactly one sample and coverage trajectory with contiguous
R1...Rn inputs, preserves each assembly identity, converts causal
not-yet-available values to JSON `null`, and excludes provisional alignment
metrics. A serialized history can be read back and revalidated against the
same schema.

## Explicit exclusions

The light model must exclude:

- QV, BUSCO, reference-derived error rate, and their deltas or trends;
- `assembly_frac`, whose provenance is ambiguous between configured Flye
  genome size and reference-based reporting in the current documentation;
- `r1_ok_group` and `stable_all_group`, which are policy/label-derived;
- plateau and score features currently defined using expensive quality
  signals;
- target columns and any feature selected using held-out test performance.

The training label may still be defined using expensive retrospective quality
assessment. That is supervision available during research, not an inference
input. This distinction must be explicit in the future manifest.

## Candidate comparisons

Three predefined families will be compared:

- `full`: the verified legacy feature contract, used only as a benchmark;
- `light`: deployment-ready low-cost metrics plus prospective historical
  deltas;
- `R1-only`: the same low-cost raw metrics restricted to the first round,
  without temporal deltas.

An alignment-augmented light candidate is exploratory until provenance and
runtime cost are validated.

## Evaluation protocol

The existing 32/9 sample split remains frozen. The nine test samples must not
be used for feature selection, hyperparameter selection, threshold tuning, or
probability calibration.

Development uses grouped cross-validation inside the 32 training samples.
The final held-out evaluation is run once after the feature contract,
hyperparameters, and conservative policy are frozen.

Primary evaluation is a prospective trajectory simulation, not only
row-level classification. It must report:

- unsafe early-stop rate;
- excess polishing rounds;
- selected final-round agreement;
- calibration and low-confidence continuation behavior;
- net compute saved after including metric-generation cost;
- uncertainty grouped or bootstrapped by biological sample.

No candidate becomes an ESDP-light release merely because it improves one
classification metric. Safety, calibration, compute savings, provenance, and
domain limitations are joint release gates.

The first development-only feasibility experiment is reported in
`ESDP_LIGHT_FEASIBILITY.md`. Under sample-grouped out-of-fold evaluation, the
current light candidates show weak signal but do not satisfy the classification
or safety-savings release gates. The result is a no-go for a production light
model and a go for a bounded endpoint/formulation study; the frozen held-out
test samples remain untouched.
