# ESDP v2 development roadmap

## Branching model

- `main` remains the stable line associated with the submitted manuscript and
  the v1.0.2 release.
- `develop-v2` is the integration branch for the next major version.
- Each change is developed in a short-lived branch and merged into
  `develop-v2` only after its acceptance criteria pass.
- Experimental models and generated benchmarks must not replace v1 artifacts.

## Delivery order

### 1. `fix/temporal-feature-leakage`

Goal: guarantee that features calculated for round `r` only use observations
available at or before round `r`.

Scope:

- Audit every engineered feature for temporal dependencies.
- Replace whole-trajectory plateau thresholds with prospective calculations.
- Add truncation-invariance tests for rounds 1 through 5.
- Record any metric changes and retrain candidate artifacts separately.

Acceptance criteria:

- Features for round `r` are identical whether the input ends at `r` or also
  contains later rounds.
- Tests fail if a future round influences a previous feature vector.
- Existing v1 model and benchmark artifacts remain unchanged.

### 2. `refactor/unified-feature-builder`

Goal: use one feature-engineering implementation for training, evaluation,
CLI, API, and workflow inference.

Scope:

- Introduce a reusable `FeatureBuilder`.
- Remove duplicated offline and online feature preparation.
- Define the canonical feature order, types, and missing-value behavior.
- Add parity tests between training and inference paths.

Acceptance criteria:

- The offline and online paths produce identical vectors for the same history.
- No production path silently invents an alternative feature definition.

### 3. `feat/trajectory-input-schema`

Goal: represent inference as a polishing history rather than an isolated row.

Scope:

- Add typed `RoundMetrics`, `PolishingTrajectory`, and `Decision` schemas.
- Validate round order, uniqueness, ranges, and required history.
- Add explicit `STOP`, `CONTINUE`, and conservative fallback actions.
- Preserve a documented compatibility path for the v1 interface.

Acceptance criteria:

- Invalid or incomplete histories produce explicit validation errors or a
  configured conservative continuation.
- A recommendation never requests a round earlier than the current round.

### 4. `feat/model-manifest`

Goal: make model artifacts traceable and compatibility-checked.

Scope:

- Add model, feature-schema, training-data, and source commit identifiers.
- Store supported assemblers, polishers, rounds, and software versions.
- Verify artifact checksums and schema compatibility at load time.
- Load and cache the model once per CLI or API process.

Acceptance criteria:

- An incompatible schema or corrupted artifact cannot be used silently.
- Every decision reports model and feature-schema versions.

### 5. `feat/production-cli`

Goal: provide a stable workflow-oriented command-line contract.

Scope:

- Package ESDP through `pyproject.toml`.
- Add `esdp decide`, `esdp validate`, and `esdp model-info`.
- Use deterministic JSON input and output.
- Separate core, API, and training dependency groups.

Acceptance criteria:

- The CLI runs without starting the REST service.
- Exit codes and outputs are documented and covered by integration tests.
- Container execution works with Docker and Apptainer-compatible runtimes.

### 6. `feat/esdp-light`

Goal: develop a lower-cost decision model that does not require expensive
quality assessment after every polishing round.

Scope:

- Define a feature set based on Flye, Minimap2, Samtools, assembly statistics,
  and historical deltas.
- Compare full, light, and R1-only variants using grouped validation.
- Include metric-generation cost in the resource benchmark.
- Calibrate confidence and conservative fallback behavior without using the
  final test set for threshold selection.

Acceptance criteria:

- Evaluation is grouped by biological sample.
- Reported savings include the cost of producing decision metrics.
- Model limitations and supported domain are encoded in the manifest.

### 7. `feat/nextflow-module`

Goal: prove adaptive Flye/Racon polishing in a minimal Nextflow DSL2 workflow
before modifying nf-core/bacass.

Scope:

- Add local ESDP and metric-collection modules.
- Implement explicit Racon rounds with per-round decisions.
- Select exactly one final assembly per sample.
- Apply Medaka after the selected Racon round.
- Add nf-test coverage, stub runs, version reporting, and conservative failure
  handling.

Acceptance criteria:

- A test dataset demonstrates both early stop and continuation to round 5.
- Failure or low-confidence behavior defaults to continued polishing.
- The workflow produces traceable decisions, metrics, and final assemblies.

## nf-core/bacass integration

After the standalone Nextflow proof of concept is validated, integration will
be developed in a separate branch of the bacass fork based on the current
`nf-core/bacass` default branch.

Initial supported path:

```text
Oxford Nanopore reads
  -> Flye
  -> adaptive Minimap2/Racon rounds controlled by ESDP
  -> Medaka
  -> final QUAST/BUSCO
```

Dragonflye and other assemblers remain on the existing path until they have
independent validation.

## Versioning and release gates

- v1 artifacts stay immutable.
- Candidate v2 models use prerelease identifiers such as `2.0.0-rc1`.
- `develop-v2` is merged into `main` only after temporal invariance, feature
  parity, artifact compatibility, CLI integration, and end-to-end workflow
  tests pass.
- The first nf-core contribution is proposed only after prospective resource
  and quality benchmarking of the Nextflow implementation.
