# ESDP model manifest

ESDP loads the bundled model through
`models/model_manifest.v1.json`. The manifest records the exact artifact,
feature contract, training dataset identity, source revision, pipeline
structure, and supported scientific domain.

The repository release version and serialized model version are different
identifiers:

- software release associated with the manuscript: `v1.0.2`;
- bundled model artifact metadata: `v1.1.0`;
- legacy feature schema used to train that artifact: `1.0.0`.

The manifest does not relabel or modify the existing model. It makes the
artifact's own metadata explicit and verifiable.

## Verified identities

The v1 manifest records SHA-256 checksums for:

- `models/best_model_pipeline.pkl`;
- `models/feature_names.txt`;
- `data/training_dataset_with_target.csv`.

Inference verifies the model and feature-name checksums before deserializing
the joblib artifact. Repository audits can additionally verify the training
dataset:

```python
from esdp_manifest import read_manifest, verify_manifest_files

manifest_path = "models/model_manifest.v1.json"
manifest = read_manifest(manifest_path)
verify_manifest_files(
    manifest,
    manifest_path,
    include_training_data=True,
)
```

## Compatibility checks

After deserialization, ESDP verifies:

- the Python artifact type;
- model version;
- exact feature names and order;
- pipeline step names, order, and implementation types;
- sample-level split grouping;
- label unit.

Any mismatch raises `ModelCompatibilityError`. Missing or checksum-mismatched
files raise `ArtifactIntegrityError`.

The manifest declares the current scientific support boundary:

- Oxford Nanopore sequencing;
- bacterial isolate genomes;
- Flye assembly;
- iterative Racon polishing;
- rounds R1–R5.

This does not imply validation for other assemblers, polishers, metagenomes,
or sequencing platforms.

## Prospective-v2 compatibility

The bundled v1 model has:

```json
{
  "feature_schema": {
    "version": "1.0.0",
    "prospective": false
  }
}
```

Its plateau features were trained using the original complete-trajectory
definition. It must not be presented as compatible with prospective feature
schema `2.0.0`.

```python
from esdp_manifest import load_verified_model

load_verified_model(
    "models/model_manifest.v1.json",
    required_feature_schema_version="2.0.0",
)
```

This call intentionally fails. A future prospective model must be retrained,
evaluated, and published with a separate v2 manifest and checksum.

## Cached loading

`load_verified_model()` caches the verified model per process. The cache key
includes the manifest checksum, so changing the manifest invalidates the
cached entry. If an artifact on disk changes after a valid model has already
been loaded, the process continues using the previously verified in-memory
object rather than deserializing the changed file.

The default `decide()` path, FastAPI model information endpoint, and batch
predictor use this loader. Custom model paths without an explicit manifest
remain supported for compatibility but report their feature schema as
`unverified`.

## Trust boundary

Joblib artifacts use Python pickle semantics. A checksum protects against
accidental corruption and substitution relative to the trusted manifest; it
does not make an untrusted manifest or pickle safe. Only manifests and model
files distributed through a trusted ESDP release should be loaded.

Cryptographic signing of release manifests is outside the current scope and
can be added to the release process later.
