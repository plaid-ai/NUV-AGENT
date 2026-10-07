# AnomalyVFM / RADIO QNN Fleet package

## Scope

The v16 MinMax W8A16 + V-range context is qualified for the verified Ventuno Q
prototype (`ventuno_q`, Linux aarch64) only. It does not run on Jetson or imply
IQ9075 compatibility. CSI capture remains outside this validation; use the
existing demo/video input. Scores are not probabilities and the initial 0.7
threshold is uncalibrated. Validate a factory threshold before production use.

## Package and identity

`tools/package_anomalyvfm.py DIRECTORY` verifies the two evaluated source hashes
and writes a deterministic manifest and schema-v2 model-store pointer.
Pointer: `anomalyvfm/ventuno-q-v16`.
Manifest digest: `sha256:31fcfef6afbc0f1f7b6f970e2d3da149cdc11102571e44a043e1bafa10ecb69b`.
The signed CONFIG_APPLY digest pins manifest.json; it pins compiled_ctx.onnx and
compiled_ctx_qnn.bin. Upload the three artifacts to the emitted immutable prefix
before publishing `pointers/anomalyvfm/ventuno-q-v16.json` in the model bucket.
Never overwrite a published version with different bytes.

## Runtime prerequisite / initial enrollment

Install this Agent plus ORT 1.30.0, onnxruntime_qnn 2.6.0, onnx, NumPy and Pillow
in a dedicated runtime with access to the existing Agent/GStreamer dependencies.
Do not change the experimental benchmark environment or the VisualAD ORT runtime.
The new backend fails closed on mismatched versions, identity, shapes, external
context references or CPU fallback. Register the QNN EP once; keep one session.

Provision the following base configuration on supported devices:

```
NUVION_ZSAD_BACKEND=anomalyvfm_qnn
NUVION_ZERO_SHOT_ENABLED=true
NUVION_MODEL_POINTER=anomalyvfm/ventuno-q-v16
NUVION_MODEL_DIGEST=sha256:31fcfef6afbc0f1f7b6f970e2d3da149cdc11102571e44a043e1bafa10ecb69b
NUVION_ANOMALYVFM_STORE=/var/lib/nuv-agent/models/anomalyvfm
NUVION_ANOMALYVFM_STATE_DIR=/var/lib/nuv-agent/anomalyvfm
```

Launch `python -s -m nuvion_app.runtime.anomalyvfm_start`, with absolute
NUV_AGENT_CONFIG, NUVION_SETTINGS_STATE_DIR and NUVION_COMMAND_INBOX_PATH.
This entrypoint runs the settings boot guard exactly once before config overlay
loading. Remove any duplicate settings_boot_guard ExecStartPre when selecting it.
The store/state paths must be writable by the service user. Existing registered
device credentials authenticate to `/devices/models/presign` with profile
`qnn-context`; no model-service key or signed download URL is persisted.

## Subsequent Fleet deployment

BE binds the exact pointer to `command.config.model.anomalyvfm_qnn.v1` and exposes
the catalog to platform admins. A fresh running device advertises this capability
only after a QNN-only profiled result. Choose eligible devices and a greater
configVersion; start Canary, verify loaded digest, then advance waves.

In DEMO mode this adapter permits only a model-only CONFIG_APPLY with
activation=RESTART. The demo input/profile stays unchanged. Mixed video, labels,
clip or collection changes remain rejected, as do unsupported/unknown modes.
The same fresh inference and exact digest checks gate commit after restart.

The preflight worker downloads to a temporary directory, checks sizes/SHA-256,
and atomically publishes a content-addressed directory. Pending downloads return
MODEL_DOWNLOADING / RETRY_EFFECT without changing active.env or restarting.
Failed downloads preserve the active model. After restart, the settings reconciler
requires fresh actual inference proof before committing; existing candidate/LKG
recovery handles failed startup. Disk replacement and stale inference withdraw
model proof. Downloads alone and runtime STARTING never count as success.

## Validation

Run the AnomalyVFM, VisualAD Fleet, inference-mode, settings-reconciler and pipeline
durable-safety/offline-effects tests. Hardware acceptance additionally requires
registered-device download through BE, QNN-only inference, fresh Fleet pointer
and digest, signed CONFIG_APPLY commit, and a failure case retaining LKG.
