# NUVION Release BOM v2 signing contract

This document is the byte-level interoperability contract for NUVION Agent
release publishers, the BE release catalog, and the device-side updater. It is
not RFC 8785/JCS. Implement the rules below exactly.

Schema v1 remains readable for legacy runtime telemetry, but it is not a valid
OTA activation manifest. OTA requires a schema v2 BOM and a verified detached
Ed25519 signature.

## BOM v2 shape

The BOM is a strict JSON object with exactly these top-level members:

```text
schemaVersion       integer, exactly 2
bomId               safe release identifier
bomDigest           sha256:<64 lowercase hex>
releaseSequence     positive signed 64-bit integer
agentVersion        SemVer
componentSha        40- or 64-character lowercase hex
configSchema        positive integer encoded as a string
minUpdaterVersion   SemVer
targets             non-empty, unique, sorted array
artifact            exact artifact object
builtAt             timezone-qualified RFC 3339 timestamp
```

Each target contains exactly:

```json
{
  "productModel": "NUVION",
  "platformProfile": "rpi5_deepx_dx_m1",
  "hardwareRevision": "rpi5-dxm1-rev-a",
  "architecture": "aarch64"
}
```

There are no wildcard targets. `productModel`, `platformProfile`,
`hardwareRevision`, and `architecture` must all match the provisioned device
identity exactly. Target entries are sorted by that field tuple in this order:
`productModel`, `platformProfile`, `hardwareRevision`, `architecture`.

The artifact object contains exactly `name`, `kind`, `sha256`, and `sizeBytes`.
Schema v2 additionally permits the self-contained `agent-bundle` kind. Schema
v1's artifact-kind allowlist is unchanged. The current Linux updater accepts
`agent-bundle` files using `.tar`, `.tar.gz`, or `.tgz`; zstd bundles are not
accepted until the updater has a pinned zstd decoder.

## BOM digest

To compute `bomDigest`:

1. Remove only the top-level `bomDigest` member.
2. Serialize the remaining object with the canonical JSON rules below.
3. Compute SHA-256 over those bytes.
4. Encode it as `sha256:` followed by 64 lowercase hexadecimal characters.

## Canonical JSON

Canonical BOM bytes are produced as follows:

- parse strict UTF-8 JSON and reject duplicate object members, invalid UTF-8,
  `NaN`, `Infinity`, and `-Infinity`;
- recursively sort object member names in ascending Unicode code-point order;
- preserve array order;
- encode integers as base-10 JSON integers;
- emit JSON strings without Unicode normalization;
- use `,` and `:` separators with no surrounding whitespace;
- encode the result as UTF-8 without a BOM;
- do not append a newline.

The accepted schema constrains release identifiers and compatibility values to
ASCII, avoiding cross-runtime Unicode ordering ambiguity. Pretty-printed BOM
files may end with a newline, but publishers and verifiers reserialize the
parsed object; raw file bytes are never signed.

Equivalent Python settings are:

```python
json.dumps(
    bom,
    ensure_ascii=False,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
```

## Detached Ed25519 signature

The exact signing input is:

```text
ASCII("NUVION-RELEASE-BOM-V2") || 0x00 || canonical_full_bom
```

`canonical_full_bom` includes the validated `bomDigest` member. The domain
separation prefix bytes in hexadecimal are:

```text
4e5556494f4e2d52454c454153452d424f4d2d563200
```

Sign that byte sequence directly with Ed25519. Do not pre-hash it for the
Ed25519 operation. The fixture also provides SHA-256 of the signing input only
as a cross-language diagnostic.

The detached signature sidecar is a strict JSON object with exactly:

```json
{
  "schemaVersion": 1,
  "keyId": "release-prod-2026",
  "algorithm": "Ed25519",
  "signature": "<RFC 4648 standard base64 with canonical padding>"
}
```

The decoded signature must be exactly 64 bytes. URL-safe base64, omitted
padding, whitespace, unknown members, and any algorithm other than the exact
case-sensitive string `Ed25519` are rejected. `keyId` selects a pinned release
publisher public key; it is not a network key-discovery URL.

The signature authenticates the full BOM, which in turn binds the artifact
name, SHA-256 digest, byte size, release sequence, minimum updater version, and
every exact target. A valid signature never bypasses compatibility or
anti-downgrade checks.

## Cross-language test vector

The normative test vector is stored at
`tests/runtime/fixtures/release-bom-v2-ed25519.json`.
It contains:

- a complete v2 BOM;
- one raw 32-byte Ed25519 public key encoded with standard base64;
- the detached signature envelope;
- the expected domain prefix, canonical BOM SHA-256, and signing-input
  SHA-256.

It intentionally contains no private key. A compatible BE or updater
implementation must reproduce both diagnostic digests and verify the detached
signature with the supplied public key.

## Publisher CLI

`generate-release-bom.py --schema-version 2` writes both the BOM and detached
signature sidecar. Provide private material through one of these mechanisms:

```text
--signing-private-key /protected/path/release-key.pem
--signing-private-key-env NUVION_RELEASE_SIGNING_KEY
```

The environment value may contain PEM or canonical base64 of raw/DER Ed25519
private-key material. Never pass private material itself as a command-line
argument, write it into a BOM/signature sidecar, or log it.
