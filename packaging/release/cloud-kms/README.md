# Company signing with Cloud KMS

NUV company signing keys are generated and used inside Google Cloud KMS. A Mac
GPG private key/passphrase is not required by the new signing tools. A Google
identity still needs valid authentication; CI uses GitHub OIDC and short-lived
Workload Identity Federation credentials instead of a service-account key file.

## Provisioned resources (2026-09-09)

Project `plaid-451114` (`472626352998`), location `asia-northeast3`.
All four keys use `EC_SIGN_ED25519`, protection level `SOFTWARE`, version `1`.
These are Cloud KMS keys, not Cloud HSM keys. No private material was exported.

| Environment | Key ring | Keys | CI service account | GitHub environment |
| --- | --- | --- | --- | --- |
| Development | `nuv-agent-dev` | `approval`, `ota` | `nuv-agent-sign-dev` | `nuv-signing-dev` |
| Production | `nuv-agent-prod` | `approval`, `ota` | `nuv-agent-sign-prod` | `nuv-signing-prod` |

Each service account has `roles/cloudkms.signer` and
`roles/cloudkms.publicKeyViewer` only on its own environment's keys. Separate
Workload Identity pools restrict GitHub's numeric repository/owner IDs, exact
environment subject, `main`, explicit dispatch, and the two reviewed workflows.
No service-account credential file was created. Company project owners retain
their existing administrative access; this change does not alter project roles.

The GitHub environments allow only the `main` branch, disable admin bypass and
contain no signing secrets. Explicit dispatch is restricted to the release admin
IDs in `release-security-policy.json`; main-branch protection was disabled by the repository owner during this
migration. Formal release settings verification still requires its configured
main-branch protection and will fail until that policy is satisfied. This matches the existing release
policy's zero environment-reviewer setting.

KMS `ADMIN_READ` and `DATA_READ` audit logging is enabled. `AsymmetricSign` is a
DATA_READ operation. `enable-cloud-kms-audit.py` preserves existing IAM role
bindings and uses the policy etag to reject concurrent changes.

The `approval.json` files pin exact versions, SPKI DER SHA-256 hashes and OpenPGP
fingerprints. `ota.json` pins exact versions, SPKI hashes and publisher key IDs.
All files in this directory contain public material only. Public key changes
require review; never replace the pin with a live key fetched during signing.

## Verification and activation boundary

Real KMS signing has passed for both environments: canonical release BOM,
detached OpenPGP evidence and Git annotated tag. The local tests also reject
wrong keys/versions/algorithms, CRC mismatches, corrupt signatures, modified
artifacts and unauthorized approval requests. The production OpenPGP public key
is added alongside the old company key so existing signed evidence remains valid.
The development approval key is deliberately absent from production tag trust.

This change does **not** mark `0.1.121` READY or resume the paused IQ9075 OTA test.
`candidate-publisher-v22` signs sequence 19 with the development OTA KMS key.
The IQ9075 development board must not trust the production OTA key. The active
IQ9075 keyring contains the new development key and the previous public key so
existing releases remain verifiable for rollback. Readiness still requires the
new candidate to be signed by the policy's active publisher key.

GitHub OIDC smoke verification passed for development (run 34311921321) and
production (run 34311923274), including BOM, detached evidence and signed Git tags.
The candidate and final IQ9075 OTA publishers use dedicated WIF providers pinned
to the exact approved workflow commit. Provision them only after merging and
creating the new signed immutable publisher tag. The final publisher keeps its
GCS credentials separate from KMS credentials; KMS never uses a stored private key.

Deploy the public keyring to BE and the updater, verify both old and new signed
BOMs, then remove `IQ9075_RELEASE_SIGNING_PRIVATE_KEY` from the two old signing
environments. Keep the old public key while any rollback release depends on it.
Changing trust does not authorize an OTA command or interrupt a device benchmark.
APT's separate GPG signing key is unchanged.

## Use without a GPG password

Use Python 3.11 (the pinned OpenPGP adapter uses PGPy 0.6). Install release tooling
in an isolated virtual environment:

```sh
python3.11 -m venv .venv-signing
.venv-signing/bin/python -m pip install --require-hashes \
  -r packaging/release/requirements-cloud-kms.txt
```

For a company-authorized developer workstation, authenticate with Google once
per organizational login session. Do not export a private key or store a Google
access token in a repository:

```sh
gcloud auth login swiftsjh@plaid.ai.kr
export NUVION_KMS_GCLOUD_ACCOUNT=swiftsjh@plaid.ai.kr
.venv-signing/bin/python packaging/release/kms-gpg.py sign \
  --config packaging/release/cloud-kms/prod/approval.json \
  --input /absolute/path/to/approved-evidence.json \
  --output /absolute/path/to/approved-evidence.json.asc
```

The signer refuses to replace an existing output. GnuPG can verify the result
offline using `prod/approval.asc`. A Google login authorizes a person; the
signature identifies the company key. Use the development configuration for
development artifacts. Personal Git signing keys are not shared company keys.

For approval tags after merge, dispatch `Sign company release approval tag with
Cloud KMS` with the exact current main SHA, new policy-approved tag, and approval
message. The workflow rejects reruns, stale main, an existing tag, an unauthorized
actor and version/publisher-policy mismatches. It signs and verifies through
GnuPG before pushing the new tag. It does not publish a release or approve OTA
acceptance; the normal release readiness checks still apply. A tag pushed with
`GITHUB_TOKEN` is not assumed to trigger another workflow automatically.

The BOM generator accepts these mutually exclusive remote signing options in
place of `--signing-private-key-env`:

```text
--signing-kms-key-version projects/.../cryptoKeyVersions/1
--signing-kms-public-key-sha256 <reviewed SHA-256 from ota.json>
--signing-key-id <reviewed keyId from ota.json>
```

The existing canonical BOM bytes, domain separator, Ed25519 signature envelope,
immutable file writes, and Agent verification remain the same. KMS validates the
request checksum; the tool checks the returned version/checksum and verifies the
signature against the pinned public key before writing output.

## Administration and rotation

`provision-cloud-kms.py` prints the resource plan by default; `--apply` creates
missing resources and adds per-key bindings. It does not rotate or destroy keys.
`enable-cloud-kms-audit.py --apply` enables signing audit events.

For rotation, create a new version, fetch its public material through the admin
path, review the new pin, distribute verifier trust, test signing/verification,
then switch the signer to the exact new version. Retain old public keys for
historical verification and rollback artifacts. Revoke signing access or disable
an obsolete private version only after checking those dependencies.

Official references: [KMS algorithms](https://docs.cloud.google.com/kms/docs/algorithms),
[signing API integrity checks](https://docs.cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign),
[GitHub federation](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines),
[KMS audit logging](https://docs.cloud.google.com/kms/docs/audit-logging).
