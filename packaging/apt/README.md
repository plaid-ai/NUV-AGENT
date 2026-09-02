# APT repo (aptly)

This directory provides a minimal `aptly` flow to host a private APT repo.

## Requirements
- `aptly`
- `gpg` (for signing)
  - Make sure a default GPG key exists (`gpg --list-keys`).

## Publish to GCS (recommended)
This flow syncs the published repo to `gs://apt.plaidai.io` and serves it via `https://apt.plaidai.io`.

Release CI sets `APT_PREVIOUS_DEB_PATH` to the highest lower Agent version from
the independently verified current `InRelease`. Because the aptly database is
ephemeral, this explicitly keeps the current and previous packages in the new
signed `Packages` index. Retaining an unindexed pool object is not considered a
rollback path.

Immutable pool/BOM/bundle objects and OTA reservation/promotion markers use
Cloud Storage `ifGenerationMatch=0` create-only CAS, followed by exact remote
byte comparison. `gsutil cp -n` is not used as a concurrency primitive.

```bash
./publish-gcs.sh /path/to/nuv-agent_0.1.0_arm64.deb
```

Publisher-authenticated OTA-only 배포는 첫 번째 호환 인자와 네 번째 exact
artifact 인자에 같은 `agent-bundle`을 전달합니다. APT `.deb` build/publish는
별도 keyless artifact job과 clean signer job에서 수행됩니다.

```bash
RELEASE_KEYRING_PATH=/secure/release-keyring.json \
RELEASE_TRUST_DOMAIN=iq9075-dev \
SKIP_APT_PUBLISH=true \
VERSION=0.1.120 \
./publish-gcs.sh \
  /path/to/nuv-agent_0.1.120_iq9075-aarch64.agent-bundle.tar.gz \
  /path/to/release-bom.json \
  /path/to/release-bom.json.sig \
  /path/to/nuv-agent_0.1.120_iq9075-aarch64.agent-bundle.tar.gz
```

세 release 파일은 동일한
`releases/by-bom-sha256/<digest>/` 디렉터리에 저장되며, 기존 remote byte와
다르면 publish를 거부합니다.

Requirements:
- `gcloud` + `gsutil`
- A GCS bucket named `apt.plaidai.io`
- A public HTTPS endpoint for the bucket (Cloud CDN + HTTPS Load Balancer recommended)

The repo is published under `.aptly/public` locally and synced to GCS.

## GPG key (important)
`aptly` signs the repo with your default GPG key. Make sure the public key you serve
matches the signing key, otherwise clients will see `NO_PUBKEY`.

Tips:
- Use a dedicated signing key for the APT repo.
- Set `GPG_KEY_ID` when running `publish.sh`/`publish-gcs.sh` to export the correct key.
- The publish scripts export `public.gpg` into `.aptly/public/public.gpg` automatically.
- The publish scripts also copy `install-apt.sh` into `.aptly/public/install-apt.sh`.

## GitHub Actions publish (optional)
If you want to publish automatically on tag push, use the `apt-publish` job in
`.github/workflows/release-publish.yml`. It expects an arm64 runner.

Required GitHub secrets:
- `APT_GPG_PRIVATE_KEY`: ASCII-armored private key (export with `gpg --export-secret-keys --armor <KEY_ID>`)
- `APT_GPG_PASSPHRASE`: passphrase for the signing key
- `GCP_SA_KEY`: GCP service account JSON with write access to the bucket
- `GCP_PROJECT_ID`: GCP project ID
- `IQ9075_RELEASE_SIGNING_PRIVATE_KEY`: Ed25519 publisher private key material
- Publisher key ID is protected-main metadata in `release-security-policy.json`
- `packaging/release/trusted-release-keyrings/iq9075-dev.json`: protected-main
  public verification keyring, byte-identical to the board keyring (not a secret)

Runner requirement:

- Release jobs are fixed to GitHub-hosted `ubuntu-24.04-arm`. The required
  `agent-release-gate` depends on a secret-zero arm64 prerequisite job that
  verifies native architecture and the exact hashed bundle lock. If that label
  is unavailable to the repository, release remains blocked. Do not silently
  move credentials to a long-lived self-hosted runner; provision a separately
  reviewed ephemeral trusted-runner design first.

Runner note:
- Default is `ubuntu-24.04-arm`. If you don't have access, change the job to `self-hosted`
  and attach an arm64 runner (e.g., Jetson or Graviton).

Provisioning (GCP):
- `packaging/apt/gcp/setup-apt-hosting.sh`
- `packaging/apt/gcp/README.md`

Client install example (arm64 only):
```bash
sudo install -d /etc/apt/keyrings
curl -fsSL https://apt.plaidai.io/public.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/plaidai.gpg
echo \"deb [signed-by=/etc/apt/keyrings/plaidai.gpg arch=arm64] https://apt.plaidai.io stable main\" | sudo tee /etc/apt/sources.list.d/plaidai.list
sudo apt update
sudo apt install nuv-agent
```

One-line install (same steps as above):
```bash
curl -fsSL https://apt.plaidai.io/install-apt.sh | bash
```

## Local publish
```bash
./publish.sh /path/to/nuv-agent_0.1.0_arm64.deb
```
The repo is published under `.aptly/public`. You can serve it via nginx.

## Kubernetes hosting (optional)
See `packaging/apt/k8s/README.md` for a minimal Nginx deployment that serves the repo from a PVC.
