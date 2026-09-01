# Packaging

This directory contains packaging templates for Homebrew and Debian/Ubuntu.

## Homebrew (tap)
The pinned wheel set targets Apple Silicon macOS (`arm64`). Intel macOS developers
must use a source/virtualenv install until a separately pinned x86_64 resource set exists.

1. Create a tap repo (e.g., `plaid-ai/homebrew-NUV-agent-homebrew`).
2. Copy `packaging/homebrew/nuv-agent.rb` into `Formula/nuv-agent.rb`.
3. Replace `__URL__` and `__SHA256__` with the release tarball URL and SHA256.
4. Tag a release matching the formula version.

Recommended service env vars (already in the formula):
- `NUV_AGENT_CONFIG`
- `DYLD_LIBRARY_PATH`
- `GI_TYPELIB_PATH`
- `GST_PLUGIN_PATH`

Demo mode:
- `nuv-agent run --demo` downloads and caches a random MVTec category slideshow locally.
- Public dataset base URL defaults to `https://storage.googleapis.com/mvtec-dataset/mvtec-ad`.

## Debian/Ubuntu (.deb)
Build a package on the target architecture (e.g., Jetson ARM64):
```bash
cd NUV-agent/packaging/deb
./build-deb.sh
```

This script:
- Copies only `pyproject.toml`, `README.md`, and `nuvion_app/` into the package;
  workflow credentials, test environments, and repository metadata never enter the `.deb`.
- Creates a venv under `/opt/nuv-agent/venv`.
- Installs the Python package.
- Installs the systemd unit.
- Creates `/etc/nuv-agent/agent.env` if missing.
- Installs optional extras for runtime bootstrap (`zsad,triton`).
- Persists the event outbox and Fleet command inbox under systemd-managed
  `/var/lib/nuv-agent/`; the service account intentionally has no home directory.
- Demo mode uses the public MVTec dataset bucket and local cache at runtime.

Python requirement: 3.10+

Runtime bootstrap:
- `nuv-agent setup` / `nuv-agent run` now try to bootstrap Docker/Triton/model bundle automatically.
- systemd unit includes docker dependency and bootstrap preflight.

## Release helpers
- `packaging/release/build-sdist.sh`: build source tarball and print SHA256.
- `packaging/release/generate-release-bom.py`: actual artifact digest, component SHA,
  config schema, compatible platform profiles로 immutable sidecar BOM을 생성합니다.
- `packaging/release/normalize-sdist.py`: commit timestamp를 기준으로 tar/gzip metadata를
  정규화해 같은 release commit에서 byte-identical sdist를 재생성합니다.
- `packaging/release/update-homebrew-formula.sh`: inject URL/SHA/version into formula.
- `packaging/release/bootstrap-homebrew-tap.sh`: create and seed the tap repo.
- `packaging/release/promote-model-pointer.sh`: promote model channel pointer (`canary.json`/`prod.json`) in GCS.
- `packaging/apt/`: minimal `aptly` repo flow (GCS recommended).

## GitHub Actions release
Workflow: `.github/workflows/release-publish.yml`

Release workflow는 sdist와 arm64 `.deb` 각각에 대해 `*.release-bom.json`을 생성합니다.
Artifact digest는 자기참조를 피하기 위해 artifact 내부가 아니라 sidecar에 보관하며,
Agent updater는 signed `AGENT_UPDATE` command의 expected BOM digest와 대조해야 합니다.
APT publish는 sidecar를 `releases/<version>/`과
`releases/by-bom-sha256/<bom digest>/`에 모두 저장하고 업로드 후 byte-compare합니다.
GitHub release asset은 기존 동일 이름을 덮어쓰지 않도록 설정되어 있으므로, 이미 발행된
version을 변경해야 할 때는 새 patch version을 발행해야 합니다.

Required secrets:
- `HOMEBREW_TAP_TOKEN` (PAT with push access to `plaid-ai/NUV-agent-homebrew`)

To host an APT repo, use a tool like `aptly` or `reprepro`, then publish the generated `.deb`.
