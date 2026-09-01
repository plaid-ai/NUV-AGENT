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
Build the IQ9075 base bundle and package on a native arm64 Docker host. Both
builders execute in digest-pinned containers:
```bash
export COMPONENT_SHA=<full-stamped-source-sha>
export SOURCE_DATE_EPOCH=<source-commit-epoch>
packaging/release/build-agent-bundle.sh \
  0.1.119 dist/nuv-agent_0.1.119_iq9075-aarch64.agent-bundle.tar.gz
BOOTSTRAP_BUNDLE_PATH=dist/nuv-agent_0.1.119_iq9075-aarch64.agent-bundle.tar.gz \
  packaging/deb/build-deb.sh
```

This script:
- Embeds the exact hash-locked bundle and its SHA-256; `postinst` performs no
  network access, package resolution, `pip`, or local source build as root.
- Creates a versioned bootstrap slot under `/opt/nuv-agent/bootstrap/<version>`
  and atomically points `/opt/nuv-agent/current` at it. Package upgrades never
  clear the currently running venv in place.
- Safely extracts only normalized regular files/directories from the verified
  bundle into the versioned slot.
- Installs the systemd unit.
- Creates `/etc/nuv-agent/agent.env` if missing.
- Includes `depthai==2.32.0.0` from a hash-pinned binary-only wheel in the
  keyless build job. Ubuntu 24.04 CPython 3.12 is the exact target runtime.
- Installs the OAK/Movidius USB rule at
  `/usr/lib/udev/rules.d/80-movidius.rules`, reloads udev, and triggers only USB
  devices with vendor ID `03e7`.
- Persists the event outbox and Fleet command inbox under systemd-managed
  `/var/lib/nuv-agent/`; the service account intentionally has no home directory.
- Installs the root updater outside the swappable Agent slot under
  `/usr/lib/nuvion-updater`, with a `root:nuvion` `0660` systemd Unix socket.
  The `nuvion` account is explicitly removed from the root-equivalent Docker
  group. Docker/Triton profiles remain OTA fail-closed until their separate
  privileged runtime helper and product health adapter are installed.
- Demo mode uses the public MVTec dataset bucket and local cache at runtime.

Python requirement: Ubuntu 24.04 system CPython 3.12.

The maintainer script supports bounded install modes:

- `NUVION_INSTALL_PROFILE=base` (default) installs the exact IQ9075
  control-plane/media bundle without model backends.
- `full` and `runtime` fail closed until they receive separate hash locks,
  immutable bundles, and the root-owned Docker/Triton lifecycle helper.
- `NUVION_INSTALL_AUTOSTART=false` installs the package but leaves the service
  stopped and disabled until identity, camera, and credentials have been checked.

### IQ-9075 with OAK-D Lite

Build and install on the Ubuntu arm64 board:

```bash
BOOTSTRAP_BUNDLE_PATH=dist/nuv-agent_<version>_iq9075-aarch64.agent-bundle.tar.gz \
  ARCH=arm64 packaging/deb/build-deb.sh
packaging/dev/install-iq9075.sh dist/nuv-agent_<version>_arm64.deb
sudo packaging/dev/test-iq9075.sh --camera oak
```

The IQ-9075 installer accepts only Ubuntu arm64 on an IQ-9075/QCS9075 device
tree, provisions a root-owned development identity, applies safe
no-model/no-Fleet defaults, selects `NUVION_VIDEO_SOURCE=oak`, and deliberately
leaves the Agent stopped and disabled. Both installer and test default to
`--camera oak`; the option is shown above to make the intended hardware explicit.

The upstream Luxonis rule grants every local process read/write access with
`MODE=0666`. The packaged rule keeps its official `03e7` vendor match but limits
the match to a USB device node and applies `MODE=0660,GROUP=nuvion`. The systemd
service already runs as the `nuvion` user/group, so it can open and re-enumerate
the camera without root while unrelated local users cannot. `postinst` reloads
the rule and issues a filtered udev trigger; reconnect the camera once if host
firmware prevents the live node from being updated. The hardware test opens the
camera as the non-root `nuvion` service user and captures a bounded RGB frame.
On package removal, `postrm` reloads the remaining udev rules and retriggers the
same filtered devices so the package-specific group permission does not linger.

The default OAK stream is RGB `640x480@30`. An empty
`NUVION_DEPTHAI_DEVICE_ID` selects the first available OAK device. Set it to the
camera MXID when multiple devices are attached. Startup/read failure bounds are
controlled by:

```dotenv
NUVION_VIDEO_SOURCE=oak
NUVION_DEPTHAI_DEVICE_ID=
NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC=15
NUVION_DEPTHAI_READ_TIMEOUT_SEC=2
NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS=3
```

Existing UVC deployments remain supported. Select the old auto-discovered V4L2
path explicitly and use the matching release test:

```bash
packaging/dev/install-iq9075.sh dist/nuv-agent_<version>_arm64.deb --camera uvc
sudo packaging/dev/test-iq9075.sh --camera uvc
```

The UVC mode writes `NUVION_VIDEO_SOURCE=auto` and
`NUVION_CAMERA_PREFERENCE=usb`; OAK-specific values remain present but inactive.
Re-running the installer with the same camera option is idempotent: it reuses
the immutable versioned bootstrap slot, does not duplicate config keys, and preserves an
existing MXID and tuned DepthAI timeout values.

Use `sudo packaging/dev/provision-iq9075.sh <credentials.json> --consume` to
atomically install Dev device credentials without printing them. Add
`--synthetic-camera` only for an isolated control-plane test when neither a UVC
nor OAK camera is attached; remove `NUVION_GST_SOURCE` before the physical-camera
release gate. Once credentials and the selected camera test pass, enable the
service with `sudo systemctl enable --now nuv-agent.service`.

Runtime privilege boundary:
- The Agent service runs without `docker.sock` access and without a Docker
  systemd dependency.
- Docker/Triton lifecycle must move to a separately audited fixed-operation
  helper; arbitrary image, volume, path, or shell parameters are not accepted by
  the Agent updater.

## Release helpers
- `packaging/release/build-sdist.sh`: build source tarball and print SHA256.
- `packaging/release/build-agent-bundle.sh`: IQ9075 OTA용 self-contained immutable
  slot bundle을 만들며, symlink 없는 tar.gz, CPython 3.12/Linux aarch64
  hash-lock, pinned DepthAI wheel, stamped component SHA를 강제합니다. 직접 실행할
  때는 `SOURCE_DATE_EPOCH=<commit epoch> COMPONENT_SHA=<full commit SHA>
  packaging/release/build-agent-bundle.sh <version> <output>`을 사용합니다.
- `packaging/release/generate-release-bom.py`: v1 telemetry BOM 또는 exact product
  target, release sequence, minimum updater version, detached Ed25519 signature를
  가진 release-bom-v2를 생성합니다.
- `packaging/release/release-bom-v2.md`: BE/Agent 공통 canonical byte/signature contract.
- `packaging/release/normalize-sdist.py`: commit timestamp를 기준으로 tar/gzip metadata를
  정규화해 같은 release commit에서 byte-identical sdist를 재생성합니다.
- `packaging/release/update-homebrew-formula.sh`: inject URL/SHA/version into formula.
- `packaging/release/bootstrap-homebrew-tap.sh`: create and seed the tap repo.
- `packaging/release/promote-model-pointer.sh`: promote model channel pointer (`canary.json`/`prod.json`) in GCS.
- `packaging/apt/`: minimal `aptly` repo flow (GCS recommended).

## GitHub Actions release
Workflow: `.github/workflows/release-publish.yml`

Release workflow는 sdist/GitHub Release와 arm64 `.deb`/APT 배포를 유지하되,
OTA는 manual opt-in 시 별도의 IQ9075 전용 `agent-bundle`로 생성합니다. Dependency
build는 signing/GCP credential이 없는 keyless ARM job에서 수행되고, SHA256으로 묶인
artifact만 별도의 clean signing/publish job으로 전달됩니다. 현재 검증된
OTA tuple은 `IQ9075_DEV:iq9075_dev:QCS9075-EVK:aarch64` 하나뿐이며 production
SKU/revision은 추정하거나 wildcard로 발행하지 않습니다. 이 dev tuple의 runtime
contract는 Ubuntu 24.04 + CPython 3.12입니다.
Artifact digest는 자기참조를 피하기 위해 artifact 내부가 아니라 sidecar에 보관하며,
OTA activation은 publisher-signed release-bom-v2만 허용하며 signed
`AGENT_UPDATE` command의 expected BOM digest와 별도로 publisher signature를 검증합니다.
독립 OTA publisher는 sidecar를 `releases/<version>/`과
`releases/by-bom-sha256/<bom digest>/`에 exact artifact 및 detached signature와
함께 create-only로 저장하고 업로드 전후 byte-compare합니다. APT job은 OTA signing
key를 받지 않으며 OTA build/signing job은 APT GPG key를 받지 않습니다.
GitHub release asset은 기존 동일 이름을 덮어쓰지 않도록 설정되어 있으므로, 이미 발행된
version을 변경해야 할 때는 새 patch version을 발행해야 합니다.

Required secrets:
- `HOMEBREW_TAP_TOKEN` (PAT with push access to `plaid-ai/NUV-agent-homebrew`)
- `IQ9075_RELEASE_SIGNING_PRIVATE_KEY` (Ed25519 private publisher key)
- `IQ9075_RELEASE_SIGNING_KEY_ID`
- `IQ9075_RELEASE_PUBLIC_KEYRING_JSON` (`trustDomain=iq9075-dev`)

To host an APT repo, use a tool like `aptly` or `reprepro`, then publish the generated `.deb`.
