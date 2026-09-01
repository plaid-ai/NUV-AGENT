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
- On arm64, installs `depthai==2.32.0.0` into the Agent venv from a
  hash-pinned, binary-only requirements file. CPython 3.10-3.14
  `manylinux_2_28_aarch64` wheels are allowlisted; an unexpected wheel or sdist
  fails installation instead of running unverified code. Ubuntu 24.04 uses the
  pinned CPython 3.12 wheel.
- Installs the OAK/Movidius USB rule at
  `/usr/lib/udev/rules.d/80-movidius.rules`, reloads udev, and triggers only USB
  devices with vendor ID `03e7`.
- Persists the event outbox and Fleet command inbox under systemd-managed
  `/var/lib/nuv-agent/`; the service account intentionally has no home directory.
- Demo mode uses the public MVTec dataset bucket and local cache at runtime.

Python requirement: 3.10+

The maintainer script supports bounded install modes:

- `NUVION_INSTALL_PROFILE=full` installs `zsad,triton` extras (default).
- `NUVION_INSTALL_PROFILE=runtime` installs only the Triton client extra.
- `NUVION_INSTALL_PROFILE=base` installs the control-plane and media runtime without
  model backends. This is the initial IQ-9075 development-board profile.
- `NUVION_INSTALL_AUTOSTART=false` installs the package but leaves the service
  stopped and disabled until identity, camera, and credentials have been checked.

### IQ-9075 with OAK-D Lite

Build and install on the Ubuntu arm64 board:

```bash
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
Re-running the installer with the same camera option is idempotent: it rebuilds
the venv from the pinned inputs, does not duplicate config keys, and preserves an
existing MXID and tuned DepthAI timeout values.

Use `sudo packaging/dev/provision-iq9075.sh <credentials.json> --consume` to
atomically install Dev device credentials without printing them. Add
`--synthetic-camera` only for an isolated control-plane test when neither a UVC
nor OAK camera is attached; remove `NUVION_GST_SOURCE` before the physical-camera
release gate. Once credentials and the selected camera test pass, enable the
service with `sudo systemctl enable --now nuv-agent.service`.

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
