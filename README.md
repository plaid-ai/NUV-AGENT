# Nuvion Agent (Device Software)

NUV-agent는 온디바이스 AI 장치에 설치하는 소프트웨어입니다. 도커 기반으로 제작되어 이식이 쉽고,
USB UVC 웹캠과 Luxonis OAK/DepthAI 카메라 스트림을 Nuvion-be 스프링 서버를 통해 송출할 수 있습니다. 동시에 제로샷 AI 모델로
이상 감지를 수행하여 공장에서 이상 감지와 생산량 추적을 할 수 있도록 해주는 프로그램입니다. 감지 결과는
영상 위에 실시간으로 오버레이 됩니다.

## Structure
- `nuvion_app/inference`: GStreamer RTP streaming + zero-shot anomaly detection

## Install (brew/apt)
Packaging templates and build scripts live in `packaging/`. See `packaging/README.md`.

Homebrew (Apple Silicon):
```bash
brew tap plaid-ai/NUV-agent-homebrew
brew install nuv-agent
```
Note: Homebrew install includes Zero-shot (torch/transformers/Pillow) deps. The download is large.
`nuv-agent setup`/`nuv-agent run` automatically bootstrap runtime dependencies (Homebrew, Docker/Colima, Triton) when needed.

APT (Jetson/Ubuntu, arm64):
```bash
sudo install -d /etc/apt/keyrings
curl -fsSL https://apt.plaidai.io/public.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/plaidai.gpg
echo "deb [signed-by=/etc/apt/keyrings/plaidai.gpg arch=arm64] https://apt.plaidai.io stable main" | sudo tee /etc/apt/sources.list.d/plaidai.list
sudo apt update
sudo apt install nuv-agent
```
One-line install:
```bash
curl -fsSL https://apt.plaidai.io/install-apt.sh | bash
```
`nuv-agent setup`/`nuv-agent run` automatically bootstrap Docker/Triton/model bundle when needed.

Config health check / migration:
```bash
# 기존 설정 파일을 최신 schema로 자동 보정하고 검증
nuv-agent doctor --fix
```

## Quick start (dev)
1) Copy `.env.example` to `.env` and fill in credentials.
2) Run locally:
   ```bash
   pip install -e .
   python -m nuvion_app.cli run
   ```

Python requirement: 3.10+

## Pull model bundle (server presign 권장)
운영 기본 경로는 `NUV-BE` presign API를 통해 signed URL을 받아 모델 번들을 내려받는 방식입니다.
```bash
# runtime: text_features + Triton model_repository (권장)
nuv-agent pull-model \
  --source server \
  --server-base-url https://api.nuvion-dev.plaidlabs.ai \
  --pointer anomalyclip/prod \
  --local-dir ~/.cache/nuvion/models/anomalyclip-current \
  --profile runtime
```

- `--access-token`을 직접 전달하거나, 생략 시 `NUVION_DEVICE_USERNAME/NUVION_DEVICE_PASSWORD`로 `/auth/login` 후 presign 호출
- 다운로드 후 각 artifact에 대해 `sha256` 무결성 검증 수행
- signed URL 다운로드 중 400/401/403 오류가 발생하면, presign URL을 자동 재발급 받아 이어서 재시도
- 결과 메타데이터: `metadata/downloaded_from_server.json`

## Pull model bundle (GCS fallback)
Profiles:
- `runtime`: Triton + text features 실행에 필요한 파일만 다운로드
- `light`: text features/metadata 중심의 경량 다운로드
- `full`: 추가 분석/검증 파일까지 포함해서 다운로드

포인터 호환:
- `artifacts` 값을 문자열로 주는 기존 포맷과
- `artifacts.<key>.path`를 사용하는 v2 포맷을 모두 지원합니다.

기본값:
- `NUVION_MODEL_POINTER=anomalyclip/prod`
- `NUVION_MODEL_PRESIGN_TTL_SECONDS=300`
- `NUVION_MODEL_SERVER_BASE_URL=https://api.nuvion-dev.plaidlabs.ai`
- `NUVION_MODEL_PROFILE=runtime`
- `NUVION_MODEL_LOCAL_DIR=~/.cache/nuvion/models/anomalyclip-current`

채널 포인터 예시:
- Canary: `gs://nuv-model/pointers/anomalyclip/canary.json`
- Prod: `gs://nuv-model/pointers/anomalyclip/prod.json`

## FSD-style 모델 롤아웃 (권장)
모델 파일은 버전 디렉토리(`v0001`, `v0002`, ...)에 immutable하게 두고, 장치는 channel pointer만 바라보게 운영합니다.

1. 새 버전 업로드: `gs://nuv-model/nuvion/anomalyclip/v0002/...`
2. Canary 포인터 승격:
   ```bash
   packaging/release/promote-model-pointer.sh \
     --source-pointer gs://nuv-model/nuvion/anomalyclip/v0002/pointer.json \
     --target-pointer gs://nuv-model/pointers/anomalyclip/canary.json
   ```
3. Prod 포인터 승격:
   ```bash
   packaging/release/promote-model-pointer.sh \
     --source-pointer gs://nuv-model/nuvion/anomalyclip/v0002/pointer.json \
     --target-pointer gs://nuv-model/pointers/anomalyclip/prod.json
   ```

## macOS dev setup (Homebrew)
Recommended for local runs on Apple Silicon.
```bash
brew install python@3.14 gobject-introspection pygobject3 gstreamer \
  gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav

/opt/homebrew/opt/python@3.14/bin/python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e .

export DYLD_LIBRARY_PATH=/opt/homebrew/lib
export GI_TYPELIB_PATH=/opt/homebrew/lib/girepository-1.0
export GST_PLUGIN_PATH=/opt/homebrew/lib/gstreamer-1.0

python -m nuvion_app.cli run
```
Note: `pygobject3` is tied to Homebrew’s Python. Using `python@3.14` and `--system-site-packages`
ensures the `gi` module is visible inside the venv.
Note: On macOS the default camera source is `avfvideosrc` (AVFoundation). Linux defaults to `/dev/video0`.

## Quick start (docker)
Build/run with docker-compose from `nuvion_app/`:
```bash
cd nuvion_app
docker compose up --build
```

Optional build args (in `nuvion_app/inference/Dockerfile.inference`):
- `INSTALL_ZSAD_DEPS=true`
- `INSTALL_TRITON_DEPS=true`

## Exhibition demo mode
기본 런타임은 카메라 입력을 사용하고, `--demo`를 주면 데모 입력으로 전환됩니다.

일반 모드:
```bash
nuv-agent run
```

MVTec 슬라이드쇼 데모 모드:
```bash
nuv-agent run --demo
```

이 경우 public demo bucket에서 `screw`, `metal_nut`, `cable`, `capsule` 중 하나를 랜덤으로 골라
`train/good` 이미지를 로컬 캐시에 내려받은 뒤 슬라이드쇼처럼 반복 재생합니다.

정책:
- `--demo`는 항상 MVTec 슬라이드쇼 입력을 사용합니다.
- MVTec 기본 공개 bucket:
  - `NUVION_DEMO_MVTEC_BASE_URL=https://storage.googleapis.com/mvtec-dataset/mvtec-ad`
- 기본 카테고리:
  - `screw,metal_nut,cable,capsule`
- 로컬 캐시 기본 경로:
  - `~/.cache/nuvion/demo/mvtec`
- MVTec demo source 설정이 잘못되면 즉시 실패(fail-fast)합니다.
- 데모 슬라이드쇼는 EOS 시 자동으로 처음부터 재생됩니다(`NUVION_DEMO_LOOP=true`).
- anomaly 이벤트 message에는 `[DEMO]` prefix가 붙습니다(기본 `NUVION_DEMO_TAG=[DEMO]`).

기본 샘플 영상 출처(CC BY 3.0):
- Gigaset Smartphone Production IV Quality Inspection (Wikimedia Commons)
  - https://commons.wikimedia.org/wiki/File:Gigaset_Smartphone_Production_IV_Quality_Inspection.webm

전시장용 대체 영상(직접 경로 지정 권장):
- Assembly line (CC BY 4.0)
  - https://commons.wikimedia.org/wiki/File:Assembly_line.webm
- Animal feed pellet production line (CC BY-SA 4.0)
  - https://commons.wikimedia.org/wiki/File:Animal_feed_pellet_production_line.webm

## Setup UI (device)
If a display is available, run:
```bash
nuv-agent setup
```
This starts a local setup UI at `http://127.0.0.1:8088` (override with `--host/--port`).
The setup UI includes an **Auto Provision** section: login with an owner/admin account to create
device credentials automatically (your account credentials are not stored on the device).
It also includes:
- **Inference Mode** quick selector (`Triton | SigLIP | SigLIP+MPS | None`)
- **Conditional settings view** (only backend-relevant fields are shown)
- **Preflight Check** button (server login / triton health / camera source or demo video source)
- **Environment override warning** when shell env values override file values

For headless devices:
```bash
nuv-agent setup --qr
```
This prints a pairing URL/QR code. After approval in the web console, the device credentials
are saved to the config file.

Default config path:
- macOS (Homebrew): `/opt/homebrew/etc/nuv-agent/agent.env` (or `/usr/local/etc/nuv-agent/agent.env`)
- Linux: `/etc/nuv-agent/agent.env`

For dev, `.env` in the repo is used automatically.

## First-time user flow (권장)
설치 직후에는 아래 순서만 실행하면 됩니다.
1. `nuv-agent setup`
2. `nuv-agent run`

자동 처리되는 항목:
- 모델 번들 pull (`source=server`, `profile=runtime|full`)
- macOS: Homebrew(미설치 시) → Docker CLI/Colima(미설치 시) → Triton 컨테이너 준비
- Jetson/Linux: Docker(미설치 시) 점검/설치 시도 → Triton 컨테이너 준비

정책:
- Docker Desktop이 이미 실행 중이면 우선 사용
- Docker Desktop 데몬이 없거나 불능이면 Colima 폴백
- bootstrap 실패 시 방송/시그널링은 유지하고, 추론 backend만 `none`으로 강등

## Service
- Linux: use `packaging/systemd/nuv-agent.service` and `systemctl enable --now nuv-agent`.
- macOS: use Homebrew service definition in `packaging/homebrew/nuv-agent.rb`.

## Device configuration
- `NUVION_VIDEO_SOURCE`: camera source. Luxonis OAK는 `oak`/`oak:<MXID>`, Linux UVC는 `/dev/video0`, macOS는 `avf` 또는 `avf:<index>`, Raspberry Pi는 `rpi`
- `NUVION_DEPTHAI_DEVICE_ID`: 여러 OAK 장치가 있을 때 선택할 MXID. 하나면 비워둔다.
- `NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC`, `NUVION_DEPTHAI_READ_TIMEOUT_SEC`, `NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS`: OAK startup/read fail-closed 경계
- `NUVION_DEMO_MODE`: 데모 모드 활성화 (`true|false`)
- `NUVION_DEMO_MVTEC_BASE_URL`: MVTec archive base URL
- `NUVION_DEMO_MVTEC_CATEGORIES`: 랜덤 선택 후보 category CSV
- `NUVION_DEMO_MVTEC_CACHE_DIR`: archive/extract/slides 캐시 경로
- `NUVION_DEMO_IMAGE_DURATION_SEC`: 이미지 1장당 재생 시간(초)
- `NUVION_DEMO_LOOP`: 데모 영상 EOS 시 반복 재생 여부 (`true|false`, 기본 `true`)
- `NUVION_DEMO_TAG`: 데모 이벤트 메시지 prefix (기본 `[DEMO]`)
- `NUVION_DEMO_VIDEO_FALLBACK_PATHS`: 추가 fallback 경로 CSV (예: `/data/demo1.webm,/data/demo2.mp4`)
- `NUVION_ANOMALY_LABELS`: comma-separated labels treated as anomalies
- `NUVION_PRODUCTION_LABELS`: comma-separated labels counted for production
- `NUVION_DEVICE_STATE_INTERVAL_SEC`: `/app/device/state` heartbeat 주기(초, 기본 `30`)
- `NUVION_EVENT_OUTBOX_PATH`: anomaly/production/state/connectivity SQLite outbox 경로 (systemd 기본 `/var/lib/nuv-agent/events.sqlite3`)
- `NUVION_EVENT_REPLAY_INTERVAL_SEC`: terminal 확인 전 event 재전송 주기(초, 기본 `5`)
- `NUVION_EVENT_OUTBOX_MAX_ROWS`: replay 대기 event 최대 행 수(기본 `10000`)
- `NUVION_EVENT_OUTBOX_MAX_BYTES`: payload와 destination/metadata를 포함한 logical record 한도(기본 `67108864`, 64 MiB)
- `NUVION_EVENT_CRITICAL_SAFETY_MAX_BYTES`: 정상 quota가 가득 찬 순간의 CRITICAL 관측 1건을 crash-safe하게 보존하는 별도 SQLite safety slot byte 한도(기본 `67108864`). 정상 outbox보다 작게 설정할 수 없으며, outbox+slot의 총 hard cap은 기본 128 MiB이다.
- `NUVION_EVENT_OUTBOX_MAX_AGE_SECONDS`: STATE/METRIC 최대 보존 기간(기본 `2592000`, CRITICAL에는 미적용)
- `NUVION_EVENT_DLQ_MAX_ROWS`: permanent rejection DLQ 보존 행 수(기본 `10000`, 낮은 priority가 높은 priority를 제거하지 않음)
- `NUVION_EVENT_DLQ_MAX_BYTES`: payload와 rejection metadata를 포함한 DLQ logical record 한도(기본 `67108864`)

CRITICAL event는 server application ACK(`ACCEPTED`/`DUPLICATE`)까지 outbox에 남는다. 정상 quota가 부족하면 같은 eventId와 canonical payload를 단일 `critical_safety_slot`에 먼저 commit한 뒤 inspection을 operator-stop하며, restart 후에도 그 slot을 복원해 정상 outbox로 옮긴다. 정상 outbox로 옮겨진 뒤에도 명시적인 operator recovery 전에는 inspection을 자동 재개하지 않는다. BE가 아직 application ACK를 보내지 않는 STATE/METRIC은 WebSocket transport send 완료를 terminal success로 간주하며, send 실패 시 SQLite에 남아 reconnect 후 replay된다. Permanent rejection을 DLQ에 full payload로 수용할 수 없으면 원본 payload는 `DLQ_BLOCKED` 상태로 보존되고 replay에서 제외되며 `DurableEventOutbox.blocked_events()`로 조회할 수 있다.
- `NUVION_CONNECTIVITY_ENABLED`: `/app/device/connectivity` 보고 활성화 (`true|false`)
- `NUVION_CONNECTIVITY_INTERVAL_SEC`: 연결 품질 샘플링 주기(초, 기본 `10`)
- `NUVION_CONNECTIVITY_MIN_SEND_INTERVAL_SEC`: 전이 이벤트 최소 전송 간격(초, 기본 `30`)
- `NUVION_CONNECTIVITY_POOR_RSSI_DBM`: POOR RSSI 임계값(dBm, 기본 `-80`)
- `NUVION_CONNECTIVITY_POOR_PACKET_LOSS_PCT`: POOR packet loss 임계값(%, 기본 `8`)
- `NUVION_CONNECTIVITY_POOR_RTT_MS`: POOR RTT 임계값(ms, 기본 `250`)
- `NUVION_CONNECTIVITY_TARGET_HOST`: ping 대상 호스트 override (기본: `NUVION_SERVER_BASE_URL` host)
- `NUVION_WIFI_INTERFACE`: Linux/Jetson에서 `iw` RSSI 수집용 인터페이스 (미지정 시 auto detect)
- `NUVION_ZERO_SHOT_ENABLED`: enable optional zero-shot anomaly detection (requires model deps)
- `NUVION_ZSAD_BACKEND`: `triton|siglip|mps|none` (`mps`는 `siglip + NUVION_ZERO_SHOT_DEVICE=mps` alias)
- `NUVION_ZERO_SHOT_MODEL`: 기본 ZSAD 모델 (`google/siglip2-base-patch16-224`)
- `NUVION_ZERO_SHOT_DEVICE`: SigLIP 디바이스 우선순위 (`auto|mps|cuda|cpu`, 기본 `auto`)
- `NUVION_MODEL_POINTER`: 서버가 해석할 model pointer (`anomalyclip/prod`)
- `NUVION_MODEL_PRESIGN_TTL_SECONDS`: server presign 요청 TTL
- `NUVION_MODEL_SERVER_BASE_URL`: server presign API base URL
- `NUVION_MODEL_SERVER_ACCESS_TOKEN`: 사전 발급 토큰(선택). 미지정 시 setup에서 저장된 device credential로 로그인 후 다운로드
- `NUVION_MODEL_PROFILE`: pull-model 프로필 (`runtime|light|full`)
- `NUVION_MODEL_DIR`: pull-model 기본 저장 루트
- `NUVION_CONFIG_SCHEMA_VERSION`: config schema 버전 (현재 `12`, `doctor --fix`로 자동 보정)
- `NUVION_RUNTIME_BOOTSTRAP_ENABLED`: setup/run bootstrap 전체 on/off
- `NUVION_HOMEBREW_AUTOINSTALL`: macOS Homebrew 자동 설치 허용
- `NUVION_DOCKER_AUTOINSTALL`: Docker/Colima(또는 docker.io) 자동 설치 허용
- `NUVION_DOCKER_AUTOSTART`: Docker daemon 자동 기동 허용
- `NUVION_DOCKER_DESKTOP_TIMEOUT_SEC`: macOS Docker Desktop daemon 준비 대기 시간(초)
- `NUVION_TRITON_AUTOSTART`: Triton 컨테이너 자동 기동 허용
- `NUVION_TRITON_AUTOSTART_ONLY_LOCAL`: local Triton URL에서만 자동 기동
- `NUVION_TRITON_AUTOSTOP_ON_EXIT`: agent 종료 시 자동 기동한 Triton 컨테이너 자동 종료 (기본 `false`)
- `NUVION_MODEL_AUTO_PULL_ON_SETUP`: setup 단계에서 model auto pull
- `NUVION_MODEL_AUTO_PULL_ON_RUN`: run 단계에서 model auto pull
- `NUVION_BOOTSTRAP_MAX_RETRIES`: bootstrap 재시도 횟수
- `NUVION_BOOTSTRAP_BACKOFF_SEC`: bootstrap 지수 백오프 시작값(초)
- `NUVION_TRITON_CONTAINER_NAME`: 자동 관리 Triton 컨테이너 이름
- `NUVION_TRITON_IMAGE`: 자동 기동할 Triton 이미지
- `NUVION_TRITON_MAC_PROFILE`: macOS auto pull profile (기본 `full`)
- `NUVION_TRITON_JETSON_PROFILE`: Jetson/Linux auto pull profile (기본 `runtime`)
- `NUVION_AGENT_ERROR_MAX_RETRIES`: 서버 agent error(`retryable=true`) 수신 시 자동 재시도 최대 횟수 (기본 `3`)
- `NUVION_AGENT_ERROR_BACKOFF_BASE_SEC`: 첫 재시도 대기 시간(초), 이후 지수 백오프 (기본 `1.0`)
- `NUVION_AGENT_ERROR_BACKOFF_MAX_SEC`: 재시도 최대 대기 시간(초) (기본 `15.0`)
- `NUVION_CLIP_EVENT_ACK_WAIT_SEC`: anomaly 저장 ACK 후 clip finalize를 시도하기 위한 최대 대기(초, 기본 `60`)
- `NUVION_CLIP_STATUS_MAX_RETRIES`: clip finalize API 최대 시도 횟수(기본 `5`)

macOS note: use `NUVION_VIDEO_SOURCE=avf` (default camera) or `avf:<index>` to select a camera.

### Agent WebSocket error queue
- Agent는 STOMP에서 `/user/queue/agent.error`를 구독합니다.
- `retryable=true` 에러는 일반 uplink payload를 백오프로 재전송합니다.
- `401/403` 같은 non-retryable 권한 오류는 uplink를 차단하고 로그에 원인(`code`, `path`, `detail`)을 남깁니다.

### Critical event delivery
- anomaly/production은 `eventId`와 `occurredAt`을 붙여 SQLite에 먼저 저장한 뒤 전송합니다.
- Agent는 `/user/queue/event.ack`의 `ACCEPTED|DUPLICATE` ACK를 받은 경우에만 outbox row를 삭제합니다.
- `REJECTED` + `retryable=false` ACK 또는 eventId가 포함된 영구 400/422 error는 원본 payload와 시도 횟수를 DLQ로 이동합니다.
- 재연결 또는 ACK timeout 시 동일한 `eventId`로 replay하므로 서버는 `eventId` 기준 idempotency를 제공해야 합니다.
- Pending quota 초과 시 기존 event를 밀어내지 않고 신규 event를 DLQ로 격리하는 reject-new 정책을 사용합니다.
- state heartbeat에는 legacy `status`와 함께 `runtimeStatus`, `inspectionStatus`, `connectivityStatus`, `agentVersion`, `componentSha`, `configSchema`, `modelPointer`, `modelVersion`이 포함됩니다.
- Fleet command를 사용할 때는 `NUVION_FLEET_COMMAND_ENABLED=true`, `NUVION_SPACE_ID`,
  `NUVION_FLEET_COMMAND_KEYRING_PATH`를 provision합니다. Agent는
  `/user/queue/fleet.command` wake-up 후 BE journal을 pull하고, device/space-bound Ed25519 JWS를
  검증한 뒤 SQLite inbox와 command별 reconcile job/history/lease/checkpoint를 먼저 기록합니다.
  실제 named encoder와 transactional settings store가 준비된 경우에만 `STREAM_POLICY`와
  `CONFIG_APPLY` effect/capability를 등록하며, `AGENT_UPDATE`는 구현 전까지 광고하지 않습니다.
  새 desired state는 적용 전의 이전 command(`WAITING_RESTART` 포함)를
  `FAILED/SUPERSEDED`로 terminal 처리하고, bounded coordinator가 transaction 밖에서 named
  `x264enc`를 변경한 뒤 readback과 reported state를 포함해 `SUCCEEDED`를 기록합니다. macOS
  개발 keyring(`macos-dev`)과 생산 keyring(`production`)은 서로 호환되지 않습니다.

Command observed state는 lifecycle ACK와 별개입니다. Agent는 SQLite outbox에 먼저 저장한 뒤
`/app/device/command.observed`로 정확히 `observationId`, `commandId`, `revision`, `observedAt`,
`reportedState`만 전송합니다. `/user/queue/fleet.command.observed.ack`의
`ACCEPTED|DUPLICATE`에서 완료하고, retryable reject/network failure는 동일 observationId로
지수 backoff 재전송하며 permanent reject는 DLQ에 보존합니다. STREAM ADAPTIVE bitrate/health
변경은 command별 monotonic revision과 signed desired superset으로 관측됩니다.

`CONFIG_APPLY`는 strict signed payload의 `model|labels|clip|video` 중 하나 이상을 요구합니다.
labels는 `inspection`/`anomaly`별 1..100개의 trim된 unique string 배열입니다. IMMEDIATE는
runtime이 실제 변경/readback 가능한 효과만 성공시키며, RESTART는 atomic config와 LKG를
stage한 뒤 새 process에서 실제 runtime readback 및 functional health를 검증합니다. Model은
env digest를 증거로 사용하지 않고 authenticated resolver pointer와 실제 다운로드 artifact
bytes에서 계산한 manifest/aggregate digest가 signed digest와 일치할 때만 성공합니다. 실패 시
active SigLIP backend가 그 authenticated local directory에서 실제 load되었다는 source identity까지
일치해야 성공합니다. Triton/remote backend가 exact loaded identity를 증명하지 못하거나 실패하면
LKG rollback/restart를 완료하기 전에는 `FUNCTIONAL_HEALTHY`를 보고하지 않습니다.

Canonical adaptive payload는 다음 field만 허용합니다. Tuning field가 없으면 Agent의 versioned
default를 적용하며, `min <= initial <= max`, bitrate `100..20000`을 검증합니다.

```json
{
  "policyVersion": 7,
  "mode": "ADAPTIVE",
  "minBitrateKbps": 300,
  "maxBitrateKbps": 3000,
  "initialBitrateKbps": 1200,
  "decreaseFactor": 0.75,
  "increaseStepKbps": 100,
  "congestionSamples": 3,
  "recoverySamples": 8,
  "cooldownSeconds": 5
}
```

`FIXED`는 `targetBitrateKbps`, `DISABLED`는 공통 `policyVersion`/`mode` 외 field를 허용하지
않습니다. Clip이 켜진 경우 WebRTC와 clip은 raw tee 뒤 독립 encoder를 사용하므로 adaptive
bitrate 변경이 forensic clip encoder에 전파되지 않습니다.

Connectivity 보고 정책:
- Adaptive controller는 `webrtcbin get-stats`의 outbound RTP loss/RTT/NACK/PLI를 우선 사용하고,
  해당 primary signal이 없을 때만 아래 OS connectivity 지표를 보조 신호로 사용합니다.
- macOS는 `airport -I`, Linux(Jetson)는 `iw dev <iface> link`에서 RSSI를 수집합니다.
- 공통으로 `ping` 평균 RTT/패킷손실을 수집합니다.
- `uplinkKbps/downlinkKbps`는 OS별 무선 링크 bitrate를 기반으로 채웁니다.
  - macOS: `airport -I`의 `lastTxRate/maxRate` 기반
  - Linux/Jetson: `iw ... link`의 `tx bitrate/rx bitrate` 기반
- `quality` 전이(`GOOD ↔ POOR`) 시점에만 `/app/device/connectivity`를 송신합니다.

Optional deps:
- Zero-shot: `pip install -e .[zsad]`
- Triton: `pip install -e .[triton]`
- `zsad` extras pins `transformers<5` for SigLIP2 runtime compatibility.

## Macbook MPS demo (SigLIP2 ZSAD)
```bash
pip install -r nuvion_app/inference/requirements-zsad.txt
nuv-agent set-inference --backend mps
nuv-agent run
```

## Triton backend demo
```bash
# pip install -r nuvion_app/inference/requirements-triton.txt
NUVION_ZSAD_BACKEND=triton python -m nuvion_app.agent.zsad_siglip_demo
```

## Triton backend notes
- 기본 운영 경로는 **Triton + AnomalyCLIP** 입니다.
- 기본 Triton 모델은 `image_encoder`, 입력은 `image`, 출력은 `image_features` 입니다.
- macOS와 Raspberry Pi에서는 TensorRT(`model.plan`)를 사용하지 않고, ONNX 기반 `model_repository_onnx`를 자동 생성/사용합니다.

### AnomalyCLIP Triton mode
AnomalyCLIP image encoder + precomputed text features를 함께 사용하려면:
```bash
export NUVION_ZSAD_BACKEND=triton
export NUVION_TRITON_MODE=anomalyclip
export NUVION_TRITON_MODEL=image_encoder
export NUVION_TRITON_INPUT=image
export NUVION_TRITON_IMAGE_FEATURES_OUTPUT=image_features
# pull-model을 --local-dir ~/.cache/nuvion/models/anomalyclip-current 로 실행했다고 가정
export NUVION_TRITON_TEXT_FEATURES=$HOME/.cache/nuvion/models/anomalyclip-current/onnx/text_features.npy
export NUVION_TRITON_THRESHOLD=0.7
```

설명:
- `NUVION_TRITON_MODE=anomalyclip`: Triton 출력 `image_features`와 `text_features.npy`를 결합해 anomaly probability 계산
- `NUVION_TRITON_TEXT_TEMPERATURE`: 기본 `0.07` (softmax temperature)
- `NUVION_TRITON_ANOMALY_INDEX`: anomaly class 인덱스 (기본 `1`)

## Troubleshooting (수동 복구)
자동 bootstrap이 정책/네트워크/권한 제약으로 실패할 때만 수동 명령을 사용하세요.

1) Triton 수동 실행
```bash
docker rm -f triton-nuv 2>/dev/null || true
docker run -d --name triton-nuv -p 8000:8000 \
  -v ~/.cache/nuvion/models/anomalyclip-current/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
```

macOS 수동 실행(ONNX):
```bash
docker rm -f triton-nuv 2>/dev/null || true
docker run -d --name triton-nuv -p 8000:8000 \
  -v ~/.cache/nuvion/models/anomalyclip-current/triton/model_repository_onnx:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
```

2) 헬스체크
```bash
curl -s http://127.0.0.1:8000/v2/health/ready
curl -s http://127.0.0.1:8000/v2/models/image_encoder/config
```

## Target platforms
- Nuvion: Raspberry Pi 5 + DEEPX DX-M1
- Nuvion Pro: Ventuno Q
- Nuvion Ultra: Jetson Orin NX
- IQ-9075 EVK 개발보드 + Luxonis OAK-D Lite
- Apple Silicon Mac (MPS) 개발/디버그

## Notes
- `nuvion_app/docker-compose.yml` is configured for the Linux device runtime.
