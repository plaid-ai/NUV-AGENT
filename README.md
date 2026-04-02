# Nuvion Agent (Device Software)

NUV-agent는 온디바이스 AI 장치에 설치하는 소프트웨어입니다. 도커 기반으로 제작되어 이식이 쉽고,
USB 기반 웹캠 카메라 스트림을 Nuvion-be 스프링 서버를 통해 송출할 수 있습니다. 동시에 제로샷 AI 모델로
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
  --server-base-url https://api.nuvion-dev.plaidai.io \
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
- `NUVION_MODEL_SERVER_BASE_URL=https://api.nuvion-dev.plaidai.io`
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
- `NUVION_VIDEO_SOURCE`: camera source. Linux는 `/dev/video0`, macOS는 `avf` 또는 `avf:<index>`, Raspberry Pi는 `rpi`
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
- `NUVION_CONFIG_SCHEMA_VERSION`: config schema 버전 (`doctor --fix`로 자동 보정)
- `NUVION_RUNTIME_BOOTSTRAP_ENABLED`: setup/run bootstrap 전체 on/off
- `NUVION_HOMEBREW_AUTOINSTALL`: macOS Homebrew 자동 설치 허용
- `NUVION_DOCKER_AUTOINSTALL`: Docker/Colima(또는 docker.io) 자동 설치 허용
- `NUVION_DOCKER_AUTOSTART`: Docker daemon 자동 기동 허용
- `NUVION_DOCKER_DESKTOP_TIMEOUT_SEC`: macOS Docker Desktop daemon 준비 대기 시간(초)
- `NUVION_TRITON_AUTOSTART`: Triton 컨테이너 자동 기동 허용
- `NUVION_TRITON_AUTOSTART_ONLY_LOCAL`: local Triton URL에서만 자동 기동
- `NUVION_TRITON_AUTOSTOP_ON_EXIT`: agent 종료 시 자동 기동한 Triton 컨테이너 자동 종료 (기본 `true`)
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

macOS note: use `NUVION_VIDEO_SOURCE=avf` (default camera) or `avf:<index>` to select a camera.

### Agent WebSocket error queue
- Agent는 STOMP에서 `/user/queue/agent/error`를 구독합니다.
- `retryable=true` 에러는 마지막 uplink payload(`/app/device/*`, `/app/broadcast/start`)를 백오프로 재전송합니다.
- `401/403` 같은 non-retryable 권한 오류는 uplink를 차단하고 로그에 원인(`code`, `path`, `detail`)을 남깁니다.

Connectivity 보고 정책:
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
- Jetson Nano / ARM 기반 장치 + Triton 서빙
- Apple Silicon Mac (MPS) 로컬 테스트

## Notes
- `nuvion_app/docker-compose.yml` is configured for Linux device runtime (USB camera).
