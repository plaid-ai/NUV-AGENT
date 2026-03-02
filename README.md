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

## Quick start (dev)
1) Copy `.env.example` to `.env` and fill in credentials.
2) Run locally:
   ```bash
   pip install -e .
   python -m nuvion_app.cli run
   ```

Python requirement: 3.10+

## Pull model bundle (GCS bucket)
GCS pointer를 통해 모델 번들을 내려받아 Triton/AnomalyCLIP 런타임에 바로 연결할 수 있습니다.
```bash
# runtime: text_features + Triton model_repository (권장)
nuv-agent pull-model \
  --gcs-pointer-uri gs://nuv-model/pointers/anomalyclip/prod.json \
  --local-dir ~/.cache/nuvion/models/anomalyclip-current \
  --profile runtime
```

Profiles:
- `runtime`: Triton + text features 실행에 필요한 파일만 다운로드
- `light`: text features/metadata 중심의 경량 다운로드
- `full`: 추가 분석/검증 파일까지 포함해서 다운로드

기본값:
- `NUVION_MODEL_GCS_POINTER_URI=gs://nuv-model/pointers/anomalyclip/prod.json`
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

## Setup UI (device)
If a display is available, run:
```bash
nuv-agent setup
```
This starts a local setup UI at `http://127.0.0.1:8088` (override with `--host/--port`).
The setup UI includes an **Auto Provision** section: login with an owner/admin account to create
device credentials automatically (your account credentials are not stored on the device).

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

## Service
- Linux: use `packaging/systemd/nuv-agent.service` and `systemctl enable --now nuv-agent`.
- macOS: use Homebrew service definition in `packaging/homebrew/nuv-agent.rb`.

## Device configuration
- `NUVION_VIDEO_SOURCE`: USB webcam path (e.g., `/dev/video0`) or `rpi` for Pi camera
- `NUVION_ANOMALY_LABELS`: comma-separated labels treated as anomalies
- `NUVION_PRODUCTION_LABELS`: comma-separated labels counted for production
- `NUVION_ZERO_SHOT_ENABLED`: enable optional zero-shot anomaly detection (requires model deps)
- `NUVION_ZSAD_BACKEND`: 기본 `triton` (장애 대응 시 `siglip`로 수동 전환 가능)
- `NUVION_ZERO_SHOT_MODEL`: 기본 ZSAD 모델 (`google/siglip2-base-patch16-224`)
- `NUVION_MODEL_GCS_POINTER_URI`: GCS pointer JSON URI (default: `gs://nuv-model/pointers/anomalyclip/prod.json`)
- `NUVION_MODEL_PROFILE`: pull-model 프로필 (`runtime|light|full`)
- `NUVION_MODEL_DIR`: pull-model 기본 저장 루트
- `NUVION_AGENT_ERROR_MAX_RETRIES`: 서버 agent error(`retryable=true`) 수신 시 자동 재시도 최대 횟수 (기본 `3`)
- `NUVION_AGENT_ERROR_BACKOFF_BASE_SEC`: 첫 재시도 대기 시간(초), 이후 지수 백오프 (기본 `1.0`)
- `NUVION_AGENT_ERROR_BACKOFF_MAX_SEC`: 재시도 최대 대기 시간(초) (기본 `15.0`)

macOS note: use `NUVION_VIDEO_SOURCE=avf` (default camera) or `avf:<index>` to select a camera.

### Agent WebSocket error queue
- Agent는 STOMP에서 `/user/queue/agent/error`를 구독합니다.
- `retryable=true` 에러는 마지막 uplink payload(`/app/device/*`, `/app/broadcast/start`)를 백오프로 재전송합니다.
- `401/403` 같은 non-retryable 권한 오류는 uplink를 차단하고 로그에 원인(`code`, `path`, `detail`)을 남깁니다.

Optional deps:
- Zero-shot: `pip install -e .[zsad]`
- Triton: `pip install -e .[triton]`
- `zsad` extras pins `transformers<5` for SigLIP2 runtime compatibility.

## Macbook MPS demo (SigLIP2 ZSAD)
```bash
pip install -r nuvion_app/inference/requirements-zsad.txt
python -m nuvion_app.agent.zsad_siglip_demo --show
```

## Triton backend demo
```bash
# pip install -r nuvion_app/inference/requirements-triton.txt
NUVION_ZSAD_BACKEND=triton python -m nuvion_app.agent.zsad_siglip_demo
```

## Triton backend notes
- 기본 운영 경로는 **Triton + AnomalyCLIP** 입니다.
- 기본 Triton 모델은 `image_encoder`, 입력은 `images`, 출력은 `image_features` 입니다.

### AnomalyCLIP Triton mode
AnomalyCLIP image encoder + precomputed text features를 함께 사용하려면:
```bash
export NUVION_ZSAD_BACKEND=triton
export NUVION_TRITON_MODE=anomalyclip
export NUVION_TRITON_MODEL=image_encoder
export NUVION_TRITON_INPUT=images
export NUVION_TRITON_IMAGE_FEATURES_OUTPUT=image_features
# pull-model을 --local-dir ~/.cache/nuvion/models/anomalyclip-current 로 실행했다고 가정
export NUVION_TRITON_TEXT_FEATURES=$HOME/.cache/nuvion/models/anomalyclip-current/onnx/text_features.npy
export NUVION_TRITON_THRESHOLD=0.7
```

설명:
- `NUVION_TRITON_MODE=anomalyclip`: Triton 출력 `image_features`와 `text_features.npy`를 결합해 anomaly probability 계산
- `NUVION_TRITON_TEXT_TEMPERATURE`: 기본 `0.07` (softmax temperature)
- `NUVION_TRITON_ANOMALY_INDEX`: anomaly class 인덱스 (기본 `1`)

## Target platforms
- Jetson Nano / ARM 기반 장치 + Triton 서빙
- Apple Silicon Mac (MPS) 로컬 테스트

## Notes
- `nuvion_app/docker-compose.yml` is configured for Linux device runtime (USB camera).
