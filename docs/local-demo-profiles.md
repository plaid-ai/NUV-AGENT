# 로컬 데모 입력 선택

`NUVION_LOCAL_DEMO_ENABLED=1`인 DEMO 런타임에서만 활성화된다. 로그인 없는 loopback UI에서 MVTec 또는 설치된 MP4를 선택한다. 운영 카메라 입력은 변경하지 않는다.

## 장치 설치

`NUVION_LOCAL_DEMO_PROFILES`(기본 `/var/lib/nuv-agent/demo/profiles`) 디렉터리를 서비스 사용자 소유로 생성한다. `profiles.json` 예:

```json
[
  {"id":"video-1","name":"현장 영상 1","file":"video-1.mp4"},
  {"id":"video-2","name":"현장 영상 2","file":"video-2.mp4"}
]
```

MVTec은 기본 항목으로 자동 포함된다. 경로는 이 디렉터리 아래 MP4만 허용한다. 요청에서 경로를 받거나 외부 URL을 다운로드하지 않는다. H.264 MP4와 GStreamer `avdec_h264`를 권장한다. 서비스가 `selected.json`을 원자적으로 기록할 수 있어야 한다.

FE의 `pnpm build:local-demo` 결과를 `NUVION_LOCAL_DEMO_ASSETS`에 설치한다. `NUVION_SUPERVISOR_RESTART_ENABLED=1`, systemd `Restart=always`가 필요하다. 입력 전환은 프로세스를 정상 종료하고 재시작하여 모델·큐·판정을 초기화한다. 반복 전환이 서비스 시작 제한에 걸리지 않도록 로컬 데모 전용 drop-in에 다음을 설정한다:

```ini
[Unit]
StartLimitIntervalSec=60
StartLimitBurst=30
```

영상은 별도 디코더에서 반복한다. FLUSH seek와 EOS는 디코더 내부에만 적용하며, 메인 appsrc는 연속된 running-time을 부여해 WebRTC 타임스탬프가 되감기지 않게 한다. 모델은 재로딩하지 않는다. 원래 영상 속도를 유지하고 최신 프레임 하나만 추론 대기열에 유지한다. 루프 경계를 넘어 도착한 결과는 폐기한다. 영상 박스 TTL은 0.75초로 제한한다. 영상에 GT가 없으므로 UNKNOWN을 표시하며 중앙 데이터셋 이벤트는 발행하지 않는다. 로컬 데모는 클립 저장을 요청하지 않는다. MVTec의 기존 정답·1.7초 사진 전환 설정은 유지된다.

## API

- `GET /api/profiles`: 설치 목록과 `activeId`, `pendingId`.
- `POST /api/profile`: JSON `{"id":"video-1"}`. localhost Host와 same-origin Origin 필수. 성공202, 잘못된 입력400, 전환 불가409. 새 프로세스의 GET 응답으로 완료를 확인한다.
- UI는 전환 중 이전 프레임을 숨기며 재시작 동안 재연결한다.

## 실기 검증 (2026-10-29)

Ventuno Q에서 MVTec → 영상1 → 영상2 → MVTec 전환, QNN/HTP 추론 약0.41초, 영상 EOS 반복 및 프레임 갱신을 확인했다. 기존 장치 전용 Cairo 결과 오버레이는 유지한 overlay 배포이며, 정식 릴리스 패키지와 구분한다. 영상은 미보정 시연 기준을 사용하므로 실제 불량 정확도 검증을 의미하지 않는다.
