# 실시간 카메라 제어 계약

## 원인과 변경

Ultra 현장 command inbox에서 LEFT가 움직인 뒤 `motor did not reach the requested position`으로 실패한 기록, 이동 전 3초 admission TTL 만료, STOP에 의한 SUPERSEDED를 확인했다. 기존 화면은 이를 같은 실패 문구로 표시하고 각도를 지웠다. 기존 이동 경로는 REST 발행 → 서명/영속 journal → Agent reconcile → REST projection polling이었다.

검토한 방법은 기존 TTL/settle timeout 조정, Fleet REST만 STOMP로 감싸기, 별도 휘발성 제어 경로였다. 처음 두 방법은 누르기마다 journal/목표각 완료 대기를 유지하므로 세 번째를 선택했다. 영구 제한값 저장은 기존 서명된 Fleet 경로를 유지한다.

## 전송 계약

- FE는 WebSocket 전용 SockJS + STOMP `/signaling` 연결을 사용한다. `/user/queue/camera.telemetry` 구독 receipt 뒤 시작한다.
- SEND `/app/camera/control`: `spaceId`, `deviceId`, UUID `requestId`, 증가하는 `sequence`, `action`(LEFT/RIGHT/UP/DOWN/STATUS/STOP), `sentAtMs`.
- BE는 실제 장치의 space 일치, OWNER 권한, 최신 RUNNING heartbeat의 `camera.control.realtime.v1`을 검사한다. STOP은 capability/rate 검사에서 제외하되 소유권 검사는 유지한다.
- BE가 실제 인증 STOMP sessionId와 입력 시각+650ms 만료를 붙여 장치의 `/user/queue/camera.control`로 전달한다. 오래된 입력의 수명을 다시 연장하지 않는다. requestId → 장치/사용자/session 응답 경로는 Redis 10초 TTL로 공유하여 두 replica 간 응답도 같은 브라우저로 전달한다.
- Agent SEND `/app/device/camera.telemetry`: 동일 requestId/sequence, status, 측정 telemetry. BE는 BROADCASTER principal과 Redis route의 장치를 대조하고 원래 브라우저 session에만 전달한다.
- 이 경로는 인증된 TLS STOMP 세션을 신뢰하는 일시적 제어이며 개별 프레임에 Fleet JWS 서명을 붙이지 않는다. 이동 intent/telemetry를 durable journal에 저장하거나 reconnect 시 재생하지 않는다. 구성/OTA/제한 저장의 서명 검증은 그대로 유지한다.

## 입력·정지·각도

- FE는 누르는 동안 150ms마다 방향을 갱신한다. release/blur/hidden/unmount는 이전 ACK를 기다리지 않고 STOP한다. 피드백이 900ms 넘게 없으면 입력을 중단한다. reconnect는 STATUS만 보내며 이전 방향을 복원하지 않는다.
- Agent는 하나의 최신 intent만 보관하고 독립 thread에서 실행한다. lease 만료/transport disconnect는 STOP한다. 입력 만료 기준은 650ms이며 실제 STOP 완료에는 진행 중인 bounded UART 요청 시간과 worker 주기가 추가된다. 하드웨어 비상정지 보증이 아니다.
- 명령 순서·중복을 검사하고 다른 session의 active press는 BUSY로 거절한다. 권한 있는 STOP은 항상 우선한다. 오류/만료 후 같은 press의 갱신은 STOP을 받기 전까지 재개하지 않는다.
- `NUV1-smooth-v3`의 position mode(3), 해당 축의 기본 범위(0..4095)에서는 `RUN` 한 번으로 연속 이동을 시작하고 이후 `KEEP`만 보낸다. 목표각을 매 주기 다시 쓰지 않는다. profile velocity 4(약 5.5°/s), 펌웨어 400ms KEEP watchdog을 유지한다. 방향 전환은 STOP 후 새 RUN이다. 펌웨어가 정지하면 KEEP로 재시작하지 않는다.
- RUN은 펌웨어 한계까지의 목표를 설정하므로, 사용자 지정 소프트웨어 범위 또는 미지원 펌웨어/모드에서는 기존 0.97° bounded JOG를 유지한다. 좁은 범위에도 연속 이동을 제공하려면 범위를 원자적으로 전달하는 펌웨어 계약이 필요하다. 범위를 무시하고 RUN하지 않는다. 이 경로도 중복 STATUS를 제거하여 armed 상태에서 STATUS+JOG 두 번으로 처리한다.
- `MOVING` 응답은 실측 각도를 포함하며 기계적 목표 도착 완료를 뜻하지 않는다. STOP/만료가 UART 조회 도중 도착하면 다음 이동 쓰기 전에 다시 검사한다.
- 속도 단위 근거: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#profile-velocity112
- 저장된 위치 제한은 매 JOG에서 적용한다. 한계 도달 시 STOP 후 실제 각도를 유지하여 표시한다. 수동 lease 중에는 자동 tracking 입력을 억제하고 STATUS 조회는 자동 제어를 점유하지 않는다.

## 배포 및 검증

순서: BE → Agent → FE. 기존 Agent는 capability가 없어 실시간 이동 요청을 거부한다. Agent는 기존 Fleet 원격 제어 활성화 및 안전 게이트를 따른다. 새 기능은 프로토타입 검증 대상이며 양산 Ultra 검증 또는 새 정식 OTA 릴리스가 아니다.

자동 검증: FE 인증/receipt/요청 상관관계, no-REST movement, hold/release, stale response, feedback loss, reconnect, 영구 LIMITS; BE owner/bound-device/expiry/rate/telemetry identity; Agent deadman/sequence/conflict/fault/기존 motor limits.

실기 수용 기준: 양 축 짧은 이동과 release 정지, 브라우저 연결 차단 후 bounded 정지, 한계 도달 시 각도 유지, reconnect 이동 미재개, 제한값 저장 및 재시작 유지. 아직 이 변경의 실기 수용 검증을 완료하지 않았다.
