# VisualAD HTP Fleet local-model 계약 v1

2026-09-07 source 구현. 실장치 Fleet 활성화·명령 E2E·signed release 승인과는 별개다.
기존 HTP 실험 모델은 계속 `PARITY_FAILED` / `UNCALIBRATED`다. 이 연결은 정확도 합격,
원격 다운로드, 임의 VisualAD 가중치 또는 다른 inference backend 지원을 의미하지 않는다.

## Wire 계약과 저장소

기존 signed `CONFIG_APPLY` schemaVersion 1을 사용한다. `activation=RESTART`만 허용하며
threshold 필드는 추가하지 않는다. `model`은 다음 두 필드를 유지한다.

```json
{"pointer":"visualad/iq9075-htp-demo","digest":"sha256:<fleet-model.json 실제 SHA256>"}
```

이 pointer에 대응하는 capability는 `command.config.model.visualad_htp.v1`이다.
기존 SigLIP capability와 구분한다. 플랫폼 static profile에 넣지 않으며 실제 trusted
Fleet runtime/keyring, CONFIG_APPLY reconciler, 현재 pipeline adapter, 실제 fresh HTP
loaded proof가 모두 있어야 광고한다. capability는 지원 능력이며 적용 완료 ACK가 아니다.

`NUVION_VISUALAD_HTP_FLEET_STORE`는 operator가 만든 canonical absolute root-owned
directory다. service가 쓸 수 없는 각 `<wrapper digest의 64자리 hex>/` 아래에
`fleet-model.json`, `manifest.json`, `visualad.onnx`를 배치한다. 파일·directory는 root
소유이고 group/world writable이면 거부한다. symlink 선택과 덮어쓰기는 지원하지 않는다.

Wrapper는 정확히 다음 구조다. 원본 ONNX manifest/graph bytes는 변경하지 않는다.

```json
{"backend":"visualad_htp","manifest":{"path":"manifest.json","sha256":"89114e2159e4e6080971d5fcbe4782829daad78406f043eb877d603885fc4795"},"pointer":"visualad/iq9075-htp-demo","schemaVersion":1}
```

CLI는 정렬된 compact JSON + 마지막 LF로 wrapper를 생성한다. 위 export-001의 wrapper
digest는 `sha256:12506ab00a993b44cf5a1354bd2fd9154db367387e8211eae20e67631f3acb56`이다.
이는 checkpoint SHA 또는 원본 manifest SHA가 아니다. 실행 시 실제 wrapper/manifest/
graph를 별도로 검증하며 문서의 digest만으로 로딩 성공을 주장하지 않는다.

Operator provisioning/검증 CLI는 모델을 실행하지 않는다. 승인된 source/dependencies의
Python 환경에서 다음 module을 사용한다. 실제 경로와 신뢰된 외부 manifest pin은 operator가
명시한다. `provision`은 새 digest directory만 생성하며 기존 디렉터리를 재사용하지 않는다.

```text
python -m nuvion_app.runtime.visualad_fleet provision --store <절대 store> --source-manifest <기존 root-owned model/manifest.json> --manifest-sha256 <신뢰된 원본 pin>
python -m nuvion_app.runtime.visualad_fleet verify --store <절대 store> --digest sha256:<wrapper digest>
```

원본 모델과 별도의 독립 파일 복사본을 만들고 실제 SHA/크기를 확인한다. CLI 결과의
`LOCAL_ARTIFACT_BYTES_ONLY`, `inferenceReady=false`는 정상이다. 다운로드나 NPU 실행은 없다.
실패한 부분 staging은 덮어쓰지 않으며 operator가 원인을 확인한다.

## Startup, 첫 LKG, 적용 증명

기존 `run-iq9075-visualad-htp-demo.sh`는 기본 Fleet=false를 유지한다. Operator가 다음
환경을 명시한 경우에만 새 Fleet startup 경로를 선택한다.

- `NUVION_VISUALAD_HTP_FLEET_ENABLED=true` — 정확히 `true` 또는 `false`; 잘못된 값은 실패.
- `NUVION_VISUALAD_HTP_FLEET_STORE` — 위 root-owned local store.
- `NUVION_MODEL_POINTER`, `NUVION_MODEL_DIGEST` — **검증된 초기 모델** wrapper identity.
- `NUVION_COMMAND_INBOX_PATH` — 기존 정확한 inbox 경로를 유지; 빈 값 불가.
- 기존 deployment root / HTP state / settings directory 설정 — 별도 승인된 경로 유지.
  State/settings는 nuvion 전용 `0700`, source/package/model은 root-owned로 유지한다.
- 기존 device/space/trust domain/keyring/서명키 설정 — 새 로컬 명령 우회 경로는 없다.

초기 wrapper를 store에 미리 배치하고 base EnvironmentFile의 pointer/digest를 유지해야
한다. 첫 Fleet model command 전 `active.env`가 비어 있으면 이 base identity가 첫 LKG다.
별도 `restart-marker.json`을 만들거나 과거 signed command/inbox/settings를 복사해 seed하지
않는다. 이후 signed model command가 활성화한 `active.env`가 base pointer/digest보다
우선하며, 실패 시 기존 AtomicSettingsStore가 이전 overlay를 복구한다.

실제 시작 순서는 `prepare_fleet_startup()` → 기존 settings boot guard → `load_env()`의
복구된 overlay → pipeline import → HTP prepare → 새 실제 프레임 추론이다. Boot guard는
process당 한 번만 실행하며 다른 settings/inbox/store context로 바뀌면 거부한다. 기존
91 override처럼 이전 ExecStartPre guard를 제거한 launch 경로에서 사용해야 한다. 같은
candidate boot에 별도 ExecStartPre guard와 새 entrypoint를 중복 실행하지 않는다.

Launcher의 `--import-check`는 dependency/module-origin 검사만 하고 boot attempt를 소비하지
않는다. 따라서 startup/모델 검증을 대신하지 않는다. 직접 Python으로 시작해야 한다면
`python -m nuvion_app.runtime.visualad_fleet_start`를 사용한다. pipeline을 먼저 import한
후 Fleet detector를 만드는 경로는 실패한다. 실제 배포는 기존 HTP SDK origin 검사도 유지한다.

새 model command는 먼저 preprovisioned target 전체 bytes를 검사한다. 없거나 변조된
target이면 `MODEL_PREFLIGHT_FAILED`이며 overlay/restart를 변경하지 않는다. 재시작 후
compile/첫 frame 대기는 `MODEL_STARTUP_PENDING` / `RETRY_EFFECT`이며 추가 restart나 성공
ACK를 보내지 않는다. process당 최대 600초이고 재시도로 연장하지 않는다. 실제 실패나
timeout은 기존 LKG rollback으로 진행한다. LKG 첫 추론 대기도 같은 한도를 사용한다.
기존 boot guard의 한 번 후보 boot / 한 번 LKG boot 제한과 inbox sequence fence는 유지한다.

Local effect worker는 WebSocket 연결 수명이 아닌 signaling process 수명에 속한다.
로그인 실패·연결 단절·재접속 중에도 compile 대기 재확인, 600초 timeout 및 LKG 복구를
계속 수행한다. Terminal ACK/관측은 durable 저장 후 연결 시 replay하며, 재접속마다
worker를 추가하지 않는다. 프로세스 종료 시 local worker를 cancel하고 await한다.

`prepare()` 성공만으로 ready/적용 완료를 주장하지 않는다. 첫 성공한 QNN-only 실행,
finite output 및 후처리를 통과하고 60초 이내 결과가 있어야 actual proof가 생긴다.
`modelObservedPointer`와 `modelDigest`는 그때 **검증하여 로드한 wrapper**에서 가져온다.
`modelAggregateDigest=unknown`; inner manifest/graph SHA는 기존 inference provenance로
별도 보고한다. 환경값·generic resolver 선언값으로 actual proof를 대신하지 않는다.

Heartbeat는 대형 모델을 다시 해싱하지 않는다. 저장된 loaded-file identity/stat과 결과
freshness를 검사하고, 변경·실패·stale이면 capability/actual proof를 철회한다. 최종 model
성공 ACK는 실제 loaded identity와 signed target 일치, wrapper/manifest/graph 전체 bytes
재검증, pipeline/encoder 및 기존 durable outbox 건강 상태를 모두 확인한 뒤에만 발생한다.

HTP-only / CPU fallback 금지, 연속 latest-frame 정책, 실험·미보정 표시는 유지한다.
기존 signed slot과 비활성 기본 설정은 이 source 변경으로 갱신되지 않는다. 실제 normal/
defect calibration, remote artifact distribution 및 배포 후 Fleet E2E는 별도 작업이다.

## 로컬 회귀 검증 경계

- 실제 `python -m` subprocess에서 canonical boot context를 detector와 공유하는지 검증한다.
- 실제 DurableCommandInbox / SettingsReconciler / FleetEffectCoordinator에 mock inference와
  가상 clock을 연결하여 offline commit, 추론 실패·600초 timeout 후 LKG 복구, reconnect
  단일 worker와 종료 취소를 검증한다.
- 테스트의 가상 clock·mock inference는 실제 IQ9075 네트워크 장애/NPU 검증을 대신하지 않는다.
- 2026-09-07 최종 isolated suite: 77 modules / 999 tests, 실패 0 / 하드웨어·플랫폼 조건부
  skip 6. `python -m` 및 offline worker 수정 합류 후 새 interpreter/config scope로 재실행했다.
