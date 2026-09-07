# IQ9075 VisualAD HTP 운영 runbook

작성 기준: 2026-09-07. 대상은 **내부 개발 검증·시연**이며 signed Fleet
release/OTA 승인 절차가 아니다. 2026-09-07 사용자가 성능 미달을 감수한 **상시 실험
배포**를 명시적으로 승인했다. export-001을 사용하되 수치 동등성은 계속 FAILED이며,
이 승인으로 정상/불량 정확도 또는 정식 릴리스 합격을 주장하지 않는다.
실측 및 논문/가중치 출처는 `docs/IQ9075_VISUALAD_DEMO.md`를 함께 확인한다.

## 1. 현재 판정과 금지 사항

- 기존 `90-iq9075-visualad-demo.conf`는 영상 전용 복귀 경로로 보존한다.
  신규 91 override는 별도 HTP 상시 실험 경로다. 연속 CPU 모델 추론은 금지한다.
- export-001: 최대 patch map 오차 `0.409958 > 0.1`로 실패.
  manifest `89114e2159e4e6080971d5fcbe4782829daad78406f043eb877d603885fc4795`를
  `deployment.env`에 식별 pin으로 고정한다. 이는 수치 합격을 의미하지 않는다.
- export-002: 수학적으로 동등한 attention 사전 scaling 후보도 최대 오차 약
  `0.418022 > 0.1`로 실패. manifest
  `bd3872db91b04fc3c15f95f9d33dfc716f59129252ca8a50d591bf597b377ddf`도 승인하지 않는다.
- 진단용 multi-output graph와 LayerNorm micro graph는 production manifest가 아니다.
  진단 완료를 모델 승인으로 바꾸지 않는다. 원본·실패 evidence는 모두 보존한다.
- 기존 `/opt/nuv-agent/current`, signed release, 기존 CPU venv, 현행 90 drop-in,
  Fleet settings/journal을 덮어쓰거나 삭제하지 않는다.
- learned graph는 QNN/HTP 전용이다. CPU/GPU 모델 fallback, 판정 강제 주입,
  실패를 NORMAL로 처리하는 fallback, 실패 후 수치 기준 완화를 허용하지 않는다.
- 정상/불량 실물 calibration, 실제 감지 event ACK와 BE/FE 일치, snapshot/clip
  READY·재생은 별도 미완료 항목이다. 이번 변환 검증 프레임에는 정답 label이 없다.

## 2. 디렉터리와 실행 주체

| 경로 | 소유·용도 |
| --- | --- |
| `/opt/nuvion-demo/20260910-visualad-htp` | root 소유의 새 HTP 개발 runtime. 기존 CPU/demo prefix와 분리 |
| `…/src` | 승인한 Agent source와 `packaging/dev/run-iq9075-visualad-htp-demo.sh` |
| `…/python/lib/python3.12/site-packages` | 검증한 HTP dependency 전체. 기존 CPU venv를 참조하지 않음 |
| `…/model/manifest.json`, `…/model/visualad.onnx` | 승인 모델 1개, embedded weights, 원본 artifact 그대로 |
| `…/deployment.env` | root:root, `0600`, 실험 배포 승인에 연결한 외부 manifest SHA pin |
| `…/evidence` | staging inventory/SHA, wheel reports/freeze, validation, 회수한 91 drop-in |
| `/var/lib/nuv-agent/visualad-htp` | nuvion:nuvion, `0700`, HTP profile/state |
| `/var/lib/nuv-agent/visualad-htp-settings` | nuvion:nuvion, `0700`, 개발 settings 전용 |

`python/`은 독립 package tree다. launcher의 실제 interpreter는 `/usr/bin/python3 -s`
(Python 3.12)이며, interpreter/OS package도 별도로 기록한다. 실행 UID/GID는 원 unit의
`nuvion:nuvion`을 유지하고 **해당 서비스에만** `SupplementaryGroups=fastrpc`를 추가한다.
기존 `video dialout` 설정은 유지한다. `usermod`, DSP device chmod, 권한 우회는 하지 않는다.
systemd의 여러 `SupplementaryGroups=` 항목은 누적된다.
[systemd 실행 설정 근거](https://github.com/systemd/systemd/blob/v255/man/systemd.exec.xml)

고정 baseline dependency 경로:

```text
/opt/nuv-agent/releases/26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1/venv/lib/python3.12/site-packages
```

이 경로에서는 DepthAI 등 기존 배포 dependency만 읽는다. `gi`는
`/usr/lib/python3/dist-packages`를 사용한다. 개발 source를 baseline source로 대체하지 않는다.

### 동일 기능을 새 prefix에 staging

launcher는 다음 operator 설정만으로 별도 staging 경로를 선택할 수 있다.
기존 installer는 원래 prefix 전용이므로 새 prefix 설치에 그대로 재실행하지 않는다.

| 변수 | 미지정 시 기존 기본값 | 허용 범위 |
| --- | --- | --- |
| `NUVION_VISUALAD_DEPLOYMENT_ROOT` | `/opt/nuvion-demo/20260910-visualad-htp` | `/opt/nuvion-demo/` 하위 |
| `NUVION_VISUALAD_DEPLOYMENT_HTP_STATE_DIR` | `/var/lib/nuv-agent/visualad-htp` | `/var/lib/nuv-agent/` 하위 |
| `NUVION_VISUALAD_DEPLOYMENT_SETTINGS_DIR` | `/var/lib/nuv-agent/visualad-htp-settings` | `/var/lib/nuv-agent/` 하위 |

세 경로는 이미 존재하는 canonical absolute directory여야 한다. 빈 값, 범위 root
자체, 상대 경로, 상위 탈출, symlink component, 중복 separator 및 trailing slash는
거부한다. 디렉터리 이름은 영문·숫자·`._-`로 제한한다. state 두 경로는 운영자가
미리 nuvion:nuvion `0700`으로 생성한다. 원 unit에서 상속된
`NUVION_SETTINGS_STATE_DIR`는 새 override로 취급하지 않으므로 원 Fleet state로
묵시적으로 전환되지 않는다. 향후 명시적으로 `/var/lib/nuv-agent/settings`를
선택할 수 있지만 Fleet 자체는 계속 비활성이다.

새 root에 정확한 git source, 검증된 HTP package tree의 독립 복사본, 동일 manifest와
graph를 배치하고 source/wheel/model inventory를 다시 검증한다. 기존 prefix 및 signed
slot은 수정하지 않는다. 새 `deployment.env`에 위 세 변수와 기존 manifest SHA pin을
기록한 뒤, 새 launcher `--import-check`에도 동일한 변수들을 명시적으로 전달한다.
실행 시 source/model/SDK 경로는 새 root를 따르고, DepthAI baseline/system GI 경로는
그대로다. HTP-only, sample 0, Fleet=false, threshold 및 experimental 표시는 변경하지 않는다.

활성화가 별도로 승인되면 기존 90/91을 보존하고 새 전용 override에서 새 launcher와
EnvironmentFile을 선택한다. 그 override만 회수하면 이전 exp3 경로로 복귀할 수 있다.
경로 일반화 이후의 source commit/SHA를 새로 기록하며 이전 source SHA로 표시하지 않는다.

## 3. 정식 수치 승인 기준과 실험 staging — 서비스 재시작 전

아래 1–4의 수치 합격은 **정식 모델 수치 승인 경로**다. 이번 사용자 승인 실험은
2번 최대 오차 실패를 그대로 보존한 예외 경로이며 `PARITY_FAILED`, `UNCALIBRATED`,
`experimental=true`를 runtime telemetry와 판정 로그에 명시한다. 화면은
`EXP / UNCALIBRATED`를 표시한다. 3번 NPU-only 및 5–7번 staging 무결성 기준은
실험에서도 완화하지 않는다. 동작 건강 상태와 검출/변환 정확도 상태를 별도로 취급한다.

1. 동일 고정 가중치/입력에 대한 FP32 export reference를 확인한다. production graph는
   ONNX opset 20, `image: FP32 [1,3,518,518]` → `patch_maps: FP32 [1,4,37,37]`이다.
   원래 `official_test_base` LN 정책과 source/backbone/checkpoint 식별자를 유지한다.
2. 실제 IQ9075의 독립 bounded probe에서 다음 **사전 정의 기준 전부**를 만족해야 한다.
   raw score 절대 오차 ≤ `0.01`, patch map 평균 절대 오차 ≤ `0.01`, 최대 절대 오차
   ≤ `0.1`, 정확한 shape/dtype, 모든 값 finite, 성공 3회, `last_error` 없음.
   기준 입력/reference의 SHA와 비교 대상 graph SHA를 함께 보관한다.
3. ORT 실행 trace의 provider가 `QNNExecutionProvider`뿐이어야 한다. provider 목록에
   CPU가 등록돼 있다는 사실과 실제 CPU node 실행을 구분한다. 세션 설정
   `session.disable_cpu_ep_fallback=1`, provider 설정 `offload_graph_io_quantization=0`,
   `libQnnHtp.so` backend, NPU device 선택 및 단일 SDK bundle 검사를 모두 유지한다.
4. 담당 운영자(인프라 관리자 또는 그 위임을 받은 자동화)가 위 결과와 artifact를
   검증한 뒤에만 외부 manifest SHA를 고정한다. 두 사람 승인은 요구하지 않는다.
   `sha256sum model/manifest.json` 출력값을 검토 없이 accepted pin으로 복사하면 안 된다.
   숫자 합격은 현장 정확도·상용 라이선스·signed OTA 승인과 별개다.
5. 새 prefix에만 source/package/model을 stage한다. 기존 경로가 이미 있으면 소유자,
   inventory와 사용 여부를 먼저 확인하고 중단한다. 기존 내용을 덮어쓰는 설치는 금지한다.
6. root 소유, 서비스 UID가 쓸 수 없는 source/python/model과 launcher 실행 권한을 확인한다.
   `deployment.env`는 regular file이어야 하며 소유자·mode를 확인한다. 별도 state/settings
   두 디렉터리는 nuvion 소유 `0700`으로 만든다. 원 Fleet settings를 복사하지 않는다.
7. 승인된 원본 inventory와 배포 파일을 **전부** 비교한다. source archive·각 source 파일,
   launcher/drop-in, interpreter와 OS package 버전, wheel URL/SHA, dependency freeze,
   QNN host library 및 V73 skeleton, model/manifest의 크기/SHA를 기록한다.
   모델 manifest는 package/source 전체의 무결성 증명이 아니다.

검증 환경의 최종 own-venv freeze SHA는
`0e0a275d8893a15573388b3588a1e03f1ccf2d5da0686cb64bd8788e6382c468`이다.
실제 staging helper가 다시 생성한 freeze 파일은
`c91d7d4f13ba44406d40bb99965e8c4dbe2f7e7762b9809d3864fc3dcc8e0940`로 별도 보존했다.
주요 pin은 ORT `1.26.0`, QNN EP `2.5.0`, ONNX `1.22.0`, NumPy `2.5.3`,
Pillow `12.3.0`, SciPy `1.18.1`, OpenCV headless `5.0.0.93`이다. wheel 보고서와
전체 freeze를 함께 검증하며 기존 full-htp-001의 이전 freeze를 갱신하지 않는다.

QNN bundle은 QAIRT `2.49.40` 전체를 사용한다. 보드의 시스템 QAIRT `2.46`을
`LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`에 추가하지 않는다. HTP v73은 자동 탐지하며
Linux `soc_id=676`을 QNN `soc_model`에 넣지 않는다. 현재 SDK에서
`enable_htp_fp16_precision=0`을 FP32 우회 옵션으로 사용하지 않는다.
[QNN EP 문서](https://github.com/onnxruntime/onnxruntime-qnn/blob/v2.5.0/docs/execution_providers/QNN-ExecutionProvider.md),
[FP16 문서 정정](https://github.com/onnxruntime/onnxruntime-qnn/pull/782)

이번 실험의 `deployment.env`는 다음과 같다. 원본 export-001의 identity일 뿐
통과 인증서가 아니다. 패키지의 `iq9075-visualad-htp-deployment.env`가 원본이다.

```ini
NUVION_VISUALAD_HTP_MANIFEST_SHA256=89114e2159e4e6080971d5fcbe4782829daad78406f043eb877d603885fc4795
NUVION_VISUALAD_THRESHOLD=0.0
```

threshold는 확률이 아닌 finite `[-8,8]` raw score다. launcher의 fallback `0.0`은
미보정 시작값일 뿐이다. 실물 calibration 전에는 그 사실을 기록하고 검출 정확도를
주장하지 않는다. `deployment.env`에 credential을 추가하지 않는다.

## 4. 실제 launch 환경 확인

launcher가 지정하는 중요한 값:

| 항목 | 값·의미 |
| --- | --- |
| `PYTHONPATH` | 새 `src` → 새 HTP package tree → 고정 baseline → system dist-packages |
| `PYTHONNOUSERSITE`, `PYTHONSAFEPATH` | 둘 다 `1`; user site와 cwd 우선 import 방지 |
| `LD_LIBRARY_PATH`, `ADSP_LIBRARY_PATH` | 새 HTP package tree의 `onnxruntime_qnn` 하나로 교체 |
| `NUV_AGENT_CONFIG` | `/etc/nuv-agent/agent.env`, 기존 파일 변경 없음 |
| backend / enabled | `visualad_htp` / `true`, 학습모델 CPU fallback 없음 |
| video / demo mode | `oak` / `false`; 실제 카메라 경로 |
| Fleet commands / face tracking | 둘 다 `false` |
| sample interval | `0`초, single worker 최신 프레임 연속 추론; 실측 약 1.503 FPS |
| BOM / agent version | development/UNCONFIGURED, `0.1.121+iq9075.visualad.htp.exp3` |

`PYTHONSAFEPATH`는 `-m/-c`의 cwd 우선 import를 막으면서 명시적 `PYTHONPATH`는 유지한다.
`-I`로 바꾸면 필요한 `PYTHONPATH`도 무시되므로 동등한 변경이 아니다.
[Python 3.12 실행 옵션](https://docs.python.org/3.12/using/cmdline.html#cmdoption-P)

root 운영자는 활성화 전에 staging SHA 검사 후 launcher `--import-check`를
**실제 nuvion identity, 명시적 deployment pin, 별도 0700 settings/state**로 실행한다.
이 검사는 pipeline/HTP adapter가 새 `src`, NumPy/OpenCV/ONNX/ORT/QNN/SciPy가 새
package tree, DepthAI가 고정 baseline, GI가 system 경로에서 왔는지 검사하고 Torch가
import되지 않았는지 확인한다. Pillow의 pin/경로도 inventory에서 확인한다.
import 검사는 모델이나 카메라 객체를 만들지 않으며 수치 합격을 대신하지 않는다.

OpenCV 5 wheel은 import 시 자체 `cv2/../../lib64`를 `LD_LIBRARY_PATH`에 추가한다.
exp1 실서비스에서 이 동작 때문에 엄격 SDK 경로 검사가 정상적으로 fail-closed했다.
exp2 launcher는 고정 package tree의 OpenCV를 먼저 import하고, 그 정확한 prefix만
허용한 뒤 단일 QNN 경로로 복원한다. 같은 Python process의 `runpy`로 main을 시작해
재import를 피한다. 실제 mapped QNN library 검사는 그대로이며 시스템 SDK는 허용하지 않는다.

`deployment.env`는 systemd가 읽는 EnvironmentFile이다. launcher를 수동 호출하면 자동으로
읽히지 않으므로 운영자가 검토한 값만 명시적으로 전달한다. 테스트를 위해 기존 서비스의
settings 권한을 변경하지 않는다. 개발 import 검사는 승인된 빈 0700 settings에서 성공했다.

## 5. 승인 이후 활성화

먼저 `systemctl show nuv-agent.service`에서 `User`, `Group`, `WorkingDirectory`,
`DropInPaths`, `ExecStart`, `SupplementaryGroups`, `EnvironmentFiles`만 필요한 범위로
확인한다. credential이 있을 수 있는 전체 환경 덤프는 피한다. 현행 90 파일의 SHA와
실제로 CPU reference가 비활성화된 상태를 기록한다. 91 이후의 다른 override가 있다면 중단한다.

1. 현재 90 drop-in과 기존 signed slot은 그대로 둔다.
2. 승인한 `91-iq9075-visualad-htp-demo.conf` **하나만** 새로 설치한다.
   이 파일은 required `deployment.env`를 읽고 `ExecStart`를 새 HTP launcher로 바꾼다.
   `ExecStartPre=`는 이전 pre-start 목록을 비우므로 signed release guard 실행과
   동일한 경로로 설명하지 않는다. 개발 staging/모델 검증은 위 절차로 별도로 수행한다.
3. 운영자가 `systemctl daemon-reload`, `systemctl restart nuv-agent.service`를 한 번 수행한다.
   예전 프로세스가 종료되기 전에 별도 camera pipeline을 실행하지 않는다.
4. systemd active만으로 합격 처리하지 않는다. 새 PID/실행 경로, 실제 영상 frame 증가,
   HTP readiness, error 없음, runtime의 실제 graph/loaded-manifest SHA, 첫 ORT QNN-only
   trace와 SDK mappings를 확인한다. 초기 compile 중의 오래된 프레임을 실시간 결과로 세지 않는다.
5. 실행 실패이면 아래 91 회수 절차로 복귀한다. 알려진 수치 실패는 승인된 실험 조건이다.
   자동 재시작을 반복하며 기다리거나
   CPU inference를 켜서 대체하지 않는다. startup memory/compile 시간은 별도 기록한다.

서로 다른 이름의 drop-in은 사전식 순서로 적용되므로 91이 90의 실행 경로를 대체한다.
[systemd drop-in 순서](https://github.com/systemd/systemd/blob/v255/man/systemd.unit.xml)

## 6. 91만 회수하여 영상 유지 모드로 복귀

운영자가 exact target을 확인한 다음
`/etc/systemd/system/nuv-agent.service.d/91-iq9075-visualad-htp-demo.conf`만
새로운 root 소유 evidence 하위 보관 경로로 **이동**한다. destination이 이미 있으면
덮어쓰지 않는다. 이어서 다음을 수행한다.

```sh
sudo systemctl daemon-reload
sudo systemctl reset-failed nuv-agent.service
sudo systemctl restart nuv-agent.service
sudo systemctl is-active nuv-agent.service
sudo systemctl show nuv-agent.service -p MainPID -p ExecStart -p DropInPaths -p SupplementaryGroups
```

- 91이 실제로 사라지고 90이 그대로 남았는지 확인한다. 이 복귀는 **현행 개발 영상 모드**로의
  복귀이며 signed baseline으로의 복귀가 아니다. 90까지 제거하는 작업은 별도 범위다.
- CPU reference opt-in이 꺼져 있는지, 영상 ICE/frame/segment가 다시 진행되는지 확인한다.
- code/model/deployment.env, HTP state, 실패 profile, event outbox를 삭제하지 않는다.
  service-only `fastrpc` 추가는 91 회수 후 새 프로세스에서 사라져야 한다.
- 종료 timeout/강제 종료가 있었으면 기록하고 graceful shutdown 성공으로 표시하지 않는다.
  검증 없이 연속 restart하거나 전체 drop-in 디렉터리를 정리하지 않는다.

전용 helper `packaging/dev/install-iq9075-visualad-htp-demo.sh`의 `stage SOURCE_SHA256`,
`activate`, `rollback`도 같은 경계를 지킨다. `rollback`은 91을 새 evidence 디렉터리로
이동하고 start-limit failure를 해제한 뒤 90 영상 모드로 복귀한다. 기존 CPU helper의
rollback은 91이 있으면 거절한다. HTP가 한번 실패하면 fail-closed sticky 상태이므로
원인/증거 확인 후 명시적 재시작 또는 rollback이 필요하다. 자동 CPU 전환은 없다.

현 배포는 source archive-001, launcher-002 이후 continuous-patch-003을 별도 검증·설치했다.
원 launcher/inventory는 보존하고 현 inventory는 `staged-files-003.sha256`다.
기존 stage/activate를 이 경로에서 재실행하여 덮어쓰지 않는다.

실측의 `inferenceSeconds`는 ORT session.run 시간, `processingSeconds`는 전/후처리
포함 시간(큐 대기·이벤트 전송 제외)이다. exp3은 sampling 지연 없이 실행하며 실측 약1.503 FPS다.
`lastResultAgeSeconds` 및 호환용 `sampleAgeSeconds`는 완료 시각 기준이며 capture age가
아니다. 연속 HTP queue는 1개 bounded latest-only이며 새 입력으로 대기 프레임을 교체한다.
초기 compile 후 대기 프레임은 버린다. frameArrivalAgeSeconds와 frameToResultSeconds는
해당 입력의 appsink 도착 시각 기준이다. 카메라 노출/브라우저 표시 latency와 혼동하지 않는다.

## 7. 시연 완료와 evidence 인계

변환 동등성 합격 뒤에도 실제 정상/불량 시료, 조명·거리·물체 조건, raw threshold,
오탐/미탐 및 반복 latency를 확인해야 한다. 실제 감지의 동일 `eventId`를 Agent outbox/ACK,
BE 저장, FE 상세까지 추적한다. snapshot/clip 업로드와 READY·재생, 새로고침·재연결,
91 회수 후 영상 복귀는 각각 별도 완료 증거가 필요하다. UI 수동 상태와 synthetic/재생
입력은 실제 감지 증거에서 제외한다.

인계에는 source/package/model inventory와 외부 승인 pin, graph SHA와 transform policy,
wheel reports/전체 freeze, CPU 변환 reference의 frame SHA, HTP profile/latency/RSS,
threshold calibration 상태, 미완료 항목 및 롤백 결과를 포함한다. checkpoint SHA가 같아도
export-001/002 그래프는 다르므로 checkpoint SHA만으로 실행 모델을 식별하지 않는다.

원 저자 코드·가중치의 별도 사용허락은 아직 확인되지 않았다. 이 경로의 private research
사용을 상용 재배포/정식 제품 탑재 승인으로 확대하지 않는다. 외부 checkout을 Agent source
bundle에 포함하지 않는다.
