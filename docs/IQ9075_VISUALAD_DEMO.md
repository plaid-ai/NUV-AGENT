# IQ9075 · VisualAD 목요일 시연

상태: 사용자 승인으로 수치 동등성 실패를 명시한 **NPU 전용 상시 실험 배포 완료**.
2초 sampling을 제거하고 최신 프레임 연속 추론으로 전환했다. Chrome 영상 표시 문제는 별도 미해결이다.
2026-09-07. 시연일 2026-09-10. Fleet Beta/정식 release 완료를 의미하지 않는다.
상관 티켓: NUV-436 / NUV-437 / NUV-439. Jira 상태는 변경하지 않았다.

## 최신 연속 추론 — exp3, 2026-09-07

사용자 추가 요구에 따라 `NUVION_ZERO_SHOT_SAMPLE_SEC=0`으로 바꾸었다.
single NPU worker가 연속 실행하며 대기 queue는 1개다. 새로운 프레임이 오면
대기 중인 오래된 프레임을 교체한다. 처리 중인 프레임의 입력은 변경하지 않는다.
다른 backend의 sampling 계약과 durable event/shutdown/safety 경계는 유지한다.

| 항목 | 실측 (07:58:12.310–07:59:12.319 UTC, 60.009초) |
| --- | --- |
| 서비스 | exp3 / PID 93966 / active / 자동 재시작 0 |
| 성공 추론 | 91회, 실제 steady cadence **1.503 FPS** (기존 약 3.03배) |
| 모델 실행 | 평균 616.59ms, p95 618ms |
| 전/후처리 포함 | 평균 664.74ms, p95 669ms |
| frame queue 대기 | 평균 15.29ms, p95 31ms, 최대 35ms |
| 앱 frame 도착→결과 | 평균 679.92ms, p95 695ms, 최대 705ms |
| 카메라 bridge | 29.956 FPS |
| 전체 Agent RSS / CPU | 최대 5.317GiB / 평균 1.390코어 상당 |
| 추론/HTP/카메라 오류 | 관측된 오류 0 |

이는 '최신 프레임 연속 추론'이며 **30 FPS 모든 프레임 검사**가 아니다.
앱 도착 시각은 sensor exposure 시각이 아니며, 위 지연은 브라우저 수신까지 포함하지 않는다.
모든 결과에 `mode=continuous_latest`, `queueWaitSeconds`, `frameToResultSeconds`와
원본001 graph / QNN/HTP provenance를 기록했다. telemetry에도 processingMode,
configuredSampleSeconds, frameArrivalAgeSeconds를 추가했다.

이번 raw score 범위는 −0.4628~−0.2647로 91회 모두 임계값 0 아래였다. 앞선 관측과
점수 분포가 달라 동일 장면/시료 조건이라고 가정하지 않는다. 정확도 비교·보정 결과가
아니며 `PARITY_FAILED` / `UNCALIBRATED`는 그대로다.

원본: [continuous-runtime-report-003.json](evidence/2026-09-07-visualad-htp/continuous-runtime-report-003.json),
SHA `64e3294034d5b44dcf5cdce285da0bcb2e1ac59ceefe44600ac84e4bf56d2fdd`.
patch archive SHA `13d8e3d81dc670e9bc5784d1986a3f124a2b93bbfaa51d993f18e23bb5af5a72`,
pipeline SHA `c7aa8997f4ef09db29a828703208a55a59535b9d7128d87962d37b83ed3f6afe`,
launcher SHA `744fd8854ea9c30e94eb2cb4dfb3304947a291b0bef1a158efd31cfea09e9610`,
inventory SHA `5bf4c25fefdf76a8d88b01d2b03b2cd0afda624867c4f2ab88b9c3c51be26f28`.
이전 exp2 파일은 보드 `evidence/continuous-patch-003/previous-*`에 보존했다.
현재 전체 회귀: 73개 모듈, **952개 중 946 PASS / 6 SKIP**, exit 0.

## 이전 2초 sampling 성능 — exp2, 2026-09-07

사용자 요청: "VisualAD 성능이 안나와도 상시 추론 배포해주고 성능 보고".
정식 정확도 합격이 아닌 private experimental deployment다. 기존 export-001을
그대로 사용하며 `PARITY_FAILED`/`UNCALIBRATED`/`experimental=true`를 유지한다.

| 항목 | 실측 |
| --- | --- |
| 서비스 | IQ9075 `nuv-agent.service`, enabled / active, PID 91038 |
| 실행 버전 | `0.1.121+iq9075.visualad.htp.exp2` |
| 관측 | 07:40:36.985–07:42:06.995 UTC, 90.009초, 실제 OAK 정상 빨간 케이블 |
| 성공 추론 | 45회, 관측된 HTP/추론 오류 0, PID 변경/자동 재시작 0 |
| NPU session.run | 평균 615.8ms, 중앙값 616ms, p95 617ms, 최대 618ms |
| 전·후처리 포함 | 평균 665.69ms, p95 672ms; 큐 대기·이벤트 전송 제외 |
| 실제 추론 빈도 | 0.496Hz, 약 2.015초 간격; 영상 30 FPS와 구분 |
| 카메라 bridge | 29.946 FPS, push counter 기준(Chrome 재생 FPS 아님) |
| 전체 Agent RSS | 평균 약 5.317GiB, 최대 5.318GiB |
| 전체 Agent CPU | 평균 1.338 코어 상당; 영상·네트워크·전/후처리 포함, learned CPU inference 아님 |
| raw anomaly score | 평균 −1.7210, 범위 −1.7970~−1.6613 |
| 현재 판정 | threshold 0.0에서 45/45 NORMAL, 관측 threshold 초과 0 |
| 기동 지연 | 07:38:22 시작→07:40:28 첫 결과, 약 126초; cold compile 포함 |

정상 시료 한 개의 상관된 반복 프레임이다. **정확도 100%라는 뜻이 아니다.**
손상 케이블이 없어 recall/미탐률/F1/AUROC를 계산할 수 없고 임계값도 보정하지 않았다.
0.616초의 역수 1.62를 실제 FPS로 보고하지 않는다. 현재 저속 관찰 시연은 가능하지만
2초 사이에 지나가는 결함을 모두 검사한다고 보장할 수 없다.

actual first-run ORT profile의 provider는 `QNNExecutionProvider`만 존재한다.
profile `visualad-htp-ort_2026-09-07_07-38-27_487.json`, 10,633 bytes,
SHA `dcf4a835a66406a564c5dbe71a2e3b3827225903a4d25d7b0df8e71fbd385741`.
실제 process maps의 QNN backend/Prepare/V73Stub/plugin은 모두 새 package bundle
하위이며 `libtorch` mapping은 없다. CPU/GPU model fallback은 활성화하지 않았다.

배포 후 07:43:33 UTC MP4를 decode하여 실제 빨간 케이블과
`EXP / UNCALIBRATED NORMAL -1.67` overlay를 확인했다. frame SHA
`afa1ba51c8ee6d7071a8274a1fd1d73349a59ae21242af17472e729a7202768a`.
원본 사진/segment는 private 보드 evidence에만 보관하며 저장소에 넣지 않는다.
Chrome은 새 페이지·팩토리 재선택·모달 재진입 후에도 0×0/readyState 0이다.
서버 uplink RTP 수신/producer와 Chrome consumer ICE/DTLS 연결은 확인했지만,
브라우저 decoded video는 미확인이다. 영상 표시까지 완료했다고 보고하지 않는다.

상시 서비스/재부팅 자동 시작은 설정했으나 이번에 물리 재부팅 시험은 하지 않았다.
HTP 실패는 fail-closed sticky 상태이므로 원인 확인 후 수동 재시작 또는 91 rollback이
필요하다. 정상 유지 시 anomaly event가 생성되지 않는 것은 기존 이벤트 정책이며
실제 DEFECT→eventId→ACK→BE/FE 증거는 아직 없다.

### 배포/검증 provenance

- source base `053bc34bacef80c99011770445bda20e22893a5b` + 로컬 개발 변경, commit/push 없음.
- source archive-001 SHA `7fca9f3b3d6a33300c1fa7402aa26539de3833989a0ada9d0053998c49de74c6`.
- 후속 launcher-002 SHA `f7ea633dd3a1f4b218f85a3f3681bb384675c570f9866cce1b674fa172a0fc90`.
- 현재 staged inventory SHA `86d29e829677daef325b0572d43836bf5aaea90b9e16d068118d83b4c17b18f2`.
- runtime module SHA `df5c4835af711eed6a818e59a9eb6fd03e84603a6f0f598afe58dcb56ffbbe08`.
- pipeline SHA `92b65643556dbb332fbab17c0ee3cc4bafada58ae883e88e9dc728bdcd7ce2af`.
- 원본 report: [always-on-runtime-report.json](evidence/2026-09-07-visualad-htp/always-on-runtime-report.json),
  SHA `c22c50b23bcd5638631377bc721b3863c431f458c667e5c713b522b24da4c0bb`.
- board evidence `/home/plaid/nuvion-demo/20260910-visualad-htp-probe/always-on-evidence.9FnmLRrO/`.
- 기존 signed pointer와 90 drop-in SHA 불변. root-owned 원 launcher/inventory 보존.
- 전체 73개 모듈 **948개: 942 PASS / 6 SKIP**, GI enum test mock 격리 후 exit 0.
  중간 native GI test crash 기록도 보존했으며 production HTP 실패와 혼동하지 않는다.

브라우저 문제의 read-only 서버 확인: uplink session `7a5d0c2f-dbd4-450e-bdd9-fff07b8449d9`
에서 07:40:42 H264 RTP forwarding, producer score 10을 확인했다. 현재 consumer는
paused=false / producerPaused=false / ICE completed / DTLS connected이며 score 6–8
피드백이 지속된다. 현재 session의 invalid-offer/RTP forwarding 오류는 없었다.
이는 브라우저 디코딩 성공 증거가 아니다. 상세 RTP/PLI counter API는 비활성화되어
조회하지 않았고 이를 위해 서버 권한·설정을 변경하지 않았다. Chrome inbound bytes와
decoded frames를 비교해 H264/keyframe 문제와 FE video 연결 문제를 분리해야 한다.

## 범위와 현재 결과

- 실제 IQ9075 + OAK-D Lite → on-board 추론 → durable anomaly event → BE → FE.
- 모델은 사용자가 지정한 [VisualAD](https://arxiv.org/abs/2603.07952).
- WinCLIP 비교 구현/격리 환경은 보존했으나 운영 Agent에 적용하지 않았다.
- 제품별 RPi5+DEEPX / Ventuno Q / Jetson Orin NX 실기 검증은 장비 확보 후 수행한다.
- 이번 개발 시연은 signed immutable Fleet release 승인/OTA 합격 증거가 아니다.
- 임의 DEFECT 주입, MVTec 파일 재생 또는 synthetic 입력을 실카메라 검증으로 세지 않는다.
- 실제 OAK 영상에 VisualAD HTP 실험 추론을 연결한다. CPU/GPU learned-model fallback은
  비활성화한다. 사용자 요청은 성능 미달에도 상시 실행하고 실측하는 것으로 변경됐다.
- `plaid-gpu-1`의 GPU 1에서 변환/reference 계산을 수행했으며 해당 작업은 종료했다.
  이번 배포에서는 추가 모델 변환·GPU 계산을 수행하지 않는다.

## HTP 전환 실측 — 아직 배포 합격 아님

518×518 / 원래 가중치 / `official_test_base` 정책을 유지한 단일 ONNX 그래프다.
ViT, SCA, SAF, cosine 연산을 HTP에서 실행하며 CPU는 이미지 전처리 및 학습되지 않은
bilinear/Gaussian/top-1% 후처리만 수행한다. CPU/GPU learned-model fallback은 금지한다.

| 검사 | 2026-09-07 실측 |
| --- | --- |
| Runtime | ONNX Runtime 1.26.0 + QNN EP 2.5.0 / bundled QAIRT 2.49.40 |
| 장치 | IQ9075 HTP v73, QNN NPU device 선택 |
| 실제 실행 provider | ORT trace `QNNExecutionProvider`만 존재, CPU node 없음 |
| 최초 graph compile | 124.13초 |
| 첫 추론 | 약 0.617초, 작은 evidence 배열 기록 비용 포함; steady latency 미측정 |
| peak RSS | 4.97GiB |
| raw score 오차 | 0.004956, 기준 0.01 이내 |
| patch map 평균 절대 오차 | 0.008179, 기준 0.01 이내 |
| patch map 최대 절대 오차 | **0.409958, 기준 0.1 초과 → FAIL** |
| CPU 비학습 후처리 자체 검증 | Torch 대비 map 최대 1.25e-6, score 최대 4.77e-7 |

이는 동일 실제 OAK 프레임에 대한 **변환 동등성** 검사다. 정상/이상 정답 시료가 아니므로
현장 검출 정확도를 뜻하지 않는다. 최대 오차 실패 후 원인을 분석했고, 이후 사용자 승인에
따라 원본 export-001을 실험 배포한다. 실패 기록이나 통과 기준을 변경하지 않는다.
층별 최대 오차는 6/12/18/24층에서 약 0.055/0.119/0.239/0.410이다.
실패 결과를 본 뒤 기준을 낮추지 않는다.

QNN의 `enable_htp_fp16_precision=0`은 현재 QAIRT에서 FP32 실행을 선택하지 않는다.
[공식 정정 PR #782](https://github.com/onnxruntime/onnxruntime-qnn/pull/782)에 따라
이를 FP32 우회 경로로 사용하거나 표시하지 않는다. 보드 시스템 QAIRT 2.46과
plugin bundle 2.49.40을 혼합하지 않는다.

후보/실측 식별자:

- ONNX `visualad.onnx`: 1,340,838,532 bytes,
  SHA-256 `1ad0cd1a216ba3bf89267e18bd7f7260bae48290067e3ab60d8cd8392fdeacb8`.
- manifest SHA-256 `89114e2159e4e6080971d5fcbe4782829daad78406f043eb877d603885fc4795`.
- full-htp-001 report SHA-256 `a20065519ee805bdc072dfbb6a6b33e6f70e8572fc90b2771ce5a982624253ac`.
- 보드 evidence: `/home/plaid/nuvion-demo/20260910-visualad-htp-probe/full-htp-001/`.
- 로컬 evidence: `/tmp/nuvion-visualad-fullhtp.3j1yRr/full-htp-001/` (임시 보관,
  정식 release evidence 저장소에 게시한 것이 아님).
- 원본 JSON/ORT profile의 영속 로컬 사본:
  [`evidence/2026-09-07-visualad-htp/`](evidence/2026-09-07-visualad-htp/).
  모델 가중치/ONNX/실카메라 사진은 저장소에 복사하지 않았다.

5-output 진단에서도 최종 patch map은 기존 HTP와 bit-exact다. 기준 특징 + HTP tokens는
최대 map 오차 0.00398, HTP 특징 + 기준 tokens는 0.41035이며, 마지막 cosine만
CPU FP32로 바꾸어도 0.41021이 남는다. 따라서 CPU 후처리 이동을 해결책으로 채택하지
않고 ViT 특징 계산의 수치 안정성을 조사한다. 진단 report SHA-256:
`6a8b01b2de329487be8b25de03c4725d8b85bd40e93461376f94bf4080f51bc4`.

Attention scaling 순서만 바꾼 두 번째 후보(`q×0.25`, `k×0.5`)도 실패했다.
ONNX CPU 출력은 기존 reference와 bit-exact지만, 실제 HTP에서는 순수 ORT 실행
0.6464초, score 오차 0.005847 / map 평균 오차 0.008238 / 최대 오차 0.418022였다.
같은 기준을 적용했으며 활성화하지 않았다. 이 후보는 manifest
`bd3872db91b04fc3c15f95f9d33dfc716f59129252ca8a50d591bf597b377ddf`,
graph `6571970e81b53b6ccffb36e08b75393f15337659f99e6d77aaa05985ef3154b2`로 구분한다.
두 실행 모두 참고 OAK 프레임 한 장의 변환 검사이며, 정상/이상 시료 검출 성적이 아니다.

실제 block14 입력을 사용한 LayerNorm 단독 비교도 수행했다. 명시적 centered two-pass
분해는 native LN 대비 최대 오차 0.01524→0.03031, 평균 오차 0.000230→0.000452,
실행 시간 0.00942→0.02043초로 악화했다. 이 방식의 full003은 만들지 않았다.
어느 방식도 CPU model fallback으로 우회하거나 실패한 정확도 기준을 낮추어 채택하지 않았다.

Agent HTP adapter는 manifest 외부 pin, 모델 SHA, 단일 embedded-weight 그래프,
descriptor-pinned 로딩, NPU-only device, CPU fallback 금지 및 실제 실행 profile을 검사한다.
모델 초기화/추론 실패를 NORMAL로 보내지 않고, 종료 후 늦은 결과도 보내지 않는다.
실제 1.34GB 그래프의 embedded-only 검사는 보드 ONNX 1.22.0 / protobuf 7.36.1에서
2.82초, peak 2.54GiB로 통과했다(모델 실행 없음). graph/manifest identity telemetry와
HTP 실패·종료·예열 회귀 검사를 포함한 최신 전체 테스트는 947개 중 941 pass / 6 skip,
73개 모듈 모두 통과했다. `graphSha256`으로 동일 checkpoint의 서로 다른 변환을 구분한다.
초기 compile은 `prepare()`에서 처리하고, compile 중 대기한 오래된 카메라 프레임 하나를
버린 후 새 프레임을 받는다. compile 완료만으로 `ready`/functional health를 올리지 않는다.
이 예열 변경의 최종 HTP module SHA-256은
`df5c4835af711eed6a818e59a9eb6fd03e84603a6f0f598afe58dcb56ffbbe08`이며 단위 테스트를 통과했다.
실측 full002는 그 이전 `b719077e…` snapshot으로 수행했으므로 최신 배포본 검사를 대신하지 않는다.

## 모델의 정확한 의미

VisualAD는 auxiliary 데이터에 이미 학습된 두 토큰과 SCA/SAF를 사용하는
target-domain zero-shot 방법이다. 학습 과정이 전혀 없는 모델은 아니다.
현장 물체 이름을 텍스트 프롬프트로 입력할 필요는 없다.

공식 기본값을 유지한다: ViT-L/14@336px visual backbone, 실제 입력 518×518,
중간 층 6/12/18/24, SCA anchors=4, SAF, 네 층 anomaly map 합,
Gaussian sigma=4, 상위 1% 픽셀 평균(공식 코드의 ceil 사용).

점수는 cosine 차이를 네 층에서 합친 **raw score**다. 확률이나 퍼센트가 아니며
음수도 정상적인 값이다. 기존 SigLIP의 0.7 임계값을 그대로 재사용하지 않는다.
초기 `NUVION_VISUALAD_THRESHOLD=0.0`은 미보정 시작값이며 실제 정상/이상 물체로
판정 경계를 확인해야 한다. 실제 검증 전 정확도나 불량 검출 성공을 주장하지 않는다.

공식 checkpoint에 학습된 `ln_post_weight/bias`가 있지만 공식 `test.py`는 이를
복원하지 않는다. 본 시연의 기본 정책은 `official_test_base`: 해당 tensor는 검증하되
적용하지 않고, 공식 평가처럼 base CLIP LN을 사용한다. 정책을 provenance에 기록한다.
학습 LN 적용은 점수가 바뀌는 별도 실험이며 자동으로 전환하지 않는다.

## 고정 출처

| 자원 | 식별자 |
| --- | --- |
| 공식 코드 | https://github.com/7HHHHH/VisualAD |
| 코드 commit | `97eb5f88a44f27c644ea7ed4ecac5e35ddef18bb` |
| VisA checkpoint | `weight/train_on_visa/CLIP.pth` |
| checkpoint 크기 / SHA-256 | `126119090` / `fed8ed5e0973e9adb53a2c91e066eb5ba3f9a42fbf80f115a26998629d1f0a5b` |
| backbone | `timm/vit_large_patch14_clip_336.openai` |
| backbone revision | `81e38efc4637de5023b10e75a7f9bd1c6fa6b010` |
| safetensors 크기 / SHA-256 | `1711828444` / `fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0` |

checkpoint 및 backbone 실제 크기/hash, CPU `torch.load(weights_only=True)`와
전체 VisualAD forward는 통과했다. CUDA FP32 기준과 decomposed ONNX CPU의 patch map
최대 오차는 6.25e-5로 1e-4 변환 기준을 통과했다. HTP 수치 검사는 위 별도 결과를 따른다.
unsafe pickle fallback, 자동 mutable weight 다운로드, 무작위 token 대체는 허용하지 않는다.

공식 VisualAD 저장소에서 LICENSE/COPYING 또는 별도의 코드·가중치 사용허락을
확인하지 못했다. 논문 CC BY-NC-ND를 코드 라이선스로 해석하지 않는다.
Agent 패키지에 소스를 vendor하지 않고 외부 checkout을 검증해 참조한다.
상용 포함·재배포·정식 릴리스는 별도 권한 확인 전 진행하지 않는다.

## 실행 및 복구 경계

- 준비 경로: `/home/plaid/nuvion-demo/20260910-visualad-probe`.
- CPU dependency 환경: `/home/plaid/nuvion-demo/20260910-model-probe/venv`.
- 이전 영상 전용 개발 경로: `/opt/nuvion-demo/20260910-visualad`.
- 현재 HTP 실험 경로: `/opt/nuvion-demo/20260910-visualad-htp`.
- HTP 격리 dependency/probe: `/home/plaid/nuvion-demo/20260910-visualad-htp-probe`.
- 원래 `/opt/nuv-agent/current` 및 signed release 파일은 변경하지 않는다.
- `packaging/dev/install-iq9075-visualad-demo.sh`는 stage/activate/rollback을 분리한다.
- 별도 systemd drop-in 하나로 실행 경로를 바꾸고, rollback은 그 파일을 보관 위치로
  옮긴 뒤 기존 서비스 실행 경로로 복귀한다. 모델/코드 파일을 삭제하지 않는다.
- 원래 Fleet settings journal과 demo settings를 분리하며 demo에서 Fleet 명령은 비활성화한다.
- CPU 검증 배포 `0.1.121+iq9075.visualad.dev1` → 영상 전용 `htp-prep1` →
  HTP 실험 `0.1.121+iq9075.visualad.htp.exp2`로 전환했다. BOM은 development / UNCONFIGURED다.
- 현행 drop-in rollback 코드가 존재하지만 이번 세션에서 원래 release 복귀를 실증하지 않았다.
- CPU 비활성화 restart에서 기존 DepthAI 프로세스가 정상 종료하지 않아 systemd가 종료했다.
  새 프로세스의 영상은 복구됐지만 해당 restart를 graceful shutdown 합격으로 세지 않는다.

## 영상 문제의 사전 증거

기존 실행 slot은 0.1.120이며 설치 Debian package 표기 0.1.121과 다르다.
STOMP login/CONNECTED는 성공하지만 uplink offer 직후 서버가 HTTP 400
`invalid-offer`로 session을 종료한다. 이후 ICE 404는 이 session 종료의 후속 증상이다.
로컬 성숙 MP4 segment도 846 bytes 헤더만 있어 ffmpeg가 프레임을 디코딩하지 못했다.

Chrome의 실제 `IQ9075-E2E-0901` 팩토리에서 대시보드와 상세 모달 모두
`영상 연결 준비 중`을 확인했다. 상세 모달에는 640px 높이의 내부 스크롤 영역이 있고
내용 높이는 1281px였다. 현재 영상 미표시는 스크롤 문제와 별개다.
기준 Agent source `053bc34bacef80c99011770445bda20e22893a5b`에는 canonical H264
offer 및 세션별 새 WebRTC branch 처리가 들어 있다. 해당 기반의 개발 배포 후 실제
SDP answer 및 ICE connected/completed를 확인했고, MP4는 실제 H264 프레임을 포함하며
ffmpeg decode에 성공했다. Chrome 상세 모달에서도 실제 OAK 화면을 확인했다.
Chrome video DOM은 640×480, `paused=false`, `readyState=4`였고 `currentTime`이
1877.795에서 1901.365초로 증가했다. 이는 영상 재생 증거이지 추론 활성화 증거가 아니다.
새로고침 후 회복은 아직 별도 완료 증거가 없다.

## 완료 기준

1. IQ9075에서 고정 모델과 실제 OAK 프레임으로 finite raw score를 반복 산출.
2. 실제 정상/이상 시료의 판정, latency 및 모델 provenance 기록.
3. 실제 판정으로 생성된 동일 eventId를 Agent outbox/ACK와 BE/FE 로그에서 확인.
4. Chrome 대시보드·상세 모달의 video frame 증가 및 실제 영상 확인.
5. 실제 이상 전환의 snapshot/clip 업로드, READY, 상세 재생 확인.
6. 새로고침·재연결 및 원래 Agent 복구 절차 기록.

## 케이블 시연 시나리오 — 정상 시료 관측

사용자 제안에 따라 1차 시연 대상을 **정상 케이블 vs 외피가 찢어지거나 벗겨진 케이블**로
잡는다. 사용자가 정상 케이블을 배치했고 2026-09-07 07:29 UTC Chrome에서 빨간
케이블 영상을 확인했다. 보드 segment 세 장도 시각/SHA와 함께 보존했다.
손상 시료는 아직 없으며 실제 검출 정확도는 평가하지 않았다.

- 정상 시료: 같은 종류·색상·굵기의 케이블 중 외피가 온전한 부분.
- 이상 시료: 외피 찢어짐/벗겨짐이 카메라 영상에서 선명하게 보이는 부분.
  내부 단선처럼 겉으로 보이지 않는 전기적 결함은 이번 영상 검사 범위에서 제외한다.
- 전원이 연결되지 않은 분리된 시료만 사용한다. 통전 케이블을 손상시키거나 만지지 않는다.
- 무늬 없는 배경, 고정된 카메라·조명·거리·배치에서 정상/이상 시료를 번갈아 놓는다.
  손과 도구를 치운 뒤 촬영하여 케이블 손상이 아닌 배경 차이를 판정 근거로 삼지 않도록 한다.
- 손상부가 영상에서 충분히 크게 보이는 상태부터 검증하며, 작은 균열의 검출 성능은
  별도 평가한다. 영상에 보이지 않는 손상의 검출을 약속하지 않는다.
- 정상과 이상을 여러 차례 교체하며 시료 ID/상태/촬영 조건을 기록한다. threshold를 고르는
  프레임과 최종 확인 프레임은 분리하고, 오탐/미탐도 보존한다. calibration용 데이터는
  모델 재학습 또는 NPU 변환 정확도 통과 증거로 혼용하지 않는다.
- 현장 검출 성능 확인과 FP32↔HTP 변환 동등성은 별도 검사다. 케이블 대상 선정만으로
  기존 HTP 최대 지도 오차 실패가 해소되지는 않는다. 상시 실험은 별도 사용자 승인이다.

## 남은 의사결정 및 검증 입력

- 정상 빨간 케이블은 사용자 확인 및 Chrome 관측을 완료했다. 손상 시료 배치·촬영이
  남아 있다. 한 정상 시료의 반복 프레임만으로 검출 정확도나 임계값 적합성을 평가하지 않는다.
- 현재 FP16 후보를 정확도 통과/정상 운영으로 표기하지 않는다. 다음 정밀도 개발에는
  대표 calibration 데이터와 HTP v73 operator별 정밀도 지원 검토가 필요하다.
  단순히 QDQ16을 넣는다고 모든 dynamic attention MatMul이 A16×A16로 실행되는 것은 아니다.
- 입력 크기 축소, 다른 모델 교체, 학습된 LN 적용, CPU/GPU 모델 fallback은 수행하지 않았다.
- 정상/이상 실제 판정 → 동일 eventId → ACK → BE/FE → 실제 clip READY 증거는 아직 없다.
- HTP runtime은 별도 `/opt/nuvion-demo/20260910-visualad-htp`에 stage 완료했다.
  91 활성화/실측 결과는 아래 최종 관측에 기록한다. 기존 90과 signed pointer는 보존한다.
- 이후 stage/검증/단일 drop-in 복귀 절차는
  [HTP 운영 runbook](IQ9075_VISUALAD_HTP_OPERATIONS.md)을 따른다. 두 사람 승인은 추가하지 않는다.
- commit/push, GCS artifact/BOM 게시, Jira 상태 변경은 이번 작업에서 하지 않았다.

주의: 이상 로그 목록의 기존 기본 날짜 범위는 지난달이다. 시연 당일을 포함한 날짜를
선택해야 한다. runtime ERROR를 모델 DEFECT로 오인하거나 UI 수동 상태 변경을
모델 실행 증거로 사용하지 않는다.
