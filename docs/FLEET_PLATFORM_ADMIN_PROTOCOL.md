# Fleet PLATFORM_ADMIN 및 모델 rollout capability

이 문서는 Agent **소스 지원 계약**이다. 운영 key provisioning, Agent 배포,
서버 명령 발행이나 실제 Fleet rollout 완료를 의미하지 않는다. 별도 IQ9075
VisualAD 실험 runtime은 이 worktree의 변경 대상이 아니며 Fleet command 비활성
상태를 이 기능 지원/배포의 증거로 사용하지 않는다.

## 서명 및 권한 경계

기본 verifier가 허용하는 `authorizationContext`는 `SPACE_ADMIN`과
`PLATFORM_ADMIN`이다. 플랫폼 관리자가 여러 space를 관리하더라도 각 명령의
signed `deviceId`와 `spaceId`는 **수신 장치의 실제 값**이어야 한다.
`PLATFORM_ADMIN`은 device/space 검사를 생략하는 wildcard가 아니다.

두 context 모두 기존 검증 경로를 그대로 사용한다.

1. 신뢰 domain에 맞는 public keyring과 파일 소유/쓰기 권한을 검증한다.
   생산은 `production`, macOS 개발은 `macos-dev`, IQ9075 개발은 `iq9075-dev`다.
2. 고정된 `kid`의 Ed25519 key로 EdDSA JWS를 검증한다. 원격 key header,
   unsigned/local 명령 또는 context 문자열만으로 권한을 부여하지 않는다.
3. 정확한 device/space, canonical claims, payload SHA, schema와 TTL을 검증한다.
   최대 TTL 24시간 등 기존 제한은 변경하지 않는다.
4. 허용 context 및 해당 명령의 live effect capability를 검증한다.
5. 동일 durable inbox의 단조 증가 sequence로 재생을 차단한다. 두 context가
   별도 sequence 공간을 갖거나 cursor를 초기화하지 않는다.

BE의 platform actor는 `admin::<uuid>`이며, Agent는 서명된 actor를 그대로 보존한다.
actor 텍스트는 독립적인 인증 수단이 아니다. 알 수 없는 context는 기존
`UNSUPPORTED_AUTHORIZATION_CONTEXT`로 거부되며, 인증에 실패한 입력이
정책 거절 ACK로 변환되어 cursor를 전진시키는 경로도 변경하지 않는다.

## 설정과 opt-out

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `NUVION_FLEET_COMMAND_ENABLED` | `false` | 전체 signed Fleet runtime 활성화 여부 |
| `NUVION_FLEET_PLATFORM_ADMIN_ENABLED` | `true` | 활성 Fleet runtime의 `PLATFORM_ADMIN` admission 허용 여부 |

플랫폼 관리자 설정은 true 값 `1/true/yes/on`, false 값 `0/false/no/off`를
대소문자 구분 없이 받는다. 전체 Fleet runtime이 활성인 경우 빈 문자열이나
오타는 설정 오류로 처리되어 runtime 초기화에 실패한다. `false`로 설정하면
`SPACE_ADMIN`은 유지하면서 새 `PLATFORM_ADMIN` 명령은 거부한다.

opt-out은 새로운 명령의 admission 정책이다. 이미 검증되어 durable inbox에
수락된 명령의 기존 crash recovery·ACK replay를 소급 취소하지는 않는다.
운영 설정 변경·재시작·key provisioning은 별도 승인과 배포 절차가 필요하다.

## 동적 광고 조건

### `fleet.auth.platform_admin.v1`

domain-bound keyring 로딩과 inbox identity binding을 통과해 실제
`FleetCommandRuntime`이 구성되고, 그 runtime의 동일한 신뢰 keyring/verifier가
현재 `PLATFORM_ADMIN`을 허용할 때만 광고한다. 초기화 전·비활성·키 누락·
잘못된 trust domain/권한·초기화 실패·opt-out·verifier 교체 상태에서는 광고하지 않는다.

static platform profile이나 일반 effect 목록에 이 문자열을 넣어도 광고되지 않는다.
pipeline이 매 heartbeat에 실제 runtime 증거로만 합성하며,
`capabilities`와 `runtimeTelemetry.capabilities`가 함께 갱신/철회된다.

BE는 새 platform 명령을 발행하기 전에 대상이 ONLINE인지와, 원본의 최신
heartbeat(최대 120초)에 이 capability가 포함되는지 확인한다. 이전 snapshot의
capability 상속이나 운영자가 입력한 profile만으로 지원을 가정하지 않는다.

### `command.config.model.siglip.v1`

다음 조건을 모두 만족할 때만 광고한다.

- 구성된 trusted Fleet runtime와 live `command.config.apply` effect가 있다.
- 등록된 실제 `SettingsReconciler`가 현재 pipeline의
  `PipelineSettingsRuntimeAdapter`를 사용한다.
- SigLIP가 enabled/ready이며 pipeline이 실행 중이다.
- `loaded_model_source()`가 현재 설정된 model directory와 정확히 일치한다.
- `can_verify_model()`과 `verify_model()`이 실제 callable이다.

미로딩·다른/이전 model source·원격 모델 식별자만 있는 상태·중지된 pipeline·
다른 backend(`none`, Triton, VisualAD 포함)·Fleet 비활성 상태에서는 광고하지 않는다.
이 capability도 static profile이나 일반 effect 문자열로 추가할 수 없다.

이는 **검증 기능의 가용성**이며 특정 target model의 다운로드·적용·정확도·
전체 파일 SHA 검증 완료 증거가 아니다. heartbeat에서는 대형 weights를 해시하지 않는다.
모델 변경은 기존 `CONFIG_APPLY` schema v1의 `model`과 `activation=RESTART`를 사용하며,
실제 적용 성공은 재시작 후 기존 `verify_model()`의 signed pointer/digest 및
다운로드된 전체 artifact bytes 검증과 functional health를 통과해야 한다.
BE의 platform 모델 dispatch에는 authorization capability와 이 모델 capability를
각각 최신 heartbeat에서 확인해야 한다.

모델 artifact identity는 startup telemetry의 기존 전체 bytes 검증 한 번에서
`modelDigest`(manifest 우선, 없으면 aggregate)와 `modelAggregateDigest`를 함께
기록한다. 두 digest는 설정이나 resolver 선언값이 아닌 검증값이며, 검증 실패 시
`unknown`이다. `modelObservedPointer`도 같은 bytes 검증이 성공한 로컬 resolver
metadata의 pointer이고, `modelPointer`는 설정값이므로 관측 증거로 대체할 수 없다.
BE의 완료 판정은 같은 fresh heartbeat의 `modelObservedPointer`가 target과 같고,
target digest가 두 검증 digest 중 하나와 일치해야 한다. 이 조건은 exact signed
payload/command/sequence ACK, `SUCCEEDED`/`CONVERGED`, RUNNING 및 capability
검사를 대체하지 않는다. `modelExpectedDigest`/`modelResolverDigest`는 참고값이며
artifact 검증 근거로 사용하지 않는다. 기존 `modelDigestMatchesExpected`는
`modelDigest` 단일 비교 의미를 유지하므로 aggregate target의 완료 gate가 아니다.

## 소스 검증 범위

합성 Ed25519 서명으로 정상 platform 명령, device/space mismatch, 서명·payload
변조, 미등록 key/context, TTL/schema/capability 위반, 양 context 간 sequence
재생 차단을 검증한다. keyring 실패·opt-out·초기화 전·runtime property 실패의
capability 미광고와 heartbeat flat/nested 철회도 단위 테스트한다.
SigLIP unloaded/old source/backend mismatch 및 heartbeat 비해시 조건을 별도로 검증한다.

단위 테스트 통과는 실제 key provisioning, 서명된 운영 명령의 effect/ACK,
모델 rollout 또는 하드웨어 동작 확인을 대신하지 않는다.
