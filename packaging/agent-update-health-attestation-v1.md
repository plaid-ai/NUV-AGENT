# Agent update health attestation v1

This contract is the final commit boundary for a staged Agent release. The
privileged updater never commits from Agent-local readiness alone.
It is implemented by updater protocol version `0.2.0`; release BOMs that depend
on this boundary must declare `minUpdaterVersion` of at least `0.2.0`.

## Trust material

The updater reads
`/etc/nuvion-updater/health-attestation-keyring.json` as a root-owned,
non-writable regular file:

```json
{
  "schemaVersion": 1,
  "trustDomain": "production",
  "purpose": "agent-update-health-attestation",
  "keys": {"health-2026-01": "<canonical standard-base64 Ed25519 public key>"}
}
```

This keyring is intentionally separate from Fleet-command and release-BOM
keys. A device with no valid health-attestation keyring keeps the updater
socket disabled.

## Local commit gate

After the root-owned physical functional probe reaches `FUNCTIONAL_HEALTHY`,
the candidate Agent calls:

```json
{
  "schemaVersion": 1,
  "operation": "BEGIN_COMMIT_GATE",
  "commandId": "<canonical UUID>"
}
```

The updater authenticates the Unix caller with Linux `SO_PEERCRED` and requires
its pid to equal systemd's live `nuv-agent.service` `MainPID`. It reads the
process start ticks from `/proc/<pid>/stat`, the kernel boot id, and the
`NUVION_ACTIVE_SLOT` process environment. Those values, the active candidate
slot, command id, BOM digest, component SHA, release sequence, and the existing
absolute commit deadline are durably bound to one gate.

The response is:

```json
{
  "schemaVersion": 1,
  "gateId": "<canonical UUID>",
  "challenge": "<43-char canonical unpadded base64url>",
  "commandId": "<canonical UUID>",
  "bomDigest": "sha256:<64 lowercase hex>",
  "componentSha": "<40 or 64 lowercase hex>",
  "releaseSequence": 2,
  "candidateSlot": "releases/<64 lowercase hex>",
  "agentPid": 412,
  "agentStartTicks": 123456,
  "bootId": "<canonical UUID>",
  "expiresAt": "<the existing absolute commit deadline>"
}
```

`challenge` is exactly 32 CSPRNG bytes encoded as canonical unpadded base64url,
therefore exactly 43 ASCII characters. Repeating `BEGIN_COMMIT_GATE` from the
same process returns the same gate and does not rotate the challenge or extend
the deadline. A different pid, process instance, boot, slot, or release identity
is rejected.

## BE issuer API

The authenticated candidate sends the following exact request to
`POST /devices/me/agent-update-health-attestations`:

```json
{
  "deviceId": "<device id>",
  "commandId": "<canonical UUID>",
  "gateId": "<canonical UUID>",
  "challenge": "<43-char canonical unpadded base64url>",
  "bomDigest": "sha256:<64 lowercase hex>",
  "componentSha": "<40 or 64 lowercase hex>",
  "releaseSequence": 2,
  "productModel": "NUVION",
  "platformProfile": "rpi5_deepx_dx_m1",
  "hardwareRevision": "REV_A",
  "architecture": "arm64"
}
```

The BE MUST authenticate the device, exact-match its ownership and immutable
release identity, and issue only after fresh server-observed runtime evidence
passes the rollout policy. At minimum this includes the current Agent process's
STOMP heartbeat, live WebRTC connectivity with increasing outbound samples,
pipeline progress, and healthy event/command outboxes. The BE MUST NOT accept a
caller-supplied health verdict.

The response is:

```json
{
  "keyId": "health-2026-01",
  "issuedAt": "2026-09-02T10:00:00Z",
  "expiresAt": "2026-09-02T10:00:30Z",
  "compactJws": "<protected>.<claims>.<signature>"
}
```

The protected header contains exactly:

```json
{"alg":"EdDSA","kid":"health-2026-01","typ":"nuvion-update-health+jws"}
```

The signed claims contain exactly:

```json
{
  "schemaVersion": 1,
  "jti": "<canonical UUID>",
  "aud": "nuvion-updater",
  "purpose": "agent-update-commit",
  "trustDomain": "production",
  "gateId": "<canonical UUID>",
  "challenge": "<43-char canonical unpadded base64url>",
  "deviceId": "<device id>",
  "commandId": "<canonical UUID>",
  "bomDigest": "sha256:<64 lowercase hex>",
  "componentSha": "<40 or 64 lowercase hex>",
  "releaseSequence": 2,
  "productModel": "NUVION",
  "platformProfile": "rpi5_deepx_dx_m1",
  "hardwareRevision": "REV_A",
  "architecture": "arm64",
  "health": "HEALTHY",
  "issuedAt": "2026-09-02T10:00:00Z",
  "expiresAt": "2026-09-02T10:00:30Z"
}
```

The TTL is positive and at most 60 seconds. Times are RFC3339 UTC. The updater
permits at most five seconds of clock skew.

## Commit request and replay rule

```json
{
  "schemaVersion": 1,
  "operation": "COMMIT",
  "commandId": "<canonical UUID>",
  "gateId": "<canonical UUID>",
  "healthAttestationJws": "<compact JWS>"
}
```

The updater rechecks `SO_PEERCRED`, `MainPID`, process start ticks, boot id, and
active slot both before and after signature validation. It then consumes the
gate, JWS SHA-256, and `jti` in the same SQLite transaction that advances the
release sequence/BOM identity and phase to `COMMITTED`. An exact retransmission
of the same request is idempotent. A different JWS or reused `jti` is rejected.
The persisted watchdog deadline is never extended; expiry follows the existing
`COMMIT_TIMEOUT` rollback path.
