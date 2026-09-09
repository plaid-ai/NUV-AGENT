# IQ9075 development OTA cutover to Cloud KMS

Current publisher: `candidate-publisher-v18`, Agent `0.1.121`, release sequence
`15`, schema `12`, minimum updater `0.2.0`. Sequences 2–14 and publishers v1–v17
remain retired, immutable evidence. The development device trusts the new
`release-iq9075-dev-kms-2026-09-v1` key and the previous verification key. The
production KMS OTA key is not part of this development keyring.

1. Merge the reviewed publisher, verify its complete CI gate, and record the
   exact main commit as `P`. Create the new tag using the production approval
   workflow; no Mac GPG key or passphrase is needed:

   ```sh
   gh workflow run kms-approve-release.yml --repo plaid-ai/NUV-AGENT --ref main \
     -f target_sha="$P" -f tag_name=candidate-publisher-v18 \
     -f tag_message='NUVION IQ9075 candidate publisher v18'
   ```

2. Verify the annotated tag's object, commit, and KMS OpenPGP signature. Add v18
   to the existing immutable candidate tag ruleset without changing old tags,
   removing update/deletion protections, or adding bypass actors. Update the
   existing candidate-sign/stage tag policies from v17 to v18 and verify them.

3. Review then apply the exact-workflow WIF plan:

   ```sh
   python3 packaging/release/provision-ota-kms.py --publisher-sha "$P"
   python3 packaging/release/provision-ota-kms.py --publisher-sha "$P" --apply
   ```

   Both providers pin `workflow_sha=P`, numeric repository and owner IDs,
   authorized actor IDs, explicit dispatch, exact environment subject and ref.
   A future workflow change requires an explicit new trust decision; this tool
   refuses to overwrite a different provider. The candidate job keeps its
   independent OIDC signature and `job_workflow_ref`/`job_workflow_sha` checks.

4. Dispatch the immutable candidate (no device update command is issued):

   ```sh
   gh workflow run iq9075-candidate-trusted-publish.yml \
     --repo plaid-ai/NUV-AGENT --ref candidate-publisher-v18 \
     -f component_sha="$P" -f version=0.1.121 -f release_sequence=15
   ```

   Preserve the complete run, canonical BOM, detached signature, artifact
   hashes, and GCS receipt. Failed attempts are not evidence of acceptance.

5. Deploy `trusted-release-keyrings/iq9075-dev.json` to the updater and the same
   flat `keys` map through BE's Helm `fleetReleaseTrust` ConfigMap. Preserve
   old public keys, command/health trust, updater state and rollback slots.
   Verify both old and new BOMs with the installed verifier. Reload only an
   idle updater and use BE's rolling deployment; do not interrupt a benchmark.

6. Register the new signed BOM in the BE catalog and verify its returned digest
   and publisher identity. Only after the replacement signing and verification
   path succeeds, remove `IQ9075_RELEASE_SIGNING_PRIVATE_KEY` from
   `iq9075-candidate-sign` and `iq9075-release`. Keep the old public key for
   rollback. Signing keys are never exported from KMS.

This cutover does not mark `0.1.121` READY. Physical automatic rollback and commit
acceptance are separate. Main-branch protection was disabled by the repository
owner; the formal release settings gate still requires its configured protection
and signed evidence. No successful settings/OTA attestation is implied here.
APT signing and GCS publishing credentials are outside this OTA key migration.

## Candidate v12 physical validation retry

The v11 sequence-8 attempt on IQ9075 (command
`9eb50d02-3892-4a3f-95b1-48724bb4d54b`) stopped before the OAK fault journal
or USB write. The updater selected the candidate slot before the Agent restart
completed, invalidating the host's readiness snapshot. The boot watchdog later
restored the signed 0.1.120 slot. This is failed diagnostic evidence; C was not
issued and no physical acceptance is claimed for that chain.

The board now checks the candidate process's slot, PID/start time and boot ID,
then confirms its PID after the USB check. A transient preflight returns an
explicit unarmed result before any fault journal, deadman or USB mutation.
Only that exact result may retry under the original host deadline. Response
loss, a recorded fault, identity mismatch and recovery failure still abort.
Sequence 9 and publisher v12 require fresh bootstrap/R/C evidence. Their WIF
providers are `candidate-v12` and `release-main-v12`; previous providers and
immutable artifacts remain unchanged.

## Candidate v13 RTP statistics compatibility

The sequence-9 commit attempt reached FUNCTIONAL_HEALTHY but timed out before
opening a commit gate. IQ9075 GStreamer returns WebRTCStatsType enums, whose
string representation contains underscores. Normalize their value_nick so
actual increasing packet/byte counters are recognized. Keep the connected
ICE, fresh camera, current-process STOMP, two RTP samples, and health
attestation requirements unchanged. Publisher v13 signs sequence 10 through
new exact-workflow candidate-v13 and release-main-v13 WIF providers. Previous
failed evidence, tags, providers, and artifact bytes remain preserved.

The host rollback poll also waits through PAUSED_HEALTH_UNKNOWN while the
exact terminal ACK is awaiting its derived projection. Missing or malformed
final rollback evidence is rejected explicitly. All final evidence predicates
remain unchanged; a paused rollout alone never constitutes success.

## Candidate v14 health attestation sampling

Sequence 10 rollback passed including the BE projection and exact cleanup.
The normal commit command `adbe754a-9411-4235-b27c-9d6365f18c2a` opened its
root-owned commit gate but the generic effect retry backoff reached 32 seconds.
BE requires continuous healthy samples with no gap above 15 seconds, so each
retry reset the soak. The root watchdog automatically restored 0.1.120 after
COMMIT_TIMEOUT; this chain is failed evidence, not acceptance.

Only AGENT_UPDATE in FUNCTIONAL_HEALTHY now caps its durable retry delay at
5 seconds. Startup retries and unrelated effects retain their backoff. The
30..60-second soak, fresh heartbeat/RTP progress, current process identity,
root commit gate, attestation signature and commit deadline remain mandatory.
Publisher v14 signs sequence 11 with new exact-workflow candidate-v14 and
release-main-v14 providers. Sequence 10 and all prior failed artifacts remain
immutable. New complete rollback and commit evidence is required.

## Candidate v15 startup and heartbeat prerequisites

Sequence 11 rollback passed; commit requested attestation every five seconds,
but the deployed heartbeat interval was still 30 seconds. Its command
`3a5ec88b-7035-477f-a622-c6bb7622fc28` therefore failed the unchanged 15-second
sample-gap check and automatically rolled back. IQ9075's supported
NUVION_DEVICE_STATE_INTERVAL_SEC setting was changed to 5 with a root-owned
backup; existing Agent reconciliation reported the actual terminal state.
The runtime default and template now also use 5 seconds. Existing explicit
30-second device settings must be migrated before attempting commit.

After the consistent backup restarts the Agent, pre-trust foundation checks
wait at most 30 seconds for transient OAK enumeration/binding recovery.
Unsafe paths and other foundation failures still abort immediately. No trust,
USB fault, or runtime mutation is performed by the wait.

BE HALT may label a queued command EXPIRED before its signed TTL. Preserve that
terminal history but exclude it from elapsed-expiry proof. Config/stream still
requires at least one predecessor whose actual signed deadline passed, adjacent
rollback/commit, a drained queue, and exact final restoration. No timestamp,
command, ACK, or old artifact is rewritten. New v15/sequence12 evidence is required.

## Candidate v16 final stream-observation drain

Sequence 12 physically passed automatic OAK-fault rollback and normal commit.
CONFIG/stream preparation then stopped the Agent after a drained queue, but
shutdown enqueued one final streaming observation after its delivery worker
stopped. The attempt restored without configuration mutation. Preserve both
failed preparation journals and the successful OTA evidence.

The preparation now waits for the running worker to drain before shutdown,
retains any final observation generated by shutdown, and waits for the restarted
worker to deliver it before returning prepared evidence. Pending command work,
reservations, DLQ rows, or a 30-second drain timeout still abort. No observation
ACK or lifecycle record is fabricated. New v16/sequence13 proof uses the signed
sequence12 runtime as its last known good rollback baseline.

## Candidate v17 bounded updater probe retry

The sequence-13 rollback harness stopped before recording or applying an OAK
fault when its authenticated updater status probe timed out. The device later
automatically rolled back to the already committed sequence-12 slot. This
failed attempt does not qualify sequence 13 for release.

The updater's functional probe can occupy its single RPC server. Both status
reads now produce the same unavailable result on socket timeout. Only an
unavailable pre-fault read is retryable under the existing host deadline;
incorrect authentication, version or command identity still abort immediately.
No fault journal, deadman or USB write occurs until the authenticated identity
and current candidate process are confirmed. Publisher v17/sequence14 requires
fresh complete physical evidence. Earlier tags, artifacts and results remain
immutable.

## Candidate v18 committed physical rollback baseline

The device committed sequence 12 after physical automatic rollback and normal
commit passed (commands 22a7199c-a889-4fea-b021-e6cabb357439 and
4b9313a4-ee4a-48df-9677-1fc35f0f9a5d). Its anti-rollback floor is now 12.
The full Fleet Runtime validator previously required the legacy promoted
sequence-1 baseline even on this device, making a new complete qualification
impossible without an inappropriate floor reset.

The signed release policy now explicitly allowlists the exact sequence-12 BOM
as an additional physical rollback baseline. Both run manifests, root markers,
bootstrap and cleanup must match the same pinned version, sequence and digest;
the candidate must have a higher sequence. Legacy promotion remains sequence 1
and publication history is unchanged. Sequence 12 is a known-good physical
baseline, not a claim of complete release qualification.

Publisher v17 was cancelled before staging or device commands. Its immutable
tag and sequence14 remain retired. Publisher v18/sequence15 carries the same
tested timeout fix plus this strict baseline selection for fresh qualification.
