# IQ9075 development OTA cutover to Cloud KMS

Current publisher: `candidate-publisher-v12`, Agent `0.1.121`, release sequence
`9`, schema `12`, minimum updater `0.2.0`. Sequences 2–8 and publishers v1–v11
remain retired, immutable evidence. The development device trusts the new
`release-iq9075-dev-kms-2026-09-v1` key and the previous verification key. The
production KMS OTA key is not part of this development keyring.

1. Merge the reviewed publisher, verify its complete CI gate, and record the
   exact main commit as `P`. Create the new tag using the production approval
   workflow; no Mac GPG key or passphrase is needed:

   ```sh
   gh workflow run kms-approve-release.yml --repo plaid-ai/NUV-AGENT --ref main \
     -f target_sha="$P" -f tag_name=candidate-publisher-v12 \
     -f tag_message='NUVION IQ9075 candidate publisher v12'
   ```

2. Verify the annotated tag's object, commit, and KMS OpenPGP signature. Add v12
   to the existing immutable candidate tag ruleset without changing old tags,
   removing update/deletion protections, or adding bypass actors. Update the
   existing candidate-sign/stage tag policies from v11 to v12 and verify them.

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
     --repo plaid-ai/NUV-AGENT --ref candidate-publisher-v12 \
     -f component_sha="$P" -f version=0.1.121 -f release_sequence=9
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
