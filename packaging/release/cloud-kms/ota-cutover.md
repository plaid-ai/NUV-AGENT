# IQ9075 development OTA cutover to Cloud KMS

Current publisher: `candidate-publisher-v11`, Agent `0.1.121`, release sequence
`8`, schema `12`, minimum updater `0.2.0`. Sequences 2–7 and publishers v1–v10
remain retired, immutable evidence. The development device trusts the new
`release-iq9075-dev-kms-2026-09-v1` key and the previous verification key. The
production KMS OTA key is not part of this development keyring.

1. Merge the reviewed publisher, verify its complete CI gate, and record the
   exact main commit as `P`. Create the new tag using the production approval
   workflow; no Mac GPG key or passphrase is needed:

   ```sh
   gh workflow run kms-approve-release.yml --repo plaid-ai/NUV-AGENT --ref main \
     -f target_sha="$P" -f tag_name=candidate-publisher-v11 \
     -f tag_message='NUVION IQ9075 candidate publisher v11'
   ```

2. Verify the annotated tag's object, commit, and KMS OpenPGP signature. Add v11
   to the existing immutable candidate tag ruleset without changing old tags,
   removing update/deletion protections, or adding bypass actors. Update the
   existing candidate-sign/stage tag policies from v10 to v11 and verify them.

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
     --repo plaid-ai/NUV-AGENT --ref candidate-publisher-v11 \
     -f component_sha="$P" -f version=0.1.121 -f release_sequence=8
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
