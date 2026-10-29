# NUVION Pro demo qualification

`qualification_target=pro-demo` qualifies the Ventuno Q prototype demo stack for
manual arm64 package installation. It does not qualify IQ9075 OTA, CSI camera
capture, OpenRB motor control, or production anomaly-detection accuracy.
The existing `iq9075` target remains the default and retains its physical/Fleet
requirements. A Pro release cannot request IQ9075 OTA publication.

1. Build the signed release tag with the pinned arm64 bundle and DEB builders.
   Keep the exact source SHA, SOURCE_DATE_EPOCH, bundle hash and DEB hash/size.
2. Install that candidate DEB on Pro, preserving its QNN runtime, model files and
   explicitly recorded demo overlays. Confirm the installed slot/source identity.
3. Run `qualify-pro-demo.py --deb /path/to/package.deb --bundle-sha256 HASH
   --output evidence.json` as root on Pro. It switches through MVTec and both
   videos, observes each for 40 seconds, then restores video-1 and restarts the
   service. It sends no camera or motor commands. Run during a demo maintenance
   window; a connected presentation will change profiles.
4. Review all observations. Require advancing frame sequences, varying JPEGs,
   QNN/HTP inference within 1.7 seconds, at least three video loops, stable
   processes and restart recovery. The document records every deployed Python/UI
   file hash, including overlays; these results apply to that exact stack.
5. Sign the evidence with the approved Cloud KMS approval key. Record its file,
   signature, hashes and signer under `proDemoRuntime` in release-readiness.json,
   together with the exact live agent-release-gate check identifiers.
6. Merge the verifier/evidence PR, pin the reviewed publisher SHA, refresh and
   KMS-sign the settings audit for that publisher, and merge the audit update.
7. Dispatch release-publish.yml with the existing signed tag,
   `qualification_target=pro-demo`, `publish_iq9075_ota=false`. The publisher
   verifies the signed proof and live gate again at publication boundaries;
   rebuilt bundle and DEB must match the tested bytes exactly.
8. Verify the published package hash matches the installed candidate, inspect
   distribution promotion, and recheck the running Pro demo. State the limited
   qualification scope in release notes. A successful package publication is
   not evidence of camera, motor, or signed OTA validation on Pro.

Do not copy IQ9075 evidence to authorize Pro or mark an untested capability as
passed. The OpenRB UART timeout is a separate hardware/firmware investigation.
