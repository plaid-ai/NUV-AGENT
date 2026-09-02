# NUV-439 combined-branch integration contract

The OTA branch predates `FleetEffectCoordinator`. When merging it with the
NUV-436/437 integration branch, preserve these boundaries:

1. Import `configure_agent_update_reconciler` in `inference/pipeline.py`.
2. Immediately before `build_fleet_command_runtime_from_env(...,
   reconciler_registry=fleet_effect_registry)`, call
   `configure_agent_update_reconciler(fleet_effect_registry)`.
3. Do not place `AgentUpdateReconciler` in
   `DurableCommandProcessor.handlers`. The combined runtime must continue to
   derive command handlers from `DurableReconcileStore.stage_verified` and
   registry command types.
4. `AgentUpdateReconciler.reconcile()` performs Unix-socket/download/service
   I/O only after `FleetEffectCoordinator` has released every SQLite
   transaction. Its restart paths return the shared `ReconcileDeferred` type.
5. If the socket is absent, unsafe, connected peer UID is not zero, or helper
   `STATUS.capabilityAvailable` is false, unregister `AGENT_UPDATE`; do not
   advertise `command.agent.update`.
6. Merge `build_updater_capability_telemetry()` into every fresh heartbeat so
   the disabled reason remains observable. Re-run
   `configure_agent_update_reconciler()` on a bounded periodic cadence (and on
   reconnect), not only at process startup, so a socket-activation race can
   self-heal and an unsafe/disappeared helper is promptly unregistered.
7. Resolve `packaging/systemd/nuv-agent.service` to the slot-aware version in
   this branch. Preserve the foundation branch's settings boot guard,
   bootstrap guard, `StartLimit*`, and restart policy, but execute both guards
   with `/opt/nuv-agent/current/venv/bin/python`; the legacy
   `/opt/nuv-agent/venv/bin/python` path does not exist after immutable-slot
   packaging. The combined source and every published `agent-bundle` must
   contain `nuvion_app.runtime.settings_boot_guard` before release.
8. Do not treat every `ReconcileDeferred` as a process restart. OTA checkpoints
   explicitly carry `restartRequired` and `nextAction`. For
   `nextAction=RESTART_AGENT`, persist `WAITING_RESTART` and call the supervisor
   restart requester. For `nextAction=RETRY_EFFECT`, release the effect lease
   back to `PENDING` (with bounded backoff) and do not call `restart_requester`.
   Add a combined-branch coordinator test proving `UPDATER_UNAVAILABLE` and
   `DOWNLOAD_FAILED` cannot create a restart loop.
9. A periodic helper readiness change must update the whole command admission
   snapshot, not only the reconciler registry. Atomically rebuild/swap
   `FleetCommandVerifier.capabilities`, command handlers, registry entries and
   advertised capabilities together (or request one bounded supervisor restart
   on false-to-true). Test startup-unavailable then helper-ready without leaving
   `AGENT_UPDATE` permanently rejected as `MISSING_CAPABILITY`.
10. `updaterVersion` is live root-helper evidence, never build/BOM/config
    metadata. Static `build_runtime_telemetry()` must publish `unknown`.
    `merge_runtime_public_state()` may accept a semantic version only alongside
    the same fresh `agentUpdate` object with `authenticatedHelper=true`; unsafe,
    absent or dead sockets force `unknown`. BE eligibility must require
    `capabilityAvailable=true`, `authenticatedHelper=true`, and
    `updaterVersion >= minUpdaterVersion`. Add combined Agent/BE contract tests
    for missing helper, non-root peer, stale static version and recovery.
11. Keep the runtime public-state allowlist explicit. In addition to the
    existing health/update evidence keys, allow only `agentUpdate` and
    `updaterVersion` through the authenticated adapter; reject unknown fields,
    non-object capability state, non-boolean auth/readiness and non-semver
    versions before promoting them to flat or nested runtime telemetry.

The terminal `SUCCEEDED` state is rejected unless it contains the complete
desired payload plus `agentVersion`, `artifactDigest`, full `componentSha`,
`configSchema`, integer `releaseSequence`, `bomVerificationStatus=VERIFIED`,
`health=FUNCTIONAL_HEALTHY`, canonical `functionalHealth`, `slot`, and
`previousVersion`.

The current bootstrap artifact is deliberately `IQ9075_DEV` base-only. `full`
and `runtime` installation profiles remain fail-closed until each has an
independent hash-locked immutable bundle and a privileged Docker/Triton helper.

## Residual work before production SKU rollout

- Bound/compact terminal updater command history while preserving a separate
  monotonic replay sequence; the current download/slot caches are bounded, but
  the root SQLite journal still retains terminal compact JWS history.
- Run the exact signed IQ9075_DEV bundle through real QCS9075-EVK + OAK-D Lite
  boot/functional failure and power-loss tests. macOS simulation is not hardware
  evidence.
- Complete the external GitHub settings prerequisite from the v0.1.121
  runbook: protected-main/tag rulesets, exact `agent-release-gate` integration,
  immutable releases, one CODEOWNER approval for general writers with the exact
  Platform-Admin single-admin PR bypass, reviewer-free exact-main environments,
  environment-scoped secrets, trusted publisher SHA, and a fresh signed settings
  attestation. Face artifact publication additionally requires an allowlisted
  Platform-Admin signature over the exact tag, component, model/channel tuple,
  and each artifact digest/size; immutable GitHub release state by itself is not
  treated as provenance.
