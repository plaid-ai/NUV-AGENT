# IQ9075 controlled OTA startup failure (2026-09-08)

The signed candidate built from `1f69427eabd91bb2deb6e3bf8413073ed91e164a`
reached VERIFIED and ACTIVATING on IQ9075, but did not reach the intended OAK
fault window. Command `bb5baf03-b1a8-477d-b09d-f8bb65c70130`, sequence 14,
ended locally in ROLLBACK_FAILED at 08:01:57 UTC. Its original signed BOM is
`sha256:c86286d97dee8d5c7cb77424c646366447eb838a2d93a2809bb7f1eb85826171`.
This attempt is not promotable physical release evidence.

Two separate startup conditions caused the failure:

1. Successful service starts also consume systemd's StartLimitBurst. Resetting
   only units already marked failed left the next controlled OTA restart able
   to hit the limit. The updater now resets before every controlled restart.
   A failed reset is tolerated only when a successful LoadState query proves
   the unit is not yet loaded; other failures stop the service.
2. A Type=simple start can return before the shell launcher execs the Agent.
   An immediate `/proc/<MainPID>/environ` read can therefore lack the launcher's
   NUVION_ACTIVE_SLOT. The updater now waits for the exact slot and stable PID
   after activation, rollback restart, and the OAK functional probe restart.
   Missing startup identity gets a bounded wait; wrong or duplicate identity
   remains an immediate failure. Timeout stops the service.

The same-ID cleanup completed, released the recovery lease, and preserved the
original updater history and sequence counters. The prior signed current slot
`26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1` was retained.
The experimental operating runtime was restored separately after cleanup.

The registered 0.1.121/sequence-2 candidate is immutable. Do not replace its
registered BOM, reuse its failed run as a successful rollback, or sign READY
using a patched helper against that old candidate. A subsequent candidate must
bind the corrected component and updater package and repeat the complete
rollback, commit, config/stream, restoration, and release-gate chain.
