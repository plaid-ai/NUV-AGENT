from __future__ import annotations

import logging
import os
import time

from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime.model_guard import ensure_model_ready
from nuvion_app.runtime.triton_manager import ensure_triton_ready

log = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_ready(stage: str = "run") -> bool:
    if not _truthy(os.getenv("NUVION_RUNTIME_BOOTSTRAP_ENABLED"), default=True):
        return True

    backend = (os.getenv("NUVION_ZSAD_BACKEND", "triton") or "triton").strip().lower()
    if backend != "triton":
        return True

    max_retries = int(os.getenv("NUVION_BOOTSTRAP_MAX_RETRIES", "3"))
    base_backoff = float(os.getenv("NUVION_BOOTSTRAP_BACKOFF_SEC", "2"))

    for attempt in range(1, max_retries + 1):
        try:
            model_dir = ensure_model_ready(stage=stage)
            ensure_triton_ready(stage=stage, model_dir=model_dir)
            return True
        except BootstrapError as exc:
            log.warning(
                "[BOOTSTRAP] stage=%s attempt=%s/%s code=%s retryable=%s message=%s",
                stage,
                attempt,
                max_retries,
                exc.code,
                exc.retryable,
                str(exc),
            )
            if not exc.retryable or attempt >= max_retries:
                os.environ["NUVION_ZSAD_BACKEND"] = "none"
                return False
            sleep_sec = min(base_backoff * (2 ** (attempt - 1)), 30.0)
            time.sleep(sleep_sec)
        except Exception as exc:  # pragma: no cover
            log.warning(
                "[BOOTSTRAP] stage=%s attempt=%s/%s code=runtime_bootstrap_failed message=%s",
                stage,
                attempt,
                max_retries,
                str(exc),
            )
            if attempt >= max_retries:
                os.environ["NUVION_ZSAD_BACKEND"] = "none"
                return False
            sleep_sec = min(base_backoff * (2 ** (attempt - 1)), 30.0)
            time.sleep(sleep_sec)

    os.environ["NUVION_ZSAD_BACKEND"] = "none"
    return False
