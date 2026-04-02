from __future__ import annotations

import logging
import os
import sys
import time

from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime.inference_mode import face_tracking_uses_triton, normalize_backend
from nuvion_app.runtime.model_guard import ensure_model_ready
from nuvion_app.runtime.triton_manager import ensure_triton_ready

log = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emit_progress(message: str) -> None:
    # Bootstrap runs before pipeline logging is initialized, so print directly.
    sys.stderr.write(f"[BOOTSTRAP] {message}\n")
    sys.stderr.flush()


def ensure_ready(stage: str = "run") -> bool:
    if not _truthy(os.getenv("NUVION_RUNTIME_BOOTSTRAP_ENABLED"), default=True):
        return True

    backend = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")
    if backend != "triton" and not face_tracking_uses_triton():
        return True

    max_retries = int(os.getenv("NUVION_BOOTSTRAP_MAX_RETRIES", "3"))
    base_backoff = float(os.getenv("NUVION_BOOTSTRAP_BACKOFF_SEC", "2"))
    _emit_progress(f"stage={stage} 시작: 모델/Triton 준비를 점검합니다.")

    for attempt in range(1, max_retries + 1):
        try:
            _emit_progress(f"stage={stage} attempt={attempt}/{max_retries} 모델 점검 중...")
            model_dir = ensure_model_ready(stage=stage)
            _emit_progress(f"stage={stage} attempt={attempt}/{max_retries} Triton 점검 중...")
            ensure_triton_ready(stage=stage, model_dir=model_dir)
            _emit_progress(f"stage={stage} 완료: runtime 준비 완료")
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
            _emit_progress(
                f"stage={stage} attempt={attempt}/{max_retries} 실패: "
                f"code={exc.code} retryable={exc.retryable} message={exc}"
            )
            if not exc.retryable or attempt >= max_retries:
                os.environ["NUVION_ZSAD_BACKEND"] = "none"
                _emit_progress("최종 실패: 추론 backend를 none으로 강등하고 서비스는 계속 유지합니다.")
                return False
            sleep_sec = min(base_backoff * (2 ** (attempt - 1)), 30.0)
            _emit_progress(f"{sleep_sec:.1f}s 후 재시도합니다.")
            time.sleep(sleep_sec)
        except Exception as exc:  # pragma: no cover
            log.warning(
                "[BOOTSTRAP] stage=%s attempt=%s/%s code=runtime_bootstrap_failed message=%s",
                stage,
                attempt,
                max_retries,
                str(exc),
            )
            _emit_progress(
                f"stage={stage} attempt={attempt}/{max_retries} 실패: "
                f"code=runtime_bootstrap_failed message={exc}"
            )
            if attempt >= max_retries:
                os.environ["NUVION_ZSAD_BACKEND"] = "none"
                _emit_progress("최종 실패: 추론 backend를 none으로 강등하고 서비스는 계속 유지합니다.")
                return False
            sleep_sec = min(base_backoff * (2 ** (attempt - 1)), 30.0)
            _emit_progress(f"{sleep_sec:.1f}s 후 재시도합니다.")
            time.sleep(sleep_sec)

    os.environ["NUVION_ZSAD_BACKEND"] = "none"
    _emit_progress("최종 실패: 추론 backend를 none으로 강등하고 서비스는 계속 유지합니다.")
    return False
