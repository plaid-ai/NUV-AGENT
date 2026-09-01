from __future__ import annotations


class UpdaterError(RuntimeError):
    """A stable, non-secret error returned by the privileged updater."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class UpdaterSecurityError(UpdaterError):
    """An authentication, path, signature, or compatibility boundary failure."""
