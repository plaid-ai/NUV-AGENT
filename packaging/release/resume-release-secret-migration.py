#!/usr/bin/env python3
"""Plan or resume deletion of legacy repository-scoped release secrets.

Only secret metadata is inspected. Secret values are never requested or emitted.
Target-environment material must be verified by the calling workflow before the
``delete`` command is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Protocol


EXPECTED_REPOSITORY = "plaid-ai/NUV-AGENT"
EXPECTED_FORBIDDEN = (
    "APT_GPG_PASSPHRASE",
    "APT_GPG_PRIVATE_KEY",
    "GCP_PROJECT_ID",
    "GCP_SA_KEY",
    "HOMEBREW_TAP_TOKEN",
    "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
    "IQ9075_RELEASE_SIGNING_KEY_ID",
    "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
)
SECRET_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
REQUEST_TIMEOUT_SECONDS = 30
DELETE_ATTEMPTS = 3


class MigrationError(RuntimeError):
    """The repository-secret migration cannot safely continue."""


class RepositorySecrets(Protocol):
    def names(self) -> set[str]: ...

    def delete(self, name: str) -> bool: ...


def _forbidden_from_policy(path: Path) -> tuple[str, ...]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MigrationError("release security policy is unreadable") from exc
    forbidden = policy.get("forbiddenRepositorySecrets") if isinstance(policy, dict) else None
    if (
        not isinstance(forbidden, list)
        or tuple(forbidden) != EXPECTED_FORBIDDEN
        or any(not isinstance(name, str) or SECRET_NAME.fullmatch(name) is None for name in forbidden)
    ):
        raise MigrationError("forbidden repository-secret policy is invalid")
    return tuple(forbidden)


def classify_source_state(
    repository_names: set[str], *, forbidden: tuple[str, ...] = EXPECTED_FORBIDDEN
) -> str:
    """Select full copy only while every original source is still present.

    Any missing source means a previous deletion may have partially completed.
    The workflow must then skip every source-value read/copy and prove all target
    environment credentials directly before deleting the remaining sources.
    """

    present = repository_names.intersection(forbidden)
    return "copy" if len(present) == len(forbidden) else "target-only-resume"


class GitHubRepositorySecrets:
    def __init__(self, repository: str, token: str) -> None:
        if repository != EXPECTED_REPOSITORY:
            raise MigrationError("release-secret repository is not allowlisted")
        if not token:
            raise MigrationError("migration administrator credential is unavailable")
        self._repository = repository
        self._environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LC_ALL": "C",
            "GH_PROMPT_DISABLED": "1",
            "GH_TOKEN": token,
        }

    def _run(self, arguments: list[str], *, capture: bool) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["gh", *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
                check=False,
                env=self._environment,
            )
        except (OSError, subprocess.SubprocessError):
            raise MigrationError("repository secret metadata operation failed") from None

    def names(self) -> set[str]:
        completed = self._run(
            [
                "api",
                "--paginate",
                f"repos/{self._repository}/actions/secrets?per_page=100",
                "--jq",
                ".secrets[].name",
            ],
            capture=True,
        )
        if completed.returncode != 0:
            raise MigrationError("repository secret metadata query failed")
        values = completed.stdout.splitlines()
        if any(SECRET_NAME.fullmatch(name) is None for name in values):
            raise MigrationError("repository secret metadata is invalid")
        if len(values) != len(set(values)):
            raise MigrationError("repository secret metadata is duplicated")
        return set(values)

    def delete(self, name: str) -> bool:
        if name not in EXPECTED_FORBIDDEN:
            raise MigrationError("repository secret deletion target is not allowlisted")
        completed = self._run(
            [
                "secret",
                "delete",
                name,
                "--repo",
                self._repository,
                "--app",
                "actions",
            ],
            capture=False,
        )
        return completed.returncode == 0


def delete_remaining_sources(
    repository_secrets: RepositorySecrets,
    *,
    forbidden: tuple[str, ...] = EXPECTED_FORBIDDEN,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Delete only sources still present and make a partial failure rerunnable."""

    present = repository_secrets.names().intersection(forbidden)
    for name in forbidden:
        if name not in present:
            continue
        deleted = False
        for attempt in range(DELETE_ATTEMPTS):
            if repository_secrets.delete(name):
                deleted = True
                break
            if attempt + 1 < DELETE_ATTEMPTS:
                sleeper(1.0)
        if not deleted and name in repository_secrets.names():
            raise MigrationError("repository-secret deletion failed after retries")

    if repository_secrets.names().intersection(forbidden):
        raise MigrationError("forbidden repository secret remains")


def _write_mode(path: Path, mode: str) -> None:
    if not path.is_absolute():
        raise MigrationError("GitHub output path must be absolute")
    try:
        with path.open("a", encoding="utf-8") as output:
            output.write(f"mode={mode}\n")
    except OSError as exc:
        raise MigrationError("GitHub output could not be written") from exc


def main() -> int:
    if not sys.flags.isolated:
        print("release-secret migration helper requires Python isolated mode (-I)", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "delete"):
        command = subcommands.add_parser(name)
        command.add_argument("--repository", required=True)
        command.add_argument("--policy", type=Path, required=True)
        if name == "plan":
            command.add_argument("--github-output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        forbidden = _forbidden_from_policy(arguments.policy.resolve())
        client = GitHubRepositorySecrets(
            arguments.repository, os.environ.get("GH_TOKEN", "")
        )
        if arguments.command == "plan":
            mode = classify_source_state(client.names(), forbidden=forbidden)
            _write_mode(arguments.github_output, mode)
            print(f"release-secret migration mode: {mode}")
        else:
            delete_remaining_sources(client, forbidden=forbidden)
            print("repository-scoped release-secret migration is complete")
    except MigrationError as exc:
        print(f"release-secret migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
