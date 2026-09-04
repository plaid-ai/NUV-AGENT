#!/usr/bin/env python3
"""Fail-closed release tag and trusted publisher identity verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


SEMVER_TAG = re.compile(r"^v([0-9]+\.[0-9]+\.[0-9]+)$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")


class VerificationError(RuntimeError):
    pass


def _strict_object(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(f"duplicate JSON member in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot load security policy {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError("security policy must be a JSON object")
    return value


def _git(repository: Path, arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
        env=environment,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise VerificationError(detail[:500])
    return result


def _git_value(repository: Path, arguments: Sequence[str]) -> str:
    value = _git(repository, arguments).stdout.strip()
    if not value or "\n" in value or "\r" in value:
        raise VerificationError("git returned an invalid scalar value")
    return value


def _primary_fingerprints(gpg_home: Path) -> set[str]:
    result = subprocess.run(
        ["gpg", "--batch", "--homedir", str(gpg_home), "--with-colons", "--list-keys"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise VerificationError("cannot inspect trusted tag signer keyring")
    primary: set[str] = set()
    expect_primary = False
    for line in result.stdout.splitlines():
        fields = line.split(":")
        record = fields[0] if fields else ""
        if record == "pub":
            expect_primary = True
        elif record == "sub":
            expect_primary = False
        elif record == "fpr" and expect_primary and len(fields) > 9:
            fingerprint = fields[9].upper()
            if FINGERPRINT.fullmatch(fingerprint):
                primary.add(fingerprint)
            expect_primary = False
    return primary


def _verify_signed_tag(
    repository: Path,
    tag: str,
    *,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise VerificationError("trusted tag signer directory is empty")
    with tempfile.TemporaryDirectory(prefix="nuv-release-tag-verify-") as temporary:
        gpg_home = Path(temporary)
        gpg_home.chmod(0o700)
        imported = subprocess.run(
            ["gpg", "--batch", "--homedir", str(gpg_home), "--import", *map(str, public_keys)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if imported.returncode != 0:
            raise VerificationError("cannot import trusted tag signer public keys")
        keyring_fingerprints = _primary_fingerprints(gpg_home)
        if keyring_fingerprints != allowed_fingerprints:
            raise VerificationError("trusted tag signer files do not exactly match policy fingerprints")
        environment = {
            **os.environ,
            "GNUPGHOME": str(gpg_home),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        }
        verified = subprocess.run(
            ["git", "-C", str(repository), "verify-tag", "--raw", tag],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
        if verified.returncode != 0:
            raise VerificationError("release tag signature verification failed")
        status = f"{verified.stdout}\n{verified.stderr}"
        observed: set[str] = set()
        for line in status.splitlines():
            if "[GNUPG:] VALIDSIG " not in line:
                continue
            for token in line.split("VALIDSIG", 1)[1].split():
                normalized = token.upper()
                if FINGERPRINT.fullmatch(normalized):
                    observed.add(normalized)
        accepted = observed & allowed_fingerprints
        if len(accepted) != 1:
            raise VerificationError("release tag signer fingerprint is not allowlisted")
        return next(iter(accepted))


def verify_release_source(
    *,
    repository: Path,
    tag: str,
    origin_main_ref: str,
    trusted_publisher_sha: str,
    event_name: str,
    policy_path: Path,
    signer_directory: Path,
) -> dict[str, str]:
    if event_name not in {"workflow_dispatch", "workflow_run"}:
        raise VerificationError("release publisher event is not allowlisted")
    match = SEMVER_TAG.fullmatch(tag)
    if match is None:
        raise VerificationError("release tag must be exact vMAJOR.MINOR.PATCH")
    if not COMMIT_SHA.fullmatch(trusted_publisher_sha):
        raise VerificationError("trusted publisher SHA must be a full lowercase commit SHA")
    if _git_value(repository, ["cat-file", "-t", f"refs/tags/{tag}"]) != "tag":
        raise VerificationError("release tag must be an annotated tag object")
    tag_object_sha = _git_value(repository, ["rev-parse", f"refs/tags/{tag}^{{tag}}"])
    if not COMMIT_SHA.fullmatch(tag_object_sha):
        raise VerificationError("release annotated tag object SHA is invalid")
    component_sha = _git_value(repository, ["rev-parse", f"refs/tags/{tag}^{{commit}}"])
    if not COMMIT_SHA.fullmatch(component_sha):
        raise VerificationError("release component SHA is invalid")
    origin_main_sha = _git_value(repository, ["rev-parse", f"{origin_main_ref}^{{commit}}"])
    for candidate, label in (
        (component_sha, "release tag"),
        (trusted_publisher_sha, "trusted publisher"),
    ):
        ancestry = _git(
            repository,
            ["merge-base", "--is-ancestor", candidate, origin_main_sha],
            check=False,
        )
        if ancestry.returncode != 0:
            raise VerificationError(f"{label} commit is not contained in protected origin/main")

    policy = _strict_object(policy_path)
    expected_keys = {
        "schemaVersion",
        "defaultBranch",
        "requiredStatusContext",
        "releaseAdminTeamId",
        "immutableReleases",
        "trustedTagSignerFingerprints",
        "legacyUnsignedReruns",
        "governance",
        "requiredEnvironments",
        "forbiddenRepositorySecrets",
        "apt",
        "iq9075",
    }
    if set(policy) != expected_keys or policy.get("schemaVersion") != 1:
        raise VerificationError("release security policy fields are invalid")
    default_branch = policy.get("defaultBranch")
    required_context = policy.get("requiredStatusContext")
    release_admin_team_id = policy.get("releaseAdminTeamId")
    governance = policy.get("governance")
    if (
        default_branch != "main"
        or required_context != "agent-release-gate"
        or policy.get("immutableReleases") is not True
        or isinstance(release_admin_team_id, bool)
        or not isinstance(release_admin_team_id, int)
        or release_admin_team_id < 1
        or governance
        != {
            "pullRequestApprovals": 1,
            "dismissStaleReviewsOnPush": True,
            "requireCodeOwnerReview": True,
            "requireLastPushApproval": True,
            "requireExtraApprovalForUnattributedChanges": True,
            "requiredReviewThreadResolution": True,
            "allowedMergeMethods": ["merge", "squash", "rebase"],
            "environmentReviewers": 0,
            "requiredStatusContext": "agent-release-gate",
            "requiredStatusIntegrationId": 15368,
        }
        or origin_main_ref
        not in {f"refs/remotes/origin/{default_branch}", f"refs/heads/{default_branch}"}
    ):
        raise VerificationError("release branch, status context, or administrator policy is invalid")
    configured = policy.get("trustedTagSignerFingerprints")
    if not isinstance(configured, list) or not configured:
        raise VerificationError("trusted tag signer allowlist is empty")
    allowed = {value for value in configured if isinstance(value, str)}
    if (
        len(allowed) != len(configured)
        or not all(FINGERPRINT.fullmatch(value) for value in allowed)
    ):
        raise VerificationError("trusted tag signer allowlist is invalid")

    legacy = policy.get("legacyUnsignedReruns")
    if not isinstance(legacy, dict):
        raise VerificationError("legacy unsigned rerun policy is invalid")
    for legacy_tag, legacy_sha in legacy.items():
        if (
            not isinstance(legacy_tag, str)
            or SEMVER_TAG.fullmatch(legacy_tag) is None
            or not isinstance(legacy_sha, str)
            or COMMIT_SHA.fullmatch(legacy_sha) is None
        ):
            raise VerificationError("legacy unsigned rerun entry is invalid")
        if _git(repository, ["cat-file", "-e", f"{legacy_sha}^{{commit}}"], check=False).returncode != 0:
            raise VerificationError("legacy unsigned rerun commit is unavailable")
        if _git(
            repository,
            ["merge-base", "--is-ancestor", legacy_sha, origin_main_sha],
            check=False,
        ).returncode != 0:
            raise VerificationError("legacy unsigned rerun commit is outside protected main")
    if legacy:
        raise VerificationError("unsigned legacy release reruns are disabled")
    signer_fingerprint = _verify_signed_tag(
        repository,
        tag,
        signer_directory=signer_directory,
        allowed_fingerprints=allowed,
    )

    checked_out_sha = _git_value(repository, ["rev-parse", "HEAD"])
    if checked_out_sha != component_sha:
        raise VerificationError("checked out release source does not match tag commit")
    if _git(repository, ["status", "--porcelain", "--untracked-files=all"]).stdout:
        raise VerificationError("release source checkout is not clean")
    built_at = _git_value(repository, ["show", "-s", "--format=%cI", component_sha])
    return {
        "tag": tag,
        "tag_object_sha": tag_object_sha,
        "version": match.group(1),
        "component_sha": component_sha,
        "trusted_publisher_sha": trusted_publisher_sha,
        "origin_main_sha": origin_main_sha,
        "tag_signer_fingerprint": signer_fingerprint,
        "built_at": built_at,
    }


def _write_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise VerificationError(f"output {key} contains a line break")
            output.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--origin-main-ref", default="refs/remotes/origin/main")
    parser.add_argument("--trusted-publisher-sha", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--signer-directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_release_source(
            repository=arguments.repository.resolve(),
            tag=arguments.tag,
            origin_main_ref=arguments.origin_main_ref,
            trusted_publisher_sha=arguments.trusted_publisher_sha,
            event_name=arguments.event_name,
            policy_path=arguments.policy.resolve(),
            signer_directory=arguments.signer_directory.resolve(),
        )
        if arguments.github_output is not None:
            _write_github_output(arguments.github_output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except VerificationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
