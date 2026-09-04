#!/usr/bin/env python3
"""Verify the immutable signed tag that authorizes the candidate publisher."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
CONFIG_SCHEMA = re.compile(r"^[1-9][0-9]*$")
EXPECTED_TAG = "candidate-publisher-v1"
EXPECTED_TAG_REF = f"refs/tags/{EXPECTED_TAG}"
EXPECTED_WORKFLOW = ".github/workflows/iq9075-candidate-trusted-publish.yml"
EXPECTED_RULESET = "protected-candidate-publisher"
POLICY_RELATIVE_PATH = Path("packaging/release/release-security-policy.json")
SIGNER_RELATIVE_PATH = Path("packaging/release/trusted-tag-signers")
CANDIDATE_PUBLISHER_KEYS = {
    "tag",
    "tagRef",
    "workflow",
    "agentVersion",
    "releaseSequence",
    "configSchema",
    "minUpdaterVersion",
    "rulesetName",
}


class CandidatePublisherVerificationError(RuntimeError):
    pass


def _strict_object(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidatePublisherVerificationError(
                    f"duplicate JSON member in {path}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidatePublisherVerificationError(
            f"cannot load security policy {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CandidatePublisherVerificationError(
            "release security policy must be a JSON object"
        )
    return value


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    }


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            timeout=30,
            check=False,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CandidatePublisherVerificationError(
            "candidate publisher Git operation failed"
        ) from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise CandidatePublisherVerificationError(str(detail)[:500])
    return result


def _git_value(repository: Path, arguments: Sequence[str]) -> str:
    value = _git(repository, arguments).stdout.strip()
    if not value or "\n" in value or "\r" in value:
        raise CandidatePublisherVerificationError(
            "candidate publisher Git operation returned an invalid scalar"
        )
    return value


def _dirty_checkout(repository: Path) -> bool:
    # Ignored files are executable surfaces too (for example forged bytecode),
    # so a trusted publisher checkout must be completely fresh.
    return bool(
        _git(
            repository,
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            ],
        ).stdout
    )


def _candidate_policy(policy_path: Path) -> tuple[dict[str, Any], set[str]]:
    policy = _strict_object(policy_path)
    if policy.get("schemaVersion") != 1 or policy.get("defaultBranch") != "main":
        raise CandidatePublisherVerificationError(
            "release security policy branch or schema is invalid"
        )

    candidate = policy.get("candidatePublisher")
    if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_PUBLISHER_KEYS:
        raise CandidatePublisherVerificationError(
            "candidate publisher policy fields are invalid"
        )
    release_sequence = candidate.get("releaseSequence")
    if (
        candidate.get("tag") != EXPECTED_TAG
        or candidate.get("tagRef") != EXPECTED_TAG_REF
        or candidate.get("workflow") != EXPECTED_WORKFLOW
        or candidate.get("rulesetName") != EXPECTED_RULESET
        or not isinstance(candidate.get("agentVersion"), str)
        or SEMVER.fullmatch(candidate["agentVersion"]) is None
        or isinstance(release_sequence, bool)
        or not isinstance(release_sequence, int)
        or release_sequence < 1
        or not isinstance(candidate.get("configSchema"), str)
        or CONFIG_SCHEMA.fullmatch(candidate["configSchema"]) is None
        or not isinstance(candidate.get("minUpdaterVersion"), str)
        or SEMVER.fullmatch(candidate["minUpdaterVersion"]) is None
    ):
        raise CandidatePublisherVerificationError(
            "candidate publisher policy values are invalid"
        )

    configured = policy.get("trustedTagSignerFingerprints")
    if not isinstance(configured, list) or not configured:
        raise CandidatePublisherVerificationError(
            "trusted tag signer allowlist is empty"
        )
    allowed = {value for value in configured if isinstance(value, str)}
    if (
        len(allowed) != len(configured)
        or not all(FINGERPRINT.fullmatch(value) for value in allowed)
    ):
        raise CandidatePublisherVerificationError(
            "trusted tag signer allowlist is invalid"
        )
    return candidate, allowed


def _primary_fingerprints(gpg_home: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["gpg", "--batch", "--homedir", str(gpg_home), "--with-colons", "--list-keys"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CandidatePublisherVerificationError(
            "cannot inspect trusted tag signer keyring"
        ) from exc
    if result.returncode != 0:
        raise CandidatePublisherVerificationError(
            "cannot inspect trusted tag signer keyring"
        )
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
    tag_object_sha: str,
    *,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise CandidatePublisherVerificationError(
            "trusted tag signer directory is empty"
        )
    # Keep this prefix short: GnuPG agent sockets have a small AF_UNIX path
    # limit on macOS and fail closed when the temporary path exceeds it.
    with tempfile.TemporaryDirectory(prefix="nuv-gpg-") as raw_home:
        gpg_home = Path(raw_home)
        gpg_home.chmod(0o700)
        try:
            imported = subprocess.run(
                ["gpg", "--batch", "--homedir", str(gpg_home), "--import", *map(str, public_keys)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CandidatePublisherVerificationError(
                "cannot import trusted tag signer public keys"
            ) from exc
        if imported.returncode != 0:
            raise CandidatePublisherVerificationError(
                "cannot import trusted tag signer public keys"
            )
        if _primary_fingerprints(gpg_home) != allowed_fingerprints:
            raise CandidatePublisherVerificationError(
                "trusted tag signer files do not exactly match policy fingerprints"
            )
        environment = {
            **_git_environment(),
            "GNUPGHOME": str(gpg_home),
        }
        try:
            verified = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "core.fsmonitor=false",
                    "-c",
                    "gpg.format=openpgp",
                    "-c",
                    "gpg.program=gpg",
                    "verify-tag",
                    "--raw",
                    tag_object_sha,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CandidatePublisherVerificationError(
                "candidate publisher tag signature verification failed"
            ) from exc
        if verified.returncode != 0:
            raise CandidatePublisherVerificationError(
                "candidate publisher tag signature verification failed"
            )
        observed: set[str] = set()
        for line in f"{verified.stdout}\n{verified.stderr}".splitlines():
            if "[GNUPG:] VALIDSIG " not in line:
                continue
            for token in line.split("VALIDSIG", 1)[1].split():
                normalized = token.upper()
                if FINGERPRINT.fullmatch(normalized):
                    observed.add(normalized)
        accepted = observed & allowed_fingerprints
        if len(accepted) != 1:
            raise CandidatePublisherVerificationError(
                "candidate publisher tag signer fingerprint is not allowlisted"
            )
        return next(iter(accepted))


def _direct_tag_target(repository: Path, tag_object_sha: str, tag: str) -> str:
    raw = _git(repository, ["cat-file", "tag", tag_object_sha], text=False).stdout
    if not isinstance(raw, bytes):
        raise CandidatePublisherVerificationError(
            "candidate publisher tag object is invalid"
        )
    headers, separator, _message = raw.partition(b"\n\n")
    lines = headers.split(b"\n")
    if (
        not separator
        or len(lines) < 4
        or not lines[0].startswith(b"object ")
        or lines[1] != b"type commit"
        or lines[2] != f"tag {tag}".encode("ascii")
        or not lines[3].startswith(b"tagger ")
    ):
        raise CandidatePublisherVerificationError(
            "candidate publisher tag must directly annotate a commit"
        )
    try:
        target = lines[0][len(b"object ") :].decode("ascii")
    except UnicodeError as exc:
        raise CandidatePublisherVerificationError(
            "candidate publisher tag target is invalid"
        ) from exc
    if COMMIT_SHA.fullmatch(target) is None:
        raise CandidatePublisherVerificationError(
            "candidate publisher tag target is invalid"
        )
    return target


def verify_candidate_publisher_tag(
    *,
    repository: Path,
    publisher_sha: str,
    component_sha: str,
    main_ref: str,
    policy_path: Path,
    signer_directory: Path,
) -> dict[str, str]:
    repository = repository.resolve()
    if policy_path.resolve() != repository / POLICY_RELATIVE_PATH:
        raise CandidatePublisherVerificationError(
            "candidate publisher policy path is not bound to its checkout"
        )
    if signer_directory.resolve() != repository / SIGNER_RELATIVE_PATH:
        raise CandidatePublisherVerificationError(
            "candidate publisher signer directory is not bound to its checkout"
        )
    if COMMIT_SHA.fullmatch(publisher_sha) is None:
        raise CandidatePublisherVerificationError(
            "candidate publisher SHA must be a full lowercase commit SHA"
        )
    if COMMIT_SHA.fullmatch(component_sha) is None:
        raise CandidatePublisherVerificationError(
            "candidate component SHA must be a full lowercase commit SHA"
        )

    candidate, allowed = _candidate_policy(policy_path)
    expected_main_refs = {"refs/heads/main", "refs/remotes/origin/main"}
    if main_ref not in expected_main_refs:
        raise CandidatePublisherVerificationError(
            "candidate component main ref is invalid"
        )
    if _dirty_checkout(repository):
        raise CandidatePublisherVerificationError(
            "candidate publisher checkout is not clean"
        )

    initial_head = _git_value(repository, ["rev-parse", "HEAD"])
    if initial_head != publisher_sha:
        raise CandidatePublisherVerificationError(
            "candidate publisher checkout SHA differs from supplied publisher SHA"
        )
    initial_main = _git_value(repository, ["rev-parse", f"{main_ref}^{{commit}}"])
    if initial_main != component_sha:
        raise CandidatePublisherVerificationError(
            "candidate component SHA differs from current protected main"
        )
    if _git_value(repository, ["cat-file", "-t", publisher_sha]) != "commit":
        raise CandidatePublisherVerificationError(
            "candidate publisher SHA is not a commit"
        )
    if _git_value(repository, ["cat-file", "-t", component_sha]) != "commit":
        raise CandidatePublisherVerificationError(
            "candidate component SHA is not a commit"
        )

    tag_ref = candidate["tagRef"]
    initial_tag_object = _git_value(
        repository, ["show-ref", "--verify", "--hash", tag_ref]
    )
    if COMMIT_SHA.fullmatch(initial_tag_object) is None:
        raise CandidatePublisherVerificationError(
            "candidate publisher tag object SHA is invalid"
        )
    if _git_value(repository, ["cat-file", "-t", initial_tag_object]) != "tag":
        raise CandidatePublisherVerificationError(
            "candidate publisher ref must contain an annotated tag object"
        )
    direct_target = _direct_tag_target(repository, initial_tag_object, candidate["tag"])
    if direct_target != publisher_sha:
        raise CandidatePublisherVerificationError(
            "candidate publisher tag does not identify the supplied publisher SHA"
        )

    ancestry = _git(
        repository,
        ["merge-base", "--is-ancestor", publisher_sha, component_sha],
        check=False,
    )
    if ancestry.returncode != 0:
        raise CandidatePublisherVerificationError(
            "candidate publisher is not an ancestor of the current main component"
        )
    signer_fingerprint = _verify_signed_tag(
        repository,
        initial_tag_object,
        signer_directory=signer_directory,
        allowed_fingerprints=allowed,
    )

    final_tag_object = _git_value(
        repository, ["show-ref", "--verify", "--hash", tag_ref]
    )
    final_head = _git_value(repository, ["rev-parse", "HEAD"])
    final_main = _git_value(repository, ["rev-parse", f"{main_ref}^{{commit}}"])
    if (
        final_tag_object != initial_tag_object
        or final_head != initial_head
        or final_main != initial_main
    ):
        raise CandidatePublisherVerificationError(
            "candidate publisher Git identity changed during verification"
        )
    if _dirty_checkout(repository):
        raise CandidatePublisherVerificationError(
            "candidate publisher checkout changed during verification"
        )
    return {
        "candidate_publisher_tag": candidate["tag"],
        "candidate_publisher_tag_ref": tag_ref,
        "candidate_publisher_tag_object_sha": initial_tag_object,
        "candidate_publisher_sha": publisher_sha,
        "component_sha": component_sha,
        "tag_signer_fingerprint": signer_fingerprint,
    }


def _write_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise CandidatePublisherVerificationError(
                    f"output {key} contains a line break"
                )
            output.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--publisher-sha", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--main-ref", default="refs/remotes/origin/main")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--signer-directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_candidate_publisher_tag(
            repository=arguments.repository,
            publisher_sha=arguments.publisher_sha,
            component_sha=arguments.component_sha,
            main_ref=arguments.main_ref,
            policy_path=arguments.policy.resolve(),
            signer_directory=arguments.signer_directory.resolve(),
        )
        if arguments.github_output is not None:
            _write_github_output(arguments.github_output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except CandidatePublisherVerificationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
