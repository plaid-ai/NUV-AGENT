#!/usr/bin/env python3
"""Read-only, fail-closed audit of GitHub release protection settings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from publisher_trust import PublisherTrustError, publisher_surface


REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class SettingsError(RuntimeError):
    pass


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SettingsError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SettingsError(f"invalid {label}: {exc}") from exc


class GitHubApi:
    def __init__(self, repository: str, token: str) -> None:
        if not REPOSITORY.fullmatch(repository):
            raise SettingsError("GitHub repository identity is invalid")
        if not token or "\n" in token or "\r" in token:
            raise SettingsError("GitHub API token is unavailable")
        self.repository = repository
        self.token = token

    def get(self, path: str) -> Any:
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "nuv-release-settings-gate/1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise SettingsError(
                f"GitHub settings API denied {path} with HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SettingsError(f"GitHub settings API failed for {path}: {exc}") from exc
        if len(payload) > 2 * 1024 * 1024:
            raise SettingsError("GitHub settings API response exceeds limit")
        return _strict_json(payload, label="GitHub API response")


def _ruleset_covers(
    rulesets: list[Any],
    *,
    target: str,
    include: str,
    required_rules: set[str],
    required_status_context: str | None = None,
    required_status_integration_id: int | None = None,
    require_pull_request_hardening: bool = False,
    required_bypass_team_id: int | None = None,
    required_bypass_mode: str | None = None,
) -> bool:
    for ruleset in rulesets:
        if (
            not isinstance(ruleset, dict)
            or ruleset.get("target") != target
            or ruleset.get("enforcement") != "active"
        ):
            continue
        conditions = ruleset.get("conditions")
        if not isinstance(conditions, dict):
            continue
        ref_name = conditions.get("ref_name")
        includes = ref_name.get("include") if isinstance(ref_name, dict) else None
        excludes = ref_name.get("exclude") if isinstance(ref_name, dict) else None
        if (
            not isinstance(includes, list)
            or include not in includes
            or not isinstance(excludes, list)
            or excludes
        ):
            continue
        rules = ruleset.get("rules")
        if not isinstance(rules, list):
            continue
        types = {
            rule.get("type")
            for rule in rules
            if isinstance(rule, dict) and isinstance(rule.get("type"), str)
        }
        if not required_rules <= types:
            continue
        if required_bypass_team_id is not None:
            bypass_actors = ruleset.get("bypass_actors")
            if not isinstance(bypass_actors, list):
                continue
            observed = {
                (
                    actor.get("actor_id"),
                    actor.get("actor_type"),
                    actor.get("bypass_mode"),
                )
                for actor in bypass_actors
                if isinstance(actor, dict)
            }
            if observed != {
                (required_bypass_team_id, "Team", required_bypass_mode)
            }:
                continue
        if required_status_context is not None:
            status_rules = [
                rule
                for rule in rules
                if isinstance(rule, dict) and rule.get("type") == "required_status_checks"
            ]
            if len(status_rules) != 1:
                continue
            parameters = status_rules[0].get("parameters")
            checks = (
                parameters.get("required_status_checks")
                if isinstance(parameters, dict)
                else None
            )
            if (
                not isinstance(parameters, dict)
                or parameters.get("strict_required_status_checks_policy") is not True
                or parameters.get("do_not_enforce_on_create") is not False
                or checks
                != [
                    {
                        "context": required_status_context,
                        "integration_id": required_status_integration_id,
                    }
                ]
            ):
                continue
        if require_pull_request_hardening:
            pull_request_rules = [
                rule
                for rule in rules
                if isinstance(rule, dict) and rule.get("type") == "pull_request"
            ]
            if len(pull_request_rules) != 1:
                continue
            parameters = pull_request_rules[0].get("parameters")
            approvals = (
                parameters.get("required_approving_review_count")
                if isinstance(parameters, dict)
                else None
            )
            if (
                isinstance(approvals, bool)
                or not isinstance(approvals, int)
                or approvals != 0
                or parameters.get("dismiss_stale_reviews_on_push") is not False
                or parameters.get("require_code_owner_review") is not False
                or parameters.get("require_last_push_approval") is not False
                or parameters.get("required_review_thread_resolution") is not True
            ):
                continue
        return True
    return False


def verify_settings(
    *,
    repository: str,
    token: str,
    policy_path: Path,
    trusted_publisher_sha: str,
    include_secret_scopes: bool,
) -> dict[str, Any]:
    if not COMMIT_SHA.fullmatch(trusted_publisher_sha):
        raise SettingsError("trusted publisher SHA is invalid")
    policy = _strict_json(policy_path.read_bytes(), label="release security policy")
    if not isinstance(policy, dict) or policy.get("schemaVersion") != 1:
        raise SettingsError("release security policy is invalid")
    default_branch = policy.get("defaultBranch")
    required_status_context = policy.get("requiredStatusContext")
    environments = policy.get("requiredEnvironments")
    release_admin_team_id = policy.get("releaseAdminTeamId")
    governance = policy.get("governance")
    if (
        not isinstance(default_branch, str)
        or not isinstance(required_status_context, str)
        or not required_status_context
        or not isinstance(environments, dict)
        or isinstance(release_admin_team_id, bool)
        or not isinstance(release_admin_team_id, int)
        or release_admin_team_id < 1
        or policy.get("immutableReleases") is not True
        or governance
        != {
            "pullRequestApprovals": 0,
            "environmentReviewers": 0,
            "requiredStatusContext": required_status_context,
            "requiredStatusIntegrationId": 15368,
        }
        or set(environments)
        != {"homebrew-release", "apt-release", "iq9075-release"}
    ):
        raise SettingsError("release security policy settings are invalid")
    api = GitHubApi(repository, token)
    repo = api.get(f"/repos/{repository}")
    if not isinstance(repo, dict) or repo.get("default_branch") != default_branch:
        raise SettingsError("repository default branch does not match release policy")
    immutable_releases = api.get(f"/repos/{repository}/immutable-releases")
    if (
        not isinstance(immutable_releases, dict)
        or immutable_releases.get("enabled") is not True
    ):
        raise SettingsError("repository immutable releases are not enabled")
    branch = api.get(f"/repos/{repository}/branches/{urllib.parse.quote(default_branch)}")
    if not isinstance(branch, dict) or branch.get("protected") is not True:
        raise SettingsError("default branch is not protected")
    ruleset_summaries = api.get(f"/repos/{repository}/rulesets?includes_parents=true")
    if not isinstance(ruleset_summaries, list):
        raise SettingsError("repository ruleset response is invalid")
    rulesets: list[Any] = []
    for summary in ruleset_summaries:
        identifier = summary.get("id") if isinstance(summary, dict) else None
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 1:
            raise SettingsError("repository ruleset identifier is invalid")
        rulesets.append(api.get(f"/repos/{repository}/rulesets/{identifier}"))
    if not _ruleset_covers(
        rulesets,
        target="branch",
        include=f"refs/heads/{default_branch}",
        required_rules={"deletion", "non_fast_forward", "pull_request", "required_status_checks"},
        required_status_context=required_status_context,
        required_status_integration_id=15368,
        require_pull_request_hardening=True,
        required_bypass_team_id=release_admin_team_id,
        required_bypass_mode="pull_request",
    ):
        raise SettingsError("active protected-main ruleset is incomplete")
    if not _ruleset_covers(
        rulesets,
        target="tag",
        include="refs/tags/v*",
        required_rules={"creation", "deletion", "non_fast_forward"},
        required_bypass_team_id=release_admin_team_id,
        required_bypass_mode="always",
    ):
        raise SettingsError("active v* tag ruleset is incomplete")
    permissions = api.get(f"/repos/{repository}/actions/permissions/workflow")
    if not isinstance(permissions, dict) or (
        permissions.get("default_workflow_permissions") != "read"
        or permissions.get("can_approve_pull_request_reviews") is not False
    ):
        raise SettingsError("default Actions token permissions are not read-only")

    for name, requirements in environments.items():
        if not isinstance(name, str) or not isinstance(requirements, dict):
            raise SettingsError("environment policy is invalid")
        required_secrets = requirements.get("requiredSecrets")
        if (
            set(requirements)
            != {
                "requireReviewers",
                "preventSelfReview",
                "reviewerTeamId",
                "requiredSecrets",
            }
            or requirements.get("requireReviewers") is not False
            or requirements.get("preventSelfReview") is not False
            or requirements.get("reviewerTeamId") is not None
            or not isinstance(required_secrets, list)
            or not required_secrets
            or not all(
                isinstance(secret, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", secret)
                for secret in required_secrets
            )
            or len(set(required_secrets)) != len(required_secrets)
        ):
            raise SettingsError(f"environment {name} policy is invalid")
        environment = api.get(
            f"/repos/{repository}/environments/{urllib.parse.quote(name, safe='')}"
        )
        if not isinstance(environment, dict) or environment.get("name") != name:
            raise SettingsError(f"required environment is unavailable: {name}")
        branch_policy = environment.get("deployment_branch_policy")
        if not isinstance(branch_policy, dict) or (
            branch_policy.get("protected_branches") is not True
            or branch_policy.get("custom_branch_policies") is not False
        ):
            raise SettingsError(f"environment {name} is not restricted to protected branches")
        protection_rules = environment.get("protection_rules")
        if not isinstance(protection_rules, list) or protection_rules:
            raise SettingsError(
                f"environment {name} must have no approval or wait protection rules"
            )

    if include_secret_scopes:
        expected_variables = {
            "RELEASE_SECURITY_POLICY_VERSION": "1",
            "RELEASE_TRUSTED_PUBLISHER_SHA": trusted_publisher_sha,
        }
        for name, expected in expected_variables.items():
            variable = api.get(f"/repos/{repository}/actions/variables/{name}")
            if not isinstance(variable, dict) or variable.get("value") != expected:
                raise SettingsError(f"repository variable {name} does not match policy")
        repository_secrets = api.get(f"/repos/{repository}/actions/secrets?per_page=100")
        if not isinstance(repository_secrets, dict) or not isinstance(
            repository_secrets.get("secrets"), list
        ):
            raise SettingsError("repository secret metadata response is invalid")
        names = {
            value.get("name")
            for value in repository_secrets["secrets"]
            if isinstance(value, dict)
        }
        forbidden = policy.get("forbiddenRepositorySecrets")
        if not isinstance(forbidden, list) or names & set(forbidden):
            raise SettingsError("high-impact release secrets remain repository-scoped")
        for name, requirements in environments.items():
            secret_payload = api.get(
                f"/repos/{repository}/environments/{urllib.parse.quote(name, safe='')}/secrets?per_page=100"
            )
            if not isinstance(secret_payload, dict) or not isinstance(
                secret_payload.get("secrets"), list
            ):
                raise SettingsError(
                    f"environment {name} secret metadata response is invalid"
                )
            available = {
                value.get("name")
                for value in secret_payload["secrets"]
                if isinstance(value, dict)
            }
            required = requirements.get("requiredSecrets")
            if not isinstance(required, list) or not set(required) <= available:
                raise SettingsError(f"environment {name} is missing required secret metadata")
    return {
        "schemaVersion": 1,
        "repository": repository,
        "defaultBranch": default_branch,
        "trustedPublisherSha": trusted_publisher_sha,
        "secretScopesChecked": include_secret_scopes,
        "governance": governance,
        "status": "VERIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trusted-publisher-sha", required=True)
    parser.add_argument("--publisher-root", type=Path)
    parser.add_argument("--include-secret-scopes", action="store_true")
    parser.add_argument("--attestation-output", type=Path)
    parser.add_argument("--valid-hours", type=int, default=24)
    arguments = parser.parse_args()
    try:
        token = os.environ.get(arguments.token_env, "")
        result = verify_settings(
            repository=arguments.repository,
            token=token,
            policy_path=arguments.policy.resolve(),
            trusted_publisher_sha=arguments.trusted_publisher_sha,
            include_secret_scopes=arguments.include_secret_scopes,
        )
        if arguments.attestation_output is not None:
            if not arguments.include_secret_scopes:
                raise SettingsError(
                    "settings attestation requires --include-secret-scopes"
                )
            if arguments.valid_hours < 1 or arguments.valid_hours > 24:
                raise SettingsError("settings attestation validity must be 1..24 hours")
            if arguments.publisher_root is None:
                raise SettingsError("settings attestation requires --publisher-root")
            surface = publisher_surface(
                arguments.publisher_root,
                expected_sha=arguments.trusted_publisher_sha,
            )
            now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            expires = now + dt.timedelta(hours=arguments.valid_hours)
            policy_bytes = arguments.policy.resolve().read_bytes()
            attestation = {
                "schemaVersion": 1,
                "kind": "nuvion-release-settings-attestation",
                "repository": arguments.repository,
                "publisherTreeSha256": surface["publisherTreeSha256"],
                "policySha256": hashlib.sha256(policy_bytes).hexdigest(),
                "verifiedAt": now.isoformat().replace("+00:00", "Z"),
                "expiresAt": expires.isoformat().replace("+00:00", "Z"),
                "settings": {
                    "defaultBranch": result["defaultBranch"],
                    "governance": result["governance"],
                    "secretScopesChecked": result["secretScopesChecked"],
                    "status": result["status"],
                },
                "trustedPublisherSha": result["trustedPublisherSha"],
                "workflowSha256": surface["workflowSha256"],
            }
            output = arguments.attestation_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output.open("x", encoding="utf-8") as handle:
                    json.dump(
                        attestation,
                        handle,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.write("\n")
            except FileExistsError as exc:
                raise SettingsError("settings attestation output already exists") from exc
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (OSError, PublisherTrustError, SettingsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
