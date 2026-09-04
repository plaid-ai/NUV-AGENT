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
SECRET_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,99}$")
MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_API_PAGES = 100
PER_PAGE = 100
API_NOT_FOUND = object()


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

    def _get(self, path: str, *, allow_not_found: bool) -> Any:
        if not path.startswith("/") or "\r" in path or "\n" in path:
            raise SettingsError("GitHub settings API path is invalid")
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
                payload = response.read(MAX_API_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return API_NOT_FOUND
            raise SettingsError(
                f"GitHub settings API denied {path} with HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SettingsError(f"GitHub settings API failed for {path}: {exc}") from exc
        if len(payload) > MAX_API_RESPONSE_BYTES:
            raise SettingsError("GitHub settings API response exceeds limit")
        return _strict_json(payload, label="GitHub API response")

    def get(self, path: str) -> Any:
        result = self._get(path, allow_not_found=False)
        if result is API_NOT_FOUND:  # pragma: no cover - guarded by allow_not_found
            raise SettingsError("GitHub settings API returned no response")
        return result

    def get_optional(self, path: str) -> Any:
        return self._get(path, allow_not_found=True)


def _page_path(path: str, page: int) -> str:
    if page < 1 or re.search(r"(?:^|[?&])(page|per_page)=", path):
        raise SettingsError("paginated GitHub API path is invalid")
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}per_page={PER_PAGE}&page={page}"


def _paginated_list(api: GitHubApi, path: str, *, label: str) -> list[Any]:
    result: list[Any] = []
    for page in range(1, MAX_API_PAGES + 1):
        payload = api.get(_page_path(path, page))
        if not isinstance(payload, list) or len(payload) > PER_PAGE:
            raise SettingsError(f"{label} page is invalid")
        result.extend(payload)
        if len(payload) < PER_PAGE:
            return result
    raise SettingsError(f"{label} pagination exceeds limit")


def _paginated_collection(
    api: GitHubApi,
    path: str,
    *,
    member: str,
    label: str,
) -> list[Any]:
    result: list[Any] = []
    expected_total: int | None = None
    for page in range(1, MAX_API_PAGES + 1):
        payload = api.get(_page_path(path, page))
        if not isinstance(payload, dict):
            raise SettingsError(f"{label} page is invalid")
        total = payload.get("total_count")
        values = payload.get(member)
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < 0
            or not isinstance(values, list)
            or len(values) > PER_PAGE
        ):
            raise SettingsError(f"{label} page is invalid")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise SettingsError(f"{label} total changed during pagination")
        result.extend(values)
        if len(result) > total:
            raise SettingsError(f"{label} exceeds declared total")
        if len(values) < PER_PAGE:
            if len(result) != total:
                raise SettingsError(f"{label} does not match declared total")
            return result
    raise SettingsError(f"{label} pagination exceeds limit")


def _secret_names(values: list[Any], *, label: str) -> set[str]:
    names: set[str] = set()
    for value in values:
        name = value.get("name") if isinstance(value, dict) else None
        if not isinstance(name, str) or not SECRET_NAME.fullmatch(name) or name in names:
            raise SettingsError(f"{label} contains invalid or duplicate metadata")
        names.add(name)
    return names


def _organization_secret_applies(
    api: GitHubApi,
    *,
    repository: str,
    repository_id: int,
    repository_private: bool,
    organization: str,
    secret: dict[str, Any],
) -> bool:
    name = secret.get("name")
    visibility = secret.get("visibility")
    if not isinstance(name, str) or not SECRET_NAME.fullmatch(name):
        raise SettingsError("organization secret metadata is invalid")
    if visibility == "all":
        return True
    if visibility == "private":
        return repository_private
    if visibility != "selected":
        raise SettingsError(f"organization secret {name} visibility is invalid")
    selected = _paginated_collection(
        api,
        f"/orgs/{urllib.parse.quote(organization, safe='')}/actions/secrets/"
        f"{urllib.parse.quote(name, safe='')}/repositories",
        member="repositories",
        label=f"organization secret {name} selected repositories",
    )
    identifiers: set[int] = set()
    for value in selected:
        identifier = value.get("id") if isinstance(value, dict) else None
        full_name = value.get("full_name") if isinstance(value, dict) else None
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 1
            or identifier in identifiers
            or not isinstance(full_name, str)
        ):
            raise SettingsError(
                f"organization secret {name} selected repository metadata is invalid"
            )
        identifiers.add(identifier)
        if identifier == repository_id and full_name != repository:
            raise SettingsError(
                f"organization secret {name} repository identity is inconsistent"
            )
    return repository_id in identifiers


def _ruleset_covers(
    rulesets: list[Any],
    *,
    target: str,
    include: str,
    required_rules: set[str],
    required_name: str | None = None,
    required_source: str | None = None,
    required_status_context: str | None = None,
    required_status_integration_id: int | None = None,
    required_pull_request_approvals: int | None = None,
    required_bypass_team_id: int | None = None,
    required_bypass_mode: str | None = None,
) -> bool:
    candidates = [
        value
        for value in rulesets
        if isinstance(value, dict)
        and value.get("target") == target
        and value.get("enforcement") == "active"
    ]
    if len(candidates) != 1:
        return False
    ruleset = candidates[0]
    if required_name is not None and ruleset.get("name") != required_name:
        return False
    if required_source is not None and (
        ruleset.get("source") != required_source
        or ruleset.get("source_type") != "Repository"
    ):
        return False
    conditions = ruleset.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != {"ref_name"}:
        return False
    ref_name = conditions.get("ref_name")
    if not isinstance(ref_name, dict) or ref_name != {
        "include": [include],
        "exclude": [],
    }:
        return False
    rules = ruleset.get("rules")
    if not isinstance(rules, list) or len(rules) != len(required_rules):
        return False
    types = [
        rule.get("type")
        for rule in rules
        if isinstance(rule, dict) and isinstance(rule.get("type"), str)
    ]
    if len(types) != len(rules) or set(types) != required_rules:
        return False
    for rule in rules:
        rule_type = rule.get("type")
        if rule_type not in {"pull_request", "required_status_checks"} and set(rule) != {
            "type"
        }:
            return False

    bypass_actors = ruleset.get("bypass_actors", [])
    if not isinstance(bypass_actors, list):
        return False
    if required_bypass_team_id is None:
        if bypass_actors:
            return False
    else:
        observed = [
            (
                actor.get("actor_id"),
                actor.get("actor_type"),
                actor.get("bypass_mode"),
            )
            for actor in bypass_actors
            if isinstance(actor, dict)
        ]
        if len(observed) != len(bypass_actors) or observed != [
            (required_bypass_team_id, "Team", required_bypass_mode)
        ]:
            return False

    if required_status_context is not None:
        status_rules = [
            rule for rule in rules if rule.get("type") == "required_status_checks"
        ]
        if len(status_rules) != 1 or status_rules[0].get("parameters") != {
            "strict_required_status_checks_policy": True,
            "do_not_enforce_on_create": False,
            "required_status_checks": [
                {
                    "context": required_status_context,
                    "integration_id": required_status_integration_id,
                }
            ],
        }:
            return False
    if required_pull_request_approvals is not None:
        pull_request_rules = [
            rule for rule in rules if rule.get("type") == "pull_request"
        ]
        if len(pull_request_rules) != 1 or pull_request_rules[0].get(
            "parameters"
        ) != {
            "allowed_merge_methods": ["merge", "squash", "rebase"],
            "dismiss_stale_reviews_on_push": True,
            "dismissal_restriction": {"allowed_actors": [], "enabled": False},
            "require_code_owner_review": True,
            "require_extra_approval_for_unattributed_changes": True,
            "require_last_push_approval": True,
            "required_approving_review_count": required_pull_request_approvals,
            "required_review_thread_resolution": True,
            "required_reviewers": [],
        }:
            return False
    return True


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
    if (
        not isinstance(policy, dict)
        or policy.get("schemaVersion") != 1
        or set(policy)
        != {
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
    ):
        raise SettingsError("release security policy is invalid")
    default_branch = policy.get("defaultBranch")
    required_status_context = policy.get("requiredStatusContext")
    environments = policy.get("requiredEnvironments")
    release_admin_team_id = policy.get("releaseAdminTeamId")
    governance = policy.get("governance")
    expected_governance = {
        "pullRequestApprovals": 1,
        "dismissStaleReviewsOnPush": True,
        "requireCodeOwnerReview": True,
        "requireLastPushApproval": True,
        "requireExtraApprovalForUnattributedChanges": True,
        "requiredReviewThreadResolution": True,
        "allowedMergeMethods": ["merge", "squash", "rebase"],
        "environmentReviewers": 0,
        "requiredStatusContext": required_status_context,
        "requiredStatusIntegrationId": 15368,
    }
    expected_environment_secrets = {
        "homebrew-release": ["HOMEBREW_TAP_TOKEN"],
        "apt-release": [
            "APT_GPG_PASSPHRASE",
            "APT_GPG_PRIVATE_KEY",
            "GCP_PROJECT_ID",
            "GCP_SA_KEY",
        ],
        "iq9075-release": [
            "GCP_PROJECT_ID",
            "GCP_SA_KEY",
            "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
        ],
        "iq9075-candidate-sign": ["IQ9075_RELEASE_SIGNING_PRIVATE_KEY"],
        "iq9075-candidate-stage": ["GCP_PROJECT_ID", "GCP_SA_KEY"],
        "face-artifacts-release": ["GCP_PROJECT_ID", "GCP_SA_KEY"],
    }
    if (
        default_branch != "main"
        or required_status_context != "agent-release-gate"
        or not isinstance(environments, dict)
        or release_admin_team_id != 16128529
        or policy.get("immutableReleases") is not True
        or governance != expected_governance
        or set(environments) != set(expected_environment_secrets)
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
    ruleset_summaries = _paginated_list(
        api,
        f"/repos/{repository}/rulesets?includes_parents=true",
        label="effective repository rulesets",
    )
    rulesets: list[Any] = []
    identifiers: set[int] = set()
    for summary in ruleset_summaries:
        identifier = summary.get("id") if isinstance(summary, dict) else None
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 1
            or identifier in identifiers
        ):
            raise SettingsError("repository ruleset identifier is invalid")
        identifiers.add(identifier)
        detail = api.get(f"/repos/{repository}/rulesets/{identifier}")
        if not isinstance(detail, dict) or detail.get("id") != identifier:
            raise SettingsError("repository ruleset detail identity is invalid")
        rulesets.append(detail)
    active_rulesets = [
        value
        for value in rulesets
        if isinstance(value, dict) and value.get("enforcement") == "active"
    ]
    active_targets = [value.get("target") for value in active_rulesets]
    if (
        len(active_rulesets) != 2
        or not all(isinstance(value, str) for value in active_targets)
        or sorted(active_targets) != ["branch", "tag"]
    ):
        raise SettingsError("effective release governance has unexpected active rulesets")
    if not _ruleset_covers(
        active_rulesets,
        target="branch",
        include=f"refs/heads/{default_branch}",
        required_rules={"deletion", "non_fast_forward", "pull_request", "required_status_checks"},
        required_name="protected-main",
        required_source=repository,
        required_status_context=required_status_context,
        required_status_integration_id=15368,
        required_pull_request_approvals=1,
        required_bypass_team_id=release_admin_team_id,
        required_bypass_mode="pull_request",
    ):
        raise SettingsError("active protected-main ruleset is incomplete")
    if not _ruleset_covers(
        active_rulesets,
        target="tag",
        include="refs/tags/v*",
        required_rules={"creation", "deletion", "non_fast_forward"},
        required_name="protected-release-tags",
        required_source=repository,
        required_bypass_team_id=release_admin_team_id,
        required_bypass_mode="always",
    ):
        raise SettingsError("active v* tag ruleset is incomplete")
    classic_protection = api.get_optional(
        f"/repos/{repository}/branches/{urllib.parse.quote(default_branch)}/protection"
    )
    if classic_protection is not API_NOT_FOUND:
        raise SettingsError(
            "classic default-branch protection must be absent; the exact ruleset is authoritative"
        )
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
        expected_branch_policy = {
            "protectedBranches": False,
            "customBranchPolicies": True,
        }
        expected_deployment_policies = [{"name": default_branch, "type": "branch"}]
        if (
            set(requirements)
            != {
                "requireReviewers",
                "preventSelfReview",
                "reviewerTeamId",
                "deploymentBranchPolicy",
                "deploymentBranchPolicies",
                "protectionRuleTypes",
                "requiredSecrets",
            }
            or requirements.get("requireReviewers") is not False
            or requirements.get("preventSelfReview") is not False
            or requirements.get("reviewerTeamId") is not None
            or requirements.get("deploymentBranchPolicy") != expected_branch_policy
            or requirements.get("deploymentBranchPolicies")
            != expected_deployment_policies
            or requirements.get("protectionRuleTypes") != ["branch_policy"]
            or not isinstance(required_secrets, list)
            or required_secrets != expected_environment_secrets[name]
            or not all(
                isinstance(secret, str) and SECRET_NAME.fullmatch(secret)
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
        if branch_policy != {
            "protected_branches": False,
            "custom_branch_policies": True,
        }:
            raise SettingsError(f"environment {name} branch-policy mode is invalid")
        protection_rules = environment.get("protection_rules")
        if (
            not isinstance(protection_rules, list)
            or len(protection_rules) != 1
            or not isinstance(protection_rules[0], dict)
            or protection_rules[0].get("type") != "branch_policy"
        ):
            raise SettingsError(
                f"environment {name} must have exactly the branch-policy protection rule"
            )
        deployment_policies = _paginated_collection(
            api,
            f"/repos/{repository}/environments/{urllib.parse.quote(name, safe='')}/deployment-branch-policies",
            member="branch_policies",
            label=f"environment {name} deployment branch policies",
        )
        normalized_policies = [
            {"name": value.get("name"), "type": value.get("type")}
            for value in deployment_policies
            if isinstance(value, dict)
        ]
        if len(normalized_policies) != len(deployment_policies) or normalized_policies != [
            {"name": default_branch, "type": "branch"}
        ]:
            raise SettingsError(
                f"environment {name} must allow exactly the main branch"
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
        forbidden = policy.get("forbiddenRepositorySecrets")
        expected_forbidden = [
            "APT_GPG_PASSPHRASE",
            "APT_GPG_PRIVATE_KEY",
            "GCP_PROJECT_ID",
            "GCP_SA_KEY",
            "HOMEBREW_TAP_TOKEN",
            "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
            "IQ9075_RELEASE_SIGNING_KEY_ID",
            "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
        ]
        if (
            not isinstance(forbidden, list)
            or not forbidden
            or forbidden != expected_forbidden
            or len(set(forbidden)) != len(forbidden)
            or not all(
                isinstance(value, str) and SECRET_NAME.fullmatch(value)
                for value in forbidden
            )
        ):
            raise SettingsError("forbidden release secret policy is invalid")
        forbidden_names = set(forbidden)
        repository_secrets = _paginated_collection(
            api,
            f"/repos/{repository}/actions/secrets",
            member="secrets",
            label="repository secret metadata",
        )
        names = _secret_names(repository_secrets, label="repository secret metadata")
        if names & forbidden_names:
            raise SettingsError("high-impact release secrets remain repository-scoped")

        repository_id = repo.get("id") if isinstance(repo, dict) else None
        repository_private = repo.get("private") if isinstance(repo, dict) else None
        owner = repo.get("owner") if isinstance(repo, dict) else None
        owner_login = owner.get("login") if isinstance(owner, dict) else None
        owner_type = owner.get("type") if isinstance(owner, dict) else None
        if (
            isinstance(repository_id, bool)
            or not isinstance(repository_id, int)
            or repository_id < 1
            or not isinstance(repository_private, bool)
            or not isinstance(owner_login, str)
            or owner_login.lower() != repository.split("/", 1)[0].lower()
            or owner_type not in {"Organization", "User"}
        ):
            raise SettingsError("repository owner metadata is invalid")
        if owner_type == "Organization":
            organization_secrets = _paginated_collection(
                api,
                f"/orgs/{urllib.parse.quote(owner_login, safe='')}/actions/secrets",
                member="secrets",
                label="organization secret metadata",
            )
            _secret_names(organization_secrets, label="organization secret metadata")
            for value in organization_secrets:
                if not isinstance(value, dict):  # guarded by _secret_names
                    raise SettingsError("organization secret metadata is invalid")
                if value.get("name") in forbidden_names and _organization_secret_applies(
                    api,
                    repository=repository,
                    repository_id=repository_id,
                    repository_private=repository_private,
                    organization=owner_login,
                    secret=value,
                ):
                    raise SettingsError(
                        "high-impact release secret is shared from the organization"
                    )
        for name, requirements in environments.items():
            secret_metadata = _paginated_collection(
                api,
                f"/repos/{repository}/environments/{urllib.parse.quote(name, safe='')}/secrets",
                member="secrets",
                label=f"environment {name} secret metadata",
            )
            available = _secret_names(
                secret_metadata, label=f"environment {name} secret metadata"
            )
            required = requirements.get("requiredSecrets")
            if not isinstance(required, list) or set(required) != available:
                raise SettingsError(
                    f"environment {name} secret metadata differs from exact policy"
                )
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
