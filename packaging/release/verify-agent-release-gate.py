#!/usr/bin/env python3
"""Bind release publication to the trusted gate for the exact component SHA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
WORKFLOW_PATH = ".github/workflows/agent-release-gate.yml"
WORKFLOW_NAME = "agent-release-gate"
GITHUB_ACTIONS_APP_SLUG = "github-actions"
MAX_API_BYTES = 4 * 1024 * 1024
MAX_CHECK_PAGES = 10
MAX_WORKFLOW_BYTES = 1024 * 1024


class ReleaseGateError(RuntimeError):
    pass


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseGateError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReleaseGateError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReleaseGateError(f"invalid {label} response") from exc


class GitHubApi:
    def __init__(self, repository: str, token: str) -> None:
        if not REPOSITORY.fullmatch(repository):
            raise ReleaseGateError("GitHub repository identity is invalid")
        if not token or "\n" in token or "\r" in token:
            raise ReleaseGateError("GitHub Actions token is unavailable")
        self.repository = repository
        self.token = token

    def get(self, path: str) -> Any:
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "nuv-release-gate-verifier/1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_API_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise ReleaseGateError(
                f"GitHub Actions API rejected {path} with HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ReleaseGateError(f"GitHub Actions API failed for {path}") from exc
        if len(raw) > MAX_API_BYTES:
            raise ReleaseGateError("GitHub Actions API response exceeds size limit")
        return _strict_json(raw, label="GitHub Actions API")

    def check_runs(self, component_sha: str) -> list[dict[str, Any]]:
        encoded_sha = urllib.parse.quote(component_sha, safe="")
        result: list[dict[str, Any]] = []
        for page in range(1, MAX_CHECK_PAGES + 1):
            payload = self.get(
                f"/repos/{self.repository}/commits/{encoded_sha}/check-runs"
                f"?filter=all&per_page=100&page={page}"
            )
            if not isinstance(payload, dict) or not isinstance(
                payload.get("check_runs"), list
            ):
                raise ReleaseGateError("GitHub check-runs response is invalid")
            page_items = payload["check_runs"]
            if any(not isinstance(item, dict) for item in page_items):
                raise ReleaseGateError("GitHub check-run entry is invalid")
            result.extend(page_items)
            if len(page_items) < 100:
                return result
        raise ReleaseGateError("GitHub check-runs response exceeds page limit")

    def workflow_run(self, run_id: int) -> dict[str, Any]:
        payload = self.get(f"/repos/{self.repository}/actions/runs/{run_id}")
        if not isinstance(payload, dict):
            raise ReleaseGateError("GitHub workflow-run response is invalid")
        return payload


def _load_policy(path: Path) -> tuple[str, int]:
    try:
        payload = _strict_json(path.read_bytes(), label="release security policy")
    except OSError as exc:
        raise ReleaseGateError("release security policy is unavailable") from exc
    governance = payload.get("governance") if isinstance(payload, dict) else None
    if not isinstance(governance, dict):
        raise ReleaseGateError("release security policy governance is invalid")
    context = payload.get("requiredStatusContext")
    nested_context = governance.get("requiredStatusContext")
    integration_id = governance.get("requiredStatusIntegrationId")
    if (
        not isinstance(context, str)
        or context != nested_context
        or context != WORKFLOW_NAME
        or isinstance(integration_id, bool)
        or not isinstance(integration_id, int)
        or integration_id < 1
    ):
        raise ReleaseGateError("release security policy gate identity is invalid")
    return context, integration_id


def verify_workflow_identity(candidate: Path, trusted: Path) -> str:
    documents: list[bytes] = []
    for path in (candidate, trusted):
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ReleaseGateError("release gate workflow must be a regular file")
            if metadata.st_size < 1 or metadata.st_size > MAX_WORKFLOW_BYTES:
                raise ReleaseGateError("release gate workflow size is invalid")
            documents.append(path.read_bytes())
        except OSError as exc:
            raise ReleaseGateError("release gate workflow is unavailable") from exc
    if documents[0] != documents[1]:
        raise ReleaseGateError(
            "candidate release gate workflow differs from trusted publisher bytes"
        )
    return hashlib.sha256(documents[0]).hexdigest()


def _positive_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _run_id_from_details_url(details_url: object, *, repository: str) -> int | None:
    if not isinstance(details_url, str):
        return None
    match = re.fullmatch(
        rf"https://github\.com/{re.escape(repository)}/actions/runs/([1-9][0-9]*)/job/[1-9][0-9]*",
        details_url,
    )
    return int(match.group(1)) if match is not None else None


def verify_release_gate(
    *,
    repository: str,
    component_sha: str,
    required_context: str,
    required_integration_id: int,
    workflow_sha256: str,
    check_runs: list[dict[str, Any]],
    workflow_run: Callable[[int], dict[str, Any]],
) -> dict[str, object]:
    if (
        not REPOSITORY.fullmatch(repository)
        or not SHA.fullmatch(component_sha)
        or not SHA256.fullmatch(workflow_sha256)
    ):
        raise ReleaseGateError("release gate source identity is invalid")
    matching: list[dict[str, Any]] = []
    for check in check_runs:
        app = check.get("app")
        if (
            check.get("name") == required_context
            and check.get("head_sha") == component_sha
            and isinstance(app, dict)
            and app.get("id") == required_integration_id
            and app.get("slug") == GITHUB_ACTIONS_APP_SLUG
            and _positive_integer(check.get("id")) is not None
        ):
            matching.append(check)
    if not matching:
        raise ReleaseGateError("exact component SHA has no trusted release gate check")

    # A failed/cancelled rerun must supersede an older success. GitHub check IDs
    # are monotonic; bind publication to the newest exact context+integration.
    check = max(matching, key=lambda item: int(item["id"]))
    check_id = int(check["id"])
    if check.get("status") != "completed" or check.get("conclusion") != "success":
        raise ReleaseGateError("latest trusted release gate check did not succeed")
    suite = check.get("check_suite")
    check_suite_id = (
        _positive_integer(suite.get("id")) if isinstance(suite, dict) else None
    )
    run_id = _run_id_from_details_url(check.get("details_url"), repository=repository)
    if check_suite_id is None or run_id is None:
        raise ReleaseGateError("trusted release gate check lacks workflow identity")

    run = workflow_run(run_id)
    run_repository = run.get("repository")
    if (
        run.get("id") != run_id
        or run.get("check_suite_id") != check_suite_id
        or run.get("head_sha") != component_sha
        or run.get("name") != WORKFLOW_NAME
        or run.get("path") != WORKFLOW_PATH
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        # PR check-runs report the branch head SHA even though checkout tests
        # refs/pull/<n>/merge. Publication may therefore consume only the
        # explicit post-merge dispatch whose checkout HEAD is the landed
        # component commit on main.
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "main"
        or not isinstance(run_repository, dict)
        or run_repository.get("full_name") != repository
    ):
        raise ReleaseGateError("trusted check does not belong to the exact release workflow run")
    return {
        "componentSha": component_sha,
        "workflow": WORKFLOW_PATH,
        "workflowSha256": workflow_sha256,
        "workflowRunId": run_id,
        "checkRunId": check_id,
        "checkSuiteId": check_suite_id,
        "context": required_context,
        "integrationId": required_integration_id,
    }


def _append_outputs(path: Path, evidence: dict[str, object]) -> None:
    lines = (
        f"gate_run_id={evidence['workflowRunId']}\n"
        f"gate_check_id={evidence['checkRunId']}\n"
        f"gate_check_suite_id={evidence['checkSuiteId']}\n"
        f"gate_workflow_sha256={evidence['workflowSha256']}\n"
    )
    try:
        with path.open("a", encoding="utf-8") as output:
            output.write(lines)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise ReleaseGateError("cannot persist release gate workflow outputs") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-workflow", type=Path, required=True)
    parser.add_argument("--trusted-workflow", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        context, integration_id = _load_policy(arguments.policy)
        workflow_sha256 = verify_workflow_identity(
            arguments.candidate_workflow,
            arguments.trusted_workflow,
        )
        api = GitHubApi(arguments.repository, os.environ.get("GITHUB_TOKEN", ""))
        evidence = verify_release_gate(
            repository=arguments.repository,
            component_sha=arguments.component_sha,
            required_context=context,
            required_integration_id=integration_id,
            workflow_sha256=workflow_sha256,
            check_runs=api.check_runs(arguments.component_sha),
            workflow_run=api.workflow_run,
        )
        if arguments.github_output is not None:
            _append_outputs(arguments.github_output, evidence)
    except ReleaseGateError as exc:
        parser.error(str(exc))
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
