#!/usr/bin/env python3
"""Generate deterministic final promotion markers after every channel succeeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from nuvion_app.runtime.release_bom import load_signed_release_bom
from nuvion_updater.trust import load_release_keyring


SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
MAX_PROMOTION_BYTES = 1024 * 1024


class PromotionError(RuntimeError):
    pass


def _candidate(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _artifact(path: Path) -> dict[str, Any]:
    path = _candidate(path)
    if path.is_symlink():
        raise PromotionError(f"promotion artifact must not be a symlink: {path}")
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PromotionError(f"cannot stat promotion artifact: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise PromotionError(f"promotion artifact must be a nonempty regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(source.fileno())
    final = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if size != before.st_size or identity(before) != identity(after) or identity(before) != identity(final):
        raise PromotionError(f"promotion artifact changed while hashing: {path}")
    return {"name": path.name, "sha256": digest.hexdigest(), "sizeBytes": size}


def _strict_json_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _candidate(path)
    metadata = _artifact(path)
    if metadata["sizeBytes"] > MAX_PROMOTION_BYTES:
        raise PromotionError("distribution promotion exceeds size limit")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PromotionError(f"duplicate distribution promotion member: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PromotionError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PromotionError("distribution promotion is not strict UTF-8 JSON") from exc
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise PromotionError("distribution promotion changed while being parsed")
    if not isinstance(payload, dict):
        raise PromotionError("distribution promotion root must be an object")
    return metadata, payload


def _validate_promoted_artifact(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "sizeBytes"}:
        raise PromotionError(f"distribution {label} identity is invalid")
    name = value.get("name")
    digest = value.get("sha256")
    size = value.get("sizeBytes")
    if (
        not isinstance(name, str)
        or not SAFE_NAME.fullmatch(name)
        or not isinstance(digest, str)
        or not SHA256.fullmatch(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise PromotionError(f"distribution {label} identity is invalid")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(document)
    except FileExistsError as exc:
        raise PromotionError("promotion output already exists") from exc


def build_distribution(arguments: argparse.Namespace) -> dict[str, Any]:
    version = arguments.version
    if not SEMVER.fullmatch(version) or arguments.tag != f"v{version}":
        raise PromotionError("distribution version/tag identity is invalid")
    if not SHA.fullmatch(arguments.component_sha) or not SHA.fullmatch(
        arguments.trusted_publisher_sha
    ):
        raise PromotionError("distribution commit identity is invalid")
    rollback: dict[str, str] | None = None
    if arguments.rollback_version or arguments.rollback_sha256:
        if not SEMVER.fullmatch(arguments.rollback_version or "") or not SHA256.fullmatch(
            arguments.rollback_sha256 or ""
        ):
            raise PromotionError("rollback package identity is incomplete")
        rollback = {
            "agentVersion": arguments.rollback_version,
            "sha256": arguments.rollback_sha256,
        }
    return {
        "schemaVersion": 1,
        "kind": "nuvion-distribution-promotion",
        "status": "PROMOTED",
        "agentVersion": version,
        "releaseTag": arguments.tag,
        "componentSha": arguments.component_sha,
        "trustedPublisherSha": arguments.trusted_publisher_sha,
        "channels": {
            "apt": "PUBLISHED",
            "github": "PUBLISHED",
            "homebrew": "PUBLISHED",
        },
        "artifacts": {
            "pythonSdist": _artifact(arguments.sdist),
            "sdistBom": _artifact(arguments.sdist_bom),
            "iq9075Bundle": _artifact(arguments.bundle),
            "iq9075Deb": _artifact(arguments.deb),
        },
        "rollbackPackage": rollback,
    }


def build_ota(arguments: argparse.Namespace) -> dict[str, Any]:
    distribution, manifest = _strict_json_artifact(arguments.distribution_promotion)
    try:
        keyring = load_release_keyring(
            arguments.keyring,
            expected_trust_domain=arguments.trust_domain,
            require_root_owner=False,
        )
        bom = load_signed_release_bom(
            arguments.bom,
            arguments.signature,
            release_keyring=keyring,
        )
    except Exception as exc:
        raise PromotionError("OTA BOM or public keyring verification failed") from exc
    if bom.release_sequence is None or bom.min_updater_version is None:
        raise PromotionError("OTA promotion requires release-bom-v2")
    expected_manifest_keys = {
        "schemaVersion",
        "kind",
        "status",
        "agentVersion",
        "releaseTag",
        "componentSha",
        "trustedPublisherSha",
        "channels",
        "artifacts",
        "rollbackPackage",
    }
    if set(manifest) != expected_manifest_keys:
        raise PromotionError("distribution promotion fields do not match schema v1")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("kind") != "nuvion-distribution-promotion"
        or manifest.get("status") != "PROMOTED"
        or manifest.get("agentVersion") != bom.agent_version
        or manifest.get("releaseTag") != f"v{bom.agent_version}"
        or manifest.get("componentSha") != bom.component_sha
        or not isinstance(manifest.get("trustedPublisherSha"), str)
        or not SHA.fullmatch(manifest["trustedPublisherSha"])
        or manifest.get("channels")
        != {"apt": "PUBLISHED", "github": "PUBLISHED", "homebrew": "PUBLISHED"}
    ):
        raise PromotionError("distribution promotion identity is inconsistent with OTA BOM")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "pythonSdist",
        "sdistBom",
        "iq9075Bundle",
        "iq9075Deb",
    }:
        raise PromotionError("distribution promotion artifact set is invalid")
    for label, value in artifacts.items():
        _validate_promoted_artifact(value, label=label)
    bundle = artifacts["iq9075Bundle"]
    if bundle != {
        "name": bom.artifact_name,
        "sha256": bom.artifact_sha256,
        "sizeBytes": bom.artifact_size_bytes,
    }:
        raise PromotionError("distribution bundle does not match signed OTA BOM")
    rollback = manifest.get("rollbackPackage")
    if rollback is not None and (
        not isinstance(rollback, dict)
        or set(rollback) != {"agentVersion", "sha256"}
        or not isinstance(rollback.get("agentVersion"), str)
        or not SEMVER.fullmatch(rollback["agentVersion"])
        or not isinstance(rollback.get("sha256"), str)
        or not SHA256.fullmatch(rollback["sha256"])
    ):
        raise PromotionError("distribution rollback package identity is invalid")
    expected_name = f"nuv_agent-{bom.agent_version}-distribution-promotion.json"
    if distribution["name"] != expected_name:
        raise PromotionError("distribution promotion filename is invalid")
    return {
        "schemaVersion": 1,
        "kind": "nuvion-iq9075-ota-promotion",
        "status": "PROMOTED",
        "agentVersion": bom.agent_version,
        "componentSha": bom.component_sha,
        "releaseSequence": bom.release_sequence,
        "bomDigest": f"sha256:{bom.bom_digest}",
        "artifactDigest": f"sha256:{bom.artifact_sha256}",
        "artifactSizeBytes": bom.artifact_size_bytes,
        "publisherKeyId": bom.publisher_key_id,
        "distributionPromotionDigest": f"sha256:{distribution['sha256']}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    distribution = subcommands.add_parser("distribution")
    distribution.add_argument("--version", required=True)
    distribution.add_argument("--tag", required=True)
    distribution.add_argument("--component-sha", required=True)
    distribution.add_argument("--trusted-publisher-sha", required=True)
    distribution.add_argument("--sdist", type=Path, required=True)
    distribution.add_argument("--sdist-bom", type=Path, required=True)
    distribution.add_argument("--bundle", type=Path, required=True)
    distribution.add_argument("--deb", type=Path, required=True)
    distribution.add_argument("--rollback-version")
    distribution.add_argument("--rollback-sha256")
    distribution.add_argument("--output", type=Path, required=True)
    ota = subcommands.add_parser("ota")
    ota.add_argument("--distribution-promotion", type=Path, required=True)
    ota.add_argument("--bom", type=Path, required=True)
    ota.add_argument("--signature", type=Path, required=True)
    ota.add_argument("--keyring", type=Path, required=True)
    ota.add_argument("--trust-domain", required=True)
    ota.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        payload = build_distribution(arguments) if arguments.command == "distribution" else build_ota(arguments)
        _write_new(arguments.output.resolve(), payload)
        print(arguments.output.resolve())
    except (PromotionError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
