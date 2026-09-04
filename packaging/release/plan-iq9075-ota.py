#!/usr/bin/env python3
"""Verify published IQ9075 releases and create a deterministic sequence reservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from nuvion_app.runtime.release_bom import (
    ReleaseKeyring,
    ReleaseTarget,
    VerifiedReleaseBom,
    build_release_bom_v2_payload,
    load_signed_release_bom,
)
from nuvion_updater.trust import load_release_keyring


SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
VERSION_BOM_OBJECT = re.compile(
    r"^releases/([0-9]+\.[0-9]+\.[0-9]+)/release-bom\.json$"
)
MAX_LISTED_OBJECTS = 10_000
MAX_LIST_PAGES = 100
MAX_HTTP_BYTES = 128 * 1024


class SequencePlanError(RuntimeError):
    pass


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SequencePlanError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SequencePlanError(f"invalid {label}: {exc}") from exc


def _read_policy(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_bytes(path.read_bytes(), label="security policy")
    except OSError as exc:
        raise SequencePlanError(f"cannot read security policy: {exc}") from exc
    if not isinstance(value, dict) or value.get("schemaVersion") != 1:
        raise SequencePlanError("security policy is invalid")
    return value


def _bounded_get(url: str, *, limit: int = MAX_HTTP_BYTES) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "nuv-release-sequence-gate/1"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise SequencePlanError(f"remote object exceeds limit: {url}")
            payload = response.read(limit + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise SequencePlanError(f"cannot fetch remote object {url}: {exc}") from exc
    if len(payload) > limit:
        raise SequencePlanError(f"remote object exceeds limit: {url}")
    return payload


def _object_url(bucket: str, object_name: str) -> str:
    encoded = urllib.parse.quote(object_name, safe="/")
    return f"https://storage.googleapis.com/{bucket}/{encoded}"


def list_version_boms(bucket: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    page_token: str | None = None
    item_count = 0
    for _ in range(MAX_LIST_PAGES):
        query: dict[str, str] = {
            "prefix": "releases/",
            "fields": "items(name,generation),nextPageToken",
            "maxResults": "1000",
        }
        if page_token is not None:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?"
            + urllib.parse.urlencode(query)
        )
        payload = _strict_json_bytes(_bounded_get(url), label="GCS object listing")
        if not isinstance(payload, dict):
            raise SequencePlanError("GCS object listing must be an object")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise SequencePlanError("GCS object listing items are invalid")
        for item in items:
            item_count += 1
            if item_count > MAX_LISTED_OBJECTS:
                raise SequencePlanError("GCS release object listing is unexpectedly large")
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise SequencePlanError("GCS release object metadata is invalid")
            match = VERSION_BOM_OBJECT.fullmatch(item["name"])
            if match is None:
                continue
            version = match.group(1)
            if version in versions:
                raise SequencePlanError(f"duplicate version BOM object: {version}")
            generation = item.get("generation")
            if not isinstance(generation, str) or not generation.isdigit():
                raise SequencePlanError("GCS release generation is invalid")
            versions[version] = generation
        page_token_value = payload.get("nextPageToken")
        if page_token_value is None:
            return versions
        if not isinstance(page_token_value, str) or not page_token_value:
            raise SequencePlanError("GCS object listing page token is invalid")
        page_token = page_token_value
    raise SequencePlanError("GCS release object listing exceeded page limit")


def _load_remote_signed_bom(
    *,
    bucket: str,
    version: str,
    generation: str,
    release_keyring: ReleaseKeyring,
) -> VerifiedReleaseBom:
    bom_url = _object_url(bucket, f"releases/{version}/release-bom.json")
    separator = "&" if "?" in bom_url else "?"
    bom_bytes = _bounded_get(f"{bom_url}{separator}generation={generation}")
    signature_bytes = _bounded_get(
        _object_url(bucket, f"releases/{version}/release-bom.json.sig")
    )
    with tempfile.TemporaryDirectory(prefix="nuv-ota-sequence-verify-") as temporary:
        root = Path(temporary)
        bom_path = root / "release-bom.json"
        signature_path = root / "release-bom.json.sig"
        bom_path.write_bytes(bom_bytes)
        signature_path.write_bytes(signature_bytes)
        try:
            return load_signed_release_bom(
                bom_path,
                signature_path,
                release_keyring=release_keyring,
            )
        except Exception as exc:
            raise SequencePlanError(
                f"published release {version} failed independent signature verification"
            ) from exc


def _target_from_policy(policy: Mapping[str, Any]) -> tuple[ReleaseTarget, str, str, dict[str, Any]]:
    iq = policy.get("iq9075")
    if not isinstance(iq, dict):
        raise SequencePlanError("IQ9075 security policy is invalid")
    bucket = iq.get("bucket")
    trust_domain = iq.get("trustDomain")
    target = iq.get("target")
    public_keyring_file = iq.get("publicKeyringFile")
    public_keyring_sha256 = iq.get("publicKeyringSha256")
    publisher_key_id = iq.get("publisherKeyId")
    if (
        not isinstance(bucket, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,221}[a-z0-9]", bucket)
        or not isinstance(trust_domain, str)
        or not trust_domain
        or not isinstance(target, dict)
        or set(target)
        != {"productModel", "platformProfile", "hardwareRevision", "architecture"}
        or not all(isinstance(value, str) and value for value in target.values())
        or not isinstance(public_keyring_file, str)
        or re.fullmatch(r"trusted-release-keyrings/[A-Za-z0-9._-]+\.json", public_keyring_file)
        is None
        or not isinstance(public_keyring_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_keyring_sha256) is None
        or not isinstance(publisher_key_id, str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", publisher_key_id) is None
    ):
        raise SequencePlanError("IQ9075 release target policy is invalid")
    return (
        ReleaseTarget(
            product_model=target["productModel"],
            platform_profile=target["platformProfile"],
            hardware_revision=target["hardwareRevision"],
            architecture=target["architecture"],
        ),
        bucket,
        trust_domain,
        iq,
    )


def _verify_pinned_keyring(
    *, policy_path: Path, keyring_path: Path, iq_policy: Mapping[str, Any]
) -> str:
    expected = policy_path.parent / iq_policy["publicKeyringFile"]
    supplied = keyring_path if keyring_path.is_absolute() else Path.cwd() / keyring_path
    if supplied.is_symlink() or expected.is_symlink():
        raise SequencePlanError("release public keyring must not be a symlink")
    try:
        if supplied.resolve(strict=True) != expected.resolve(strict=True):
            raise SequencePlanError("release public keyring is not the policy-pinned file")
        raw = supplied.read_bytes()
    except OSError as exc:
        raise SequencePlanError("cannot read policy-pinned release public keyring") from exc
    if not raw or len(raw) > 64 * 1024:
        raise SequencePlanError("release public keyring size is invalid")
    if hashlib.sha256(raw).hexdigest() != iq_policy["publicKeyringSha256"]:
        raise SequencePlanError("release public keyring digest does not match policy")
    payload = _strict_json_bytes(raw, label="release public keyring")
    publisher_key_id = iq_policy["publisherKeyId"]
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "trustDomain", "keys"}
        or payload.get("schemaVersion") != 1
        or payload.get("trustDomain") != iq_policy["trustDomain"]
        or not isinstance(payload.get("keys"), dict)
        or set(payload["keys"]) != {publisher_key_id}
    ):
        raise SequencePlanError("release public keyring identity does not match policy")
    return publisher_key_id


def _promotion_is_complete(
    *,
    bucket: str,
    bom: VerifiedReleaseBom,
    legacy_baseline: Mapping[str, Any] | None,
) -> bool:
    expected_digest = f"sha256:{bom.bom_digest}"
    if isinstance(legacy_baseline, Mapping) and (
        legacy_baseline.get("agentVersion") == bom.agent_version
        and legacy_baseline.get("releaseSequence") == bom.release_sequence
        and legacy_baseline.get("bomDigest") == expected_digest
    ):
        return True
    url = _object_url(
        bucket, f"releases/promotions/iq9075/{bom.agent_version}.json"
    )
    try:
        payload = _strict_json_bytes(_bounded_get(url), label="OTA promotion marker")
    except SequencePlanError:
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("schemaVersion") == 1
        and payload.get("kind") == "nuvion-iq9075-ota-promotion"
        and payload.get("agentVersion") == bom.agent_version
        and payload.get("releaseSequence") == bom.release_sequence
        and payload.get("bomDigest") == expected_digest
        and payload.get("componentSha") == bom.component_sha
        and payload.get("artifactDigest") == f"sha256:{bom.artifact_sha256}"
        and payload.get("status") == "PROMOTED"
    )


def plan_sequence(
    *,
    policy_path: Path,
    keyring_path: Path,
    artifact_path: Path,
    version: str,
    component_sha: str,
    requested_sequence: int,
    config_schema: str,
    min_updater_version: str,
    built_at: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not SEMVER.fullmatch(version):
        raise SequencePlanError("agent version must be exact SemVer")
    if not COMMIT_SHA.fullmatch(component_sha):
        raise SequencePlanError("component SHA must be full lowercase SHA-1")
    if requested_sequence < 1 or requested_sequence > 2**63 - 1:
        raise SequencePlanError("requested sequence is outside signed 64-bit range")
    if not config_schema.isdigit() or int(config_schema) < 1:
        raise SequencePlanError("config schema is invalid")
    if not SEMVER.fullmatch(min_updater_version):
        raise SequencePlanError("minimum updater version is invalid")
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise SequencePlanError("release artifact must be a regular non-symlink file")

    policy = _read_policy(policy_path)
    target, bucket, trust_domain, iq_policy = _target_from_policy(policy)
    publisher_key_id = _verify_pinned_keyring(
        policy_path=policy_path,
        keyring_path=keyring_path,
        iq_policy=iq_policy,
    )
    try:
        release_keyring = load_release_keyring(
            keyring_path,
            expected_trust_domain=trust_domain,
            require_root_owner=False,
        )
        expected_payload = build_release_bom_v2_payload(
            bom_id=f"nuv-agent-{version}-iq9075-aarch64",
            release_sequence=requested_sequence,
            agent_version=version,
            component_sha=component_sha,
            config_schema=config_schema,
            min_updater_version=min_updater_version,
            targets=[target],
            artifact_path=artifact_path,
            artifact_kind="agent-bundle",
            built_at=built_at,
        )
    except Exception as exc:
        raise SequencePlanError(
            "public keyring or requested release identity is invalid"
        ) from exc
    objects = list_version_boms(bucket)
    releases: dict[str, VerifiedReleaseBom] = {}
    sequences: dict[int, VerifiedReleaseBom] = {}
    for published_version, generation in sorted(objects.items()):
        bom = _load_remote_signed_bom(
            bucket=bucket,
            version=published_version,
            generation=generation,
            release_keyring=release_keyring,
        )
        if (
            bom.agent_version != published_version
            or bom.release_sequence is None
            or bom.targets != (target,)
        ):
            raise SequencePlanError(
                f"published release {published_version} has inconsistent identity"
            )
        previous = sequences.get(bom.release_sequence)
        if previous is not None and previous.bom_digest != bom.bom_digest:
            raise SequencePlanError(
                f"releaseSequence {bom.release_sequence} is already equivocated"
            )
        sequences[bom.release_sequence] = bom
        releases[published_version] = bom

    existing = releases.get(version)
    latest_sequence = max(sequences, default=0)
    latest = sequences.get(latest_sequence)
    mode: str
    if existing is not None:
        comparable = {
            "schemaVersion": existing.schema_version,
            "bomId": existing.bom_id,
            "bomDigest": f"sha256:{existing.bom_digest}",
            "releaseSequence": existing.release_sequence,
            "agentVersion": existing.agent_version,
            "componentSha": existing.component_sha,
            "configSchema": existing.config_schema,
            "minUpdaterVersion": existing.min_updater_version,
            "targets": [value.to_payload() for value in existing.targets],
            "artifact": {
                "name": existing.artifact_name,
                "kind": existing.artifact_kind,
                "sha256": existing.artifact_sha256,
                "sizeBytes": existing.artifact_size_bytes,
            },
            "builtAt": existing.built_at,
        }
        if comparable != expected_payload:
            raise SequencePlanError(
                "existing version path does not exactly match the requested release"
            )
        mode = "idempotent-existing"
    else:
        if requested_sequence != latest_sequence + 1:
            raise SequencePlanError(
                f"requested releaseSequence must be latest+1 ({latest_sequence + 1})"
            )
        if latest is not None and not _promotion_is_complete(
            bucket=bucket,
            bom=latest,
            legacy_baseline=iq_policy.get("legacyPromotedBaseline"),
        ):
            raise SequencePlanError(
                "latest signed IQ9075 release is not promoted; rerun it before advancing"
            )
        mode = "new"

    reservation = {
        "schemaVersion": 1,
        "kind": "nuvion-iq9075-sequence-reservation",
        "releaseSequence": requested_sequence,
        "agentVersion": version,
        "componentSha": component_sha,
        "expectedBomDigest": expected_payload["bomDigest"],
        "target": target.to_payload(),
        "artifact": expected_payload["artifact"],
    }
    reservation_bytes = (
        json.dumps(reservation, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    outputs = {
        "mode": mode,
        "latest_sequence": str(latest_sequence),
        "latest_version": latest.agent_version if latest is not None else "none",
        "expected_bom_digest": expected_payload["bomDigest"],
        "reservation_sha256": hashlib.sha256(reservation_bytes).hexdigest(),
        "reservation_object": f"releases/reservations/iq9075/{requested_sequence}.json",
        "publisher_key_id": publisher_key_id,
    }
    return reservation, outputs


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(document)
    except FileExistsError as exc:
        raise SequencePlanError("reservation output already exists") from exc


def _write_github_output(path: Path, values: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise SequencePlanError(f"invalid GitHub output: {key}")
            output.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--keyring", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--release-sequence", required=True)
    parser.add_argument("--config-schema", required=True)
    parser.add_argument("--min-updater-version", required=True)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        if not POSITIVE_INTEGER.fullmatch(arguments.release_sequence):
            raise SequencePlanError("release sequence must be a positive integer")
        reservation, outputs = plan_sequence(
            policy_path=arguments.policy,
            keyring_path=arguments.keyring,
            artifact_path=arguments.artifact,
            version=arguments.version,
            component_sha=arguments.component_sha,
            requested_sequence=int(arguments.release_sequence),
            config_schema=arguments.config_schema,
            min_updater_version=arguments.min_updater_version,
            built_at=arguments.built_at,
        )
        _write_new(arguments.output.resolve(), reservation)
        if arguments.github_output is not None:
            _write_github_output(arguments.github_output, outputs)
        print(json.dumps(outputs, sort_keys=True, separators=(",", ":")))
    except SequencePlanError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
