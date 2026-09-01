from __future__ import annotations

import argparse
import base64
import binascii
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nuvion_app.runtime.release_bom import (
    ReleaseTarget,
    build_release_bom_payload,
    build_release_bom_signature,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
    canonical_release_bom_signature_json,
    verify_release_artifact,
    verify_release_bom,
)

_SIGNING_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


def _absolute_output_path(raw_path: str) -> Path:
    output = Path(raw_path).expanduser()
    return output if output.is_absolute() else Path.cwd() / output


def _ensure_distinct_output(
    parser: argparse.ArgumentParser,
    output: Path,
    artifact: Path,
    *,
    option: str,
) -> None:
    if output == artifact or (output.exists() and output.samefile(artifact)):
        parser.error(f"{option} must not overwrite --artifact")
    if output.is_symlink():
        parser.error(f"{option} must not be a symbolic link")


def _write_immutable_document(
    parser: argparse.ArgumentParser,
    *,
    output: Path,
    document: str,
    label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file():
            parser.error(f"{label} output must be a regular file")
        try:
            existing = output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read existing {label} output: {exc}")
        if existing != document:
            parser.error(f"refusing to overwrite an existing {label}")
        return

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeError(f"{label} output appeared during generation") from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_target(raw_target: str) -> ReleaseTarget:
    values = raw_target.split(":")
    if len(values) != 4 or any(not value for value in values):
        raise ValueError(
            "target must be PRODUCT_MODEL:PLATFORM_PROFILE:"
            "HARDWARE_REVISION:ARCHITECTURE"
        )
    return ReleaseTarget(
        product_model=values[0],
        platform_profile=values[1],
        hardware_revision=values[2],
        architecture=values[3],
    )


def _read_private_key_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat signing private key: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("signing private key path must not be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("signing private key path must be a regular file")
    if metadata.st_size > 64 * 1024:
        raise ValueError("signing private key exceeds size limit")
    if metadata.st_mode & 0o077:
        raise ValueError(
            "signing private key must not be accessible by group or other users"
        )
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, open_flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            material = handle.read(64 * 1024 + 1)
            after = os.fstat(handle.fileno())
        final_metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot read signing private key: {exc}") from exc
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if identity != (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) or identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or identity != (
        final_metadata.st_dev,
        final_metadata.st_ino,
        final_metadata.st_mode,
        final_metadata.st_size,
        final_metadata.st_mtime_ns,
        final_metadata.st_ctime_ns,
    ):
        raise ValueError("signing private key changed while being read")
    if len(material) > 64 * 1024:
        raise ValueError("signing private key exceeds size limit")
    if not material:
        raise ValueError("signing private key must not be empty")
    return material


def _private_key_from_material(material: bytes) -> Ed25519PrivateKey:
    if len(material) == 32:
        try:
            return Ed25519PrivateKey.from_private_bytes(material)
        except ValueError as exc:
            raise ValueError("invalid raw Ed25519 signing private key") from exc
    private_key: object | None = None
    try:
        private_key = serialization.load_pem_private_key(material, password=None)
    except (TypeError, ValueError):
        try:
            private_key = serialization.load_der_private_key(material, password=None)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid PEM/DER Ed25519 signing private key") from exc
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("release signing private key must be Ed25519")
    return private_key


def _load_private_key_from_environment(variable_name: str) -> Ed25519PrivateKey:
    if not _SIGNING_ENV_NAME.fullmatch(variable_name):
        raise ValueError("signing private key environment variable name is invalid")
    raw_value = os.environ.get(variable_name)
    if raw_value is None or not raw_value.strip():
        raise ValueError(
            f"signing private key environment variable {variable_name} is empty"
        )
    material = raw_value.encode("utf-8")
    try:
        return _private_key_from_material(material)
    except ValueError:
        try:
            decoded = base64.b64decode(raw_value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "signing private key environment value must be PEM or canonical base64"
            ) from exc
        if base64.b64encode(decoded).decode("ascii") != raw_value:
            raise ValueError(
                "signing private key environment value must use canonical base64"
            )
        return _private_key_from_material(decoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a content-addressed NUVION release BOM"
    )
    parser.add_argument("--schema-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--bom-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--config-schema", required=True)
    parser.add_argument("--updater-version")
    parser.add_argument("--platform-profile", action="append")
    parser.add_argument("--release-sequence", type=int)
    parser.add_argument("--min-updater-version")
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "v2 exact target as PRODUCT_MODEL:PLATFORM_PROFILE:"
            "HARDWARE_REVISION:ARCHITECTURE"
        ),
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-kind", required=True)
    parser.add_argument("--built-at")
    parser.add_argument("--output", required=True)
    parser.add_argument("--signature-output")
    parser.add_argument("--signing-key-id")
    signing_source = parser.add_mutually_exclusive_group()
    signing_source.add_argument("--signing-private-key")
    signing_source.add_argument("--signing-private-key-env")
    args = parser.parse_args()

    built_at = args.built_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    artifact = Path(args.artifact).expanduser()
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    output = _absolute_output_path(args.output)
    _ensure_distinct_output(parser, output, artifact, option="--output")

    signature_document: str | None = None
    signature_output: Path | None = None
    if args.schema_version == 1:
        if not args.updater_version:
            parser.error("--updater-version is required for schema v1")
        if not args.platform_profile:
            parser.error("--platform-profile is required for schema v1")
        if any(
            value is not None
            for value in (
                args.release_sequence,
                args.min_updater_version,
                args.target,
                args.signature_output,
                args.signing_key_id,
                args.signing_private_key,
                args.signing_private_key_env,
            )
        ):
            parser.error("v2 target/signing options are not valid for schema v1")
        payload = build_release_bom_payload(
            bom_id=args.bom_id,
            agent_version=args.version,
            component_sha=args.component_sha.lower(),
            config_schema=args.config_schema,
            updater_version=args.updater_version,
            platform_profiles=args.platform_profile,
            artifact_path=artifact,
            artifact_kind=args.artifact_kind,
            built_at=built_at,
        )
    else:
        if args.updater_version is not None or args.platform_profile is not None:
            parser.error(
                "--updater-version/--platform-profile are schema v1 options"
            )
        if args.release_sequence is None:
            parser.error("--release-sequence is required for schema v2")
        if not args.min_updater_version:
            parser.error("--min-updater-version is required for schema v2")
        if not args.target:
            parser.error("at least one --target is required for schema v2")
        if not args.signing_key_id:
            parser.error("--signing-key-id is required for schema v2")
        if not args.signing_private_key and not args.signing_private_key_env:
            parser.error(
                "--signing-private-key or --signing-private-key-env is required "
                "for schema v2"
            )
        try:
            targets = [_parse_target(target) for target in args.target]
            payload = build_release_bom_v2_payload(
                bom_id=args.bom_id,
                release_sequence=args.release_sequence,
                agent_version=args.version,
                component_sha=args.component_sha.lower(),
                config_schema=args.config_schema,
                min_updater_version=args.min_updater_version,
                targets=targets,
                artifact_path=artifact,
                artifact_kind=args.artifact_kind,
                built_at=built_at,
            )
            if args.signing_private_key:
                key_material = _read_private_key_file(
                    Path(args.signing_private_key).expanduser()
                )
                private_key = _private_key_from_material(key_material)
            else:
                private_key = _load_private_key_from_environment(
                    args.signing_private_key_env
                )
            signature_payload = build_release_bom_signature(
                payload,
                key_id=args.signing_key_id,
                private_key=private_key,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        signature_output = _absolute_output_path(
            args.signature_output or f"{output}.sig"
        )
        _ensure_distinct_output(
            parser, signature_output, artifact, option="--signature-output"
        )
        if signature_output == output or (
            signature_output.exists()
            and output.exists()
            and signature_output.samefile(output)
        ):
            parser.error("--signature-output must differ from --output")
        signature_document = canonical_release_bom_signature_json(
            signature_payload
        )

    document = canonical_release_bom_json(payload)
    # Re-read the artifact immediately before publishing immutable sidecars so
    # a replacement after initial hashing cannot produce a stale release BOM.
    verify_release_artifact(verify_release_bom(payload), artifact)
    _write_immutable_document(
        parser, output=output, document=document, label="release BOM"
    )
    if signature_document is not None and signature_output is not None:
        _write_immutable_document(
            parser,
            output=signature_output,
            document=signature_document,
            label="release signature",
        )
    print(output)
    if signature_output is not None:
        print(signature_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
