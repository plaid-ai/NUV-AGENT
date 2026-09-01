from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from pathlib import Path, PurePosixPath

from nuvion_app.runtime.release_bom import VerifiedReleaseBom
from nuvion_updater.errors import UpdaterError, UpdaterSecurityError
from nuvion_updater.util import ensure_directory, fsync_directory, parse_digest

SLOT_MARKER = ".nuvion/release.json"
SLOT_ENTRYPOINT = "bin/nuv-agent"
DEFAULT_INSTALL_DISK_RESERVE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_RELEASE_SLOTS = 8


class ReleaseSlotManager:
    """Stages immutable Agent bundles and atomically switches current/previous."""

    def __init__(
        self,
        install_root: str | Path = "/opt/nuv-agent",
        *,
        require_root_owner: bool = True,
        max_unpacked_bytes: int = 8 * 1024 * 1024 * 1024,
        disk_reserve_bytes: int = DEFAULT_INSTALL_DISK_RESERVE_BYTES,
        max_release_slots: int = DEFAULT_MAX_RELEASE_SLOTS,
    ) -> None:
        if max_unpacked_bytes < 1 or disk_reserve_bytes < 0:
            raise ValueError("slot size and disk reserve limits are invalid")
        if max_release_slots < 2 or max_release_slots > 1_000:
            raise ValueError("max_release_slots must be in [2, 1000]")
        self.install_root = ensure_directory(
            install_root,
            mode=0o755,
            require_root_owner=require_root_owner,
        )
        self.releases_root = ensure_directory(
            self.install_root / "releases",
            mode=0o755,
            require_root_owner=require_root_owner,
        )
        self.require_root_owner = require_root_owner
        self.max_unpacked_bytes = max_unpacked_bytes
        self.disk_reserve_bytes = disk_reserve_bytes
        self.max_release_slots = max_release_slots
        self._purge_stale_incoming()

    def slot_path(self, bom_digest: str) -> Path:
        digest = parse_digest(bom_digest)
        return self.releases_root / digest

    def stage_bundle(
        self,
        *,
        bom: VerifiedReleaseBom,
        bom_path: str | Path,
        signature_path: str | Path,
        artifact_path: str | Path,
    ) -> Path:
        if bom.schema_version != 2 or bom.publisher_key_id is None:
            raise UpdaterSecurityError(
                "UNSIGNED_RELEASE",
                "only publisher-authenticated release-bom-v2 may be staged",
            )
        if bom.artifact_kind != "agent-bundle":
            raise UpdaterSecurityError(
                "UNSUPPORTED_ARTIFACT",
                "OTA activation requires a self-contained agent-bundle",
            )
        if not bom.artifact_name.endswith((".tar", ".tar.gz", ".tgz")):
            raise UpdaterSecurityError(
                "UNSUPPORTED_ARTIFACT",
                "agent-bundle must use tar, tar.gz, or tgz compression",
            )
        final_slot = self.slot_path(f"sha256:{bom.bom_digest}")
        if final_slot.exists():
            self._verify_existing_slot(final_slot, bom)
            return final_slot
        self._assert_slot_capacity()

        incoming = self.releases_root / (
            f".incoming-{bom.bom_digest}-{uuid.uuid4().hex}"
        )
        incoming.mkdir(mode=0o700)
        try:
            self._extract_bundle(
                Path(artifact_path),
                incoming,
                reserved_copy_bytes=bom.artifact_size_bytes,
            )
            entrypoint = incoming / SLOT_ENTRYPOINT
            self._require_regular_file(entrypoint, executable=True)

            metadata_root = incoming / ".nuvion"
            metadata_root.mkdir(mode=0o755)
            self._copy_regular(Path(bom_path), metadata_root / "release-bom.json")
            self._copy_regular(
                Path(signature_path), metadata_root / "release-bom.json.sig"
            )
            self._copy_regular(
                Path(artifact_path), metadata_root / bom.artifact_name
            )
            marker = {
                "schemaVersion": 2,
                "bomDigest": f"sha256:{bom.bom_digest}",
                "agentVersion": bom.agent_version,
                "releaseSequence": bom.release_sequence,
                "artifactDigest": f"sha256:{bom.artifact_sha256}",
                "componentSha": bom.component_sha,
                "configSchema": bom.config_schema,
                "publisherKeyId": bom.publisher_key_id,
            }
            marker_path = incoming / SLOT_MARKER
            marker_bytes = (
                json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._write_new_file(marker_path, marker_bytes, 0o644)
            self._make_tree_immutable(incoming)
            fsync_directory(incoming)
            try:
                os.rename(incoming, final_slot)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                self._verify_existing_slot(final_slot, bom)
                shutil.rmtree(incoming)
            fsync_directory(self.releases_root)
        except BaseException:
            if incoming.exists():
                shutil.rmtree(incoming)
            raise
        self._verify_existing_slot(final_slot, bom)
        return final_slot

    def activate(self, candidate_slot: str | Path) -> tuple[str, str]:
        candidate = self._validate_slot_path(Path(candidate_slot))
        current = self.current_slot()
        candidate_target = self._relative_target(candidate)
        if current == candidate_target:
            previous = self.previous_slot()
            if previous is None:
                raise UpdaterError(
                    "ROLLBACK_UNAVAILABLE", "candidate is active but no previous slot exists"
                )
            return candidate_target, previous
        if current is None:
            raise UpdaterError(
                "ROLLBACK_UNAVAILABLE",
                "OTA activation requires a known-good current slot",
            )
        self._resolve_link_target(current)
        self._atomic_symlink("previous", current)
        self._atomic_symlink("current", candidate_target)
        return candidate_target, current

    def rollback(self) -> tuple[str, str]:
        current = self.current_slot()
        previous = self.previous_slot()
        if current is None or previous is None:
            raise UpdaterError(
                "ROLLBACK_UNAVAILABLE", "current and previous slots are required"
            )
        self._resolve_link_target(previous)
        self._atomic_symlink("current", previous)
        self._atomic_symlink("previous", current)
        return previous, current

    def restore(self, target: str) -> tuple[str, str | None]:
        """Idempotently make the durable known-good target current.

        Unlike swapping ``current``/``previous``, this remains correct if power
        failed after the first symlink write and recovery invokes it again.
        """

        self._resolve_link_target(target)
        current = self.current_slot()
        if current == target:
            return target, self.previous_slot()
        if current is None:
            raise UpdaterError(
                "ROLLBACK_UNAVAILABLE", "current slot is required for restore"
            )
        self._resolve_link_target(current)
        self._atomic_symlink("previous", current)
        self._atomic_symlink("current", target)
        return target, current

    def current_slot(self) -> str | None:
        return self._read_slot_link("current")

    def previous_slot(self) -> str | None:
        return self._read_slot_link("previous")

    def is_active(self, candidate_slot: str | Path) -> bool:
        candidate = self._validate_slot_path(Path(candidate_slot))
        return self.current_slot() == self._relative_target(candidate)

    def relative_target(self, candidate_slot: str | Path) -> str:
        return self._relative_target(Path(candidate_slot))

    def slot_version(self, target: str) -> str:
        resolved = self._resolve_link_target(target)
        if target.startswith("bootstrap/"):
            return target.split("/", 1)[1]
        marker = self._read_release_marker(resolved)
        version = marker.get("agentVersion")
        if not isinstance(version, str) or not version:
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release slot agentVersion is invalid"
            )
        return version

    def release_metadata(self, target: str) -> dict[str, object]:
        if not target.startswith("releases/"):
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release metadata requires a release slot"
            )
        return self._read_release_marker(self._resolve_link_target(target))

    def _extract_bundle(
        self,
        artifact: Path,
        destination: Path,
        *,
        reserved_copy_bytes: int,
    ) -> None:
        self._require_regular_file(artifact)
        try:
            archive = tarfile.open(artifact, mode="r:*")  # noqa: SIM115
        except (tarfile.TarError, OSError) as exc:
            raise UpdaterSecurityError(
                "INVALID_BUNDLE", "agent-bundle must be a readable tar archive"
            ) from exc
        with archive:
            members = archive.getmembers()
            if not members:
                raise UpdaterSecurityError("INVALID_BUNDLE", "agent-bundle is empty")
            if len(members) > 100_000:
                raise UpdaterSecurityError(
                    "INVALID_BUNDLE", "agent-bundle contains too many members"
                )
            total_size = 0
            seen: set[str] = set()
            validated: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    len(member.name) > 1024
                    or len(relative.parts) > 64
                    or relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise UpdaterSecurityError(
                        "UNSAFE_BUNDLE_PATH", "agent-bundle contains an unsafe path"
                    )
                normalized = relative.as_posix()
                if normalized in seen:
                    raise UpdaterSecurityError(
                        "INVALID_BUNDLE", "agent-bundle contains duplicate paths"
                    )
                seen.add(normalized)
                if not (member.isdir() or member.isreg()):
                    raise UpdaterSecurityError(
                        "UNSAFE_BUNDLE_TYPE",
                        "agent-bundle may contain only directories and regular files",
                    )
                if member.isreg():
                    total_size += member.size
                    if total_size > self.max_unpacked_bytes:
                        raise UpdaterSecurityError(
                            "BUNDLE_TOO_LARGE", "agent-bundle unpacked size exceeds limit"
                        )
                validated.append((member, relative))

            free = shutil.disk_usage(self.releases_root).free
            required = total_size + reserved_copy_bytes + self.disk_reserve_bytes
            if free < required:
                raise UpdaterError(
                    "INSUFFICIENT_INSTALL_DISK",
                    "insufficient install filesystem capacity for unpacked slot, "
                    f"artifact copy, and reserve: required={required}, free={free}",
                )

            for member, relative in validated:
                output = destination.joinpath(*relative.parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True, mode=0o755)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(member)
                if source is None:
                    raise UpdaterSecurityError(
                        "INVALID_BUNDLE", "cannot read regular bundle member"
                    )
                mode = 0o755 if member.mode & 0o111 else 0o644
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(output, flags, mode)
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    os.fchmod(descriptor, mode)
                finally:
                    os.close(descriptor)

    def _purge_stale_incoming(self) -> None:
        pattern = re.compile(r"^\.incoming-[0-9a-f]{64}-[0-9a-f]{32}$")
        removed = False
        for candidate in self.releases_root.iterdir():
            if not pattern.fullmatch(candidate.name):
                continue
            metadata = candidate.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_mode & 0o022
                or (self.require_root_owner and metadata.st_uid != 0)
            ):
                raise UpdaterSecurityError(
                    "UNSAFE_INCOMING_SLOT",
                    "stale incoming release slot is unsafe",
                )
            shutil.rmtree(candidate)
            removed = True
        if removed:
            fsync_directory(self.releases_root)

    def _assert_slot_capacity(self) -> None:
        count = 0
        for candidate in self.releases_root.iterdir():
            if not re.fullmatch(r"[0-9a-f]{64}", candidate.name):
                continue
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise UpdaterSecurityError(
                    "INVALID_SLOT",
                    "release slot capacity contains an unsafe entry",
                )
            count += 1
        if count >= self.max_release_slots:
            raise UpdaterError(
                "SLOT_CAPACITY_EXHAUSTED",
                f"release slot limit reached ({self.max_release_slots}); "
                "operator GC is required",
            )

    @staticmethod
    def _copy_regular(source: Path, destination: Path) -> None:
        ReleaseSlotManager._require_regular_file(source)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o644)
        try:
            source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            source_descriptor = os.open(source, source_flags)
            try:
                with (
                    os.fdopen(source_descriptor, "rb", closefd=False) as source_file,
                    os.fdopen(descriptor, "wb", closefd=False) as target_file,
                ):
                    shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
                    target_file.flush()
                    os.fsync(target_file.fileno())
            finally:
                os.close(source_descriptor)
            os.fchmod(descriptor, 0o644)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_new_file(path: Path, payload: bytes, mode: int) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, mode)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_regular_file(path: Path, *, executable: bool = False) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise UpdaterSecurityError(
                "MISSING_FILE", f"required file is unavailable: {path.name}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise UpdaterSecurityError(
                "UNSAFE_FILE", f"{path.name} must be a regular file"
            )
        if executable and not metadata.st_mode & 0o111:
            raise UpdaterSecurityError(
                "INVALID_BUNDLE", f"{path.name} must be executable"
            )

    @staticmethod
    def _make_tree_immutable(root: Path) -> None:
        for directory, subdirs, files in os.walk(root):
            for name in subdirs:
                os.chmod(Path(directory) / name, 0o755)
            for name in files:
                path = Path(directory) / name
                mode = path.stat(follow_symlinks=False).st_mode
                os.chmod(path, 0o755 if mode & 0o111 else 0o644)
        os.chmod(root, 0o755)

    def _verify_existing_slot(
        self, slot: Path, bom: VerifiedReleaseBom
    ) -> None:
        validated = self._validate_slot_path(slot)
        marker_path = validated / SLOT_MARKER
        self._require_regular_file(marker_path)
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UpdaterSecurityError(
                "SLOT_COLLISION", "existing slot marker is invalid"
            ) from exc
        expected = {
            "schemaVersion": 2,
            "bomDigest": f"sha256:{bom.bom_digest}",
            "agentVersion": bom.agent_version,
            "releaseSequence": bom.release_sequence,
            "artifactDigest": f"sha256:{bom.artifact_sha256}",
            "componentSha": bom.component_sha,
            "configSchema": bom.config_schema,
            "publisherKeyId": bom.publisher_key_id,
        }
        if marker != expected:
            raise UpdaterSecurityError(
                "SLOT_COLLISION", "existing slot does not match the verified release"
            )
        self._require_regular_file(validated / SLOT_ENTRYPOINT, executable=True)

    def _read_release_marker(self, slot: Path) -> dict[str, object]:
        validated = self._validate_slot_path(slot)
        marker_path = validated / SLOT_MARKER
        self._require_regular_file(marker_path)
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release slot marker is invalid"
            ) from exc
        expected_fields = {
            "schemaVersion",
            "bomDigest",
            "agentVersion",
            "releaseSequence",
            "artifactDigest",
            "componentSha",
            "configSchema",
            "publisherKeyId",
        }
        if not isinstance(marker, dict) or set(marker) != expected_fields:
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release slot marker fields are invalid"
            )
        return marker

    def _validate_slot_path(self, slot: Path) -> Path:
        try:
            metadata = slot.lstat()
            resolved = slot.resolve(strict=True)
        except OSError as exc:
            raise UpdaterSecurityError("INVALID_SLOT", "release slot is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise UpdaterSecurityError("INVALID_SLOT", "release slot must be a directory")
        if resolved.parent != self.releases_root:
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release slot must be directly under the fixed release root"
            )
        if not resolved.name or len(resolved.name) != 64:
            raise UpdaterSecurityError("INVALID_SLOT", "release slot name is invalid")
        parse_digest(f"sha256:{resolved.name}")
        if self.require_root_owner and metadata.st_uid != 0:
            raise UpdaterSecurityError("INVALID_SLOT", "release slot must be root-owned")
        if metadata.st_mode & 0o022:
            raise UpdaterSecurityError(
                "INVALID_SLOT", "release slot must not be group/other writable"
            )
        return resolved

    def _relative_target(self, slot: Path) -> str:
        validated = self._validate_slot_path(slot)
        return f"releases/{validated.name}"

    def _resolve_link_target(self, target: str) -> Path:
        if target.startswith("releases/") and target.count("/") == 1:
            return self._validate_slot_path(self.install_root / target)
        if target.startswith("bootstrap/") and target.count("/") == 1:
            version = target.split("/", 1)[1]
            if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,99}", version):
                raise UpdaterSecurityError(
                    "INVALID_SLOT_LINK", "bootstrap slot target is unsafe"
                )
            path = self.install_root / target
            try:
                metadata = path.lstat()
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise UpdaterSecurityError(
                    "INVALID_SLOT_LINK", "bootstrap slot is unavailable"
                ) from exc
            bootstrap_root = (self.install_root / "bootstrap").resolve(strict=True)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or resolved.parent != bootstrap_root
                or metadata.st_mode & 0o022
                or (self.require_root_owner and metadata.st_uid != 0)
            ):
                raise UpdaterSecurityError(
                    "INVALID_SLOT_LINK", "bootstrap slot is unsafe"
                )
            self._require_regular_file(resolved / SLOT_ENTRYPOINT, executable=True)
            return resolved
        raise UpdaterSecurityError("INVALID_SLOT_LINK", "slot link target is unsafe")

    def _read_slot_link(self, name: str) -> str | None:
        link = self.install_root / name
        try:
            metadata = link.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise UpdaterSecurityError(
                "INVALID_SLOT_LINK", f"cannot inspect {name} slot link"
            ) from exc
        if not stat.S_ISLNK(metadata.st_mode):
            raise UpdaterSecurityError(
                "INVALID_SLOT_LINK", f"{name} must be a symbolic link"
            )
        target = os.readlink(link)
        self._resolve_link_target(target)
        return target

    def _atomic_symlink(self, name: str, target: str) -> None:
        self._resolve_link_target(target)
        temporary = self.install_root / f".{name}.{uuid.uuid4().hex}"
        os.symlink(target, temporary)
        try:
            os.replace(temporary, self.install_root / name)
            fsync_directory(self.install_root)
        finally:
            if temporary.is_symlink():
                temporary.unlink()
