"""Local-only Fleet selection for the pinned experimental VisualAD HTP backend.

The signed CONFIG_APPLY digest identifies fleet-model.json, not the checkpoint.
No network resolver, executable model code, or CPU fallback is provided here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from nuvion_app.runtime.settings_overlay import SHA256_PATTERN
from nuvion_app.runtime.visualad import _opened_regular_file
from nuvion_app.runtime.visualad_htp import VisualADHTPAnomalyDetector, verify_manifest

VISUALAD_FLEET_POINTER = "visualad/iq9075-htp-demo"
STORE_ENV = "NUVION_VISUALAD_HTP_FLEET_STORE"
_MODEL_OWNER_UID = 0
_FRESH_SECONDS = 60.0


def _immutable_identity(path: Path, *, directory: bool = False) -> tuple[int, ...]:
    info = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_type(info.st_mode)
        or info.st_uid != _MODEL_OWNER_UID
        or info.st_mode & 0o022
    ):
        raise ValueError(
            "Fleet model paths must be root-owned and immutable to the service"
        )
    if directory:
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid)
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
        info.st_uid,
    )


def _store_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or str(path) != str(value):
        raise ValueError("VisualAD Fleet store must be a canonical absolute directory")
    if path.resolve(strict=True) != path:
        raise ValueError("VisualAD Fleet store cannot contain symlink components")
    _immutable_identity(path, directory=True)
    return path


@dataclass(frozen=True)
class VisualADFleetSelection:
    store: Path
    directory: Path
    pointer: str
    digest: str
    manifest_sha256: str

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def fingerprint(self) -> tuple[tuple[int, ...], ...]:
        _store_path(self.store)
        return (
            _immutable_identity(self.store, directory=True),
            _immutable_identity(self.directory, directory=True),
            *(
                _immutable_identity(self.directory / name)
                for name in ("fleet-model.json", "manifest.json", "visualad.onnx")
            ),
        )


def select_visualad_fleet_model(
    store: str | Path, pointer: str, digest: str, *, verify_artifacts: bool = False
) -> VisualADFleetSelection:
    """Bind an allowed pointer to actual wrapper bytes in a preprovisioned store."""
    if pointer != VISUALAD_FLEET_POINTER or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError("Unsupported VisualAD Fleet pointer or digest")
    root = _store_path(store)
    directory = root / digest.removeprefix("sha256:")
    _immutable_identity(directory, directory=True)
    wrapper = directory / "fleet-model.json"
    _immutable_identity(wrapper)
    with _opened_regular_file(wrapper) as (fd, info):
        if not 1 <= info.st_size <= 4096:
            raise ValueError("VisualAD Fleet wrapper size is invalid")
        raw = os.read(fd, 4097)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("VisualAD Fleet wrapper digest mismatch")
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "backend", "pointer", "manifest"}
        or type(payload["schemaVersion"]) is not int
        or payload["schemaVersion"] != 1
        or payload["backend"] != "visualad_htp"
        or payload["pointer"] != pointer
    ):
        raise ValueError("VisualAD Fleet wrapper contract mismatch")
    manifest = payload["manifest"]
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"path", "sha256"}
        or manifest["path"] != "manifest.json"
        or not isinstance(manifest["sha256"], str)
        or not SHA256_PATTERN.fullmatch("sha256:" + manifest["sha256"])
    ):
        raise ValueError("VisualAD Fleet inner manifest pin is invalid")
    selection = VisualADFleetSelection(
        root, directory, pointer, digest, manifest["sha256"]
    )
    before = selection.fingerprint()
    if verify_artifacts:
        verify_manifest(str(selection.manifest_path), selection.manifest_sha256)
        if before != selection.fingerprint():
            raise ValueError("VisualAD Fleet bundle changed during validation")
    return selection


def build_visualad_fleet_detector(environ):
    from nuvion_app.runtime.visualad_fleet_start import require_fleet_boot_guard

    require_fleet_boot_guard(environ)
    selection = select_visualad_fleet_model(
        environ[STORE_ENV],
        environ.get("NUVION_MODEL_POINTER", ""),
        environ.get("NUVION_MODEL_DIGEST", ""),
    )
    return FleetVisualADHTPAnomalyDetector(
        selection=selection,
        state_dir=environ.get("NUVION_VISUALAD_HTP_STATE_DIR", ""),
        threshold=float(environ.get("NUVION_VISUALAD_THRESHOLD", "0.0")),
    )


def provision_visualad_fleet_model(
    store: str | Path, source_manifest: str | Path, manifest_sha256: str
) -> VisualADFleetSelection:
    """Copy an operator-pinned local model into a new digest directory only.

    This is an explicit provisioning tool, never called by command handling.
    A partial failed publication is retained and cannot be overwritten/reused.
    """
    root = _store_path(store)
    source = Path(source_manifest)
    if (
        not source.is_absolute()
        or source.name != "manifest.json"
        or source.resolve(strict=True) != source
    ):
        raise ValueError("Provisioning requires a canonical absolute source manifest")
    _immutable_identity(source.parent, directory=True)
    _immutable_identity(source)
    _immutable_identity(source.parent / "visualad.onnx")
    verify_manifest(str(source), manifest_sha256)
    wrapper = {
        "schemaVersion": 1,
        "backend": "visualad_htp",
        "pointer": VISUALAD_FLEET_POINTER,
        "manifest": {"path": "manifest.json", "sha256": manifest_sha256},
    }
    raw = (json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    destination = root / digest.removeprefix("sha256:")
    if os.path.lexists(destination):
        raise FileExistsError(
            "VisualAD Fleet digest directory already exists; refusing overwrite"
        )
    with tempfile.TemporaryDirectory(prefix=".visualad-stage-", dir=root) as temporary:
        staged = Path(temporary)
        for name in ("manifest.json", "visualad.onnx"):
            with (
                _opened_regular_file(source.parent / name) as (fd, _info),
                (staged / name).open("xb") as output,
            ):
                while chunk := os.read(fd, 1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            (staged / name).chmod(0o644)
        verify_manifest(str(staged / "manifest.json"), manifest_sha256)
        with (staged / "fleet-model.json").open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        (staged / "fleet-model.json").chmod(0o644)
        # mkdir is exclusive even if another operator provisions concurrently.
        # Publish the wrapper last: incomplete contents can never select a model.
        destination.mkdir(mode=0o755)
        for name in ("visualad.onnx", "manifest.json", "fleet-model.json"):
            os.link(staged / name, destination / name)
        for directory in (destination, root):
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return select_visualad_fleet_model(
        root, VISUALAD_FLEET_POINTER, digest, verify_artifacts=True
    )


class FleetVisualADHTPAnomalyDetector(VisualADHTPAnomalyDetector):
    """Publish loaded identity only after a real, fresh QNN-only model result."""

    def __init__(
        self, *, selection: VisualADFleetSelection, state_dir: str, threshold=0.0
    ):
        super().__init__(
            enabled=True,
            manifest_path=str(selection.manifest_path),
            manifest_sha256=selection.manifest_sha256,
            state_dir=state_dir,
            threshold=threshold,
        )
        self.selection = selection
        self._fleet_loaded_selection = None
        self._fleet_loaded_fingerprint = None
        self._fleet_result_at = None

    def _load(self):
        current = select_visualad_fleet_model(
            self.selection.store, self.selection.pointer, self.selection.digest
        )
        if current != self.selection:
            raise ValueError("VisualAD Fleet selection changed before loading")
        before = current.fingerprint()
        super()._load()
        if before != current.fingerprint():
            raise ValueError("VisualAD Fleet bundle changed while loading")
        self._fleet_loaded_selection = current
        self._fleet_loaded_fingerprint = before

    def _fail(self, exc):
        self._fleet_result_at = None
        super()._fail(exc)

    def classify(self, frame_rgb):
        if self._fleet_loaded_selection is not None:
            try:
                if self.selection.fingerprint() != self._fleet_loaded_fingerprint:
                    raise ValueError("VisualAD Fleet loaded bundle changed on disk")
            except (OSError, ValueError) as exc:
                self._fail(exc)
                return None
        result = super().classify(frame_rgb)
        if result is not None:
            self._fleet_result_at = time.monotonic()
        return result

    def startup_pending(self) -> bool:
        # Never wait for the compilation lock in a heartbeat/reconcile thread.
        return self.enabled and not self.ready and self.last_error is None

    def loaded_model_proof(self) -> dict[str, str] | None:
        selection = self._fleet_loaded_selection
        timestamp = self._fleet_result_at
        if (
            not self.enabled
            or not self.ready
            or self.last_error is not None
            or selection != self.selection
            or timestamp is None
            or not 0 <= time.monotonic() - timestamp <= _FRESH_SECONDS
            or self.execution_provider != "QNNExecutionProvider/HTP"
            or self.inference_count < 1
            or self.loaded_manifest_sha256 != selection.manifest_sha256
            or (self.manifest_path, self.manifest_sha256)
            != (str(selection.manifest_path), selection.manifest_sha256)
            or self.graph_sha256 != getattr(self, "_verified_graph_sha256", None)
        ):
            return None
        try:
            if selection.fingerprint() != self._fleet_loaded_fingerprint:
                return None
        except (OSError, ValueError):
            return None
        return {"pointer": selection.pointer, "digest": selection.digest}

    def verify_model(self, desired) -> dict[str, str]:
        proof = self.loaded_model_proof()
        if proof is None or proof != dict(desired):
            raise RuntimeError(
                "VisualAD Fleet has no matching fresh loaded model proof"
            )
        selected = select_visualad_fleet_model(
            self.selection.store,
            proof["pointer"],
            proof["digest"],
            verify_artifacts=True,
        )
        if (
            selected != self._fleet_loaded_selection
            or self.loaded_model_proof() != proof
        ):
            raise RuntimeError("VisualAD Fleet model changed during apply verification")
        return proof


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("provision", "verify"))
    parser.add_argument("--store", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.action == "provision":
        if not args.source_manifest or not args.manifest_sha256 or args.digest:
            parser.error(
                "provision requires --source-manifest and --manifest-sha256 only"
            )
        selection = provision_visualad_fleet_model(
            args.store, args.source_manifest, args.manifest_sha256
        )
    else:
        if not args.digest or args.source_manifest or args.manifest_sha256:
            parser.error("verify requires --digest only")
        selection = select_visualad_fleet_model(
            args.store, VISUALAD_FLEET_POINTER, args.digest, verify_artifacts=True
        )
    print(
        json.dumps(
            {
                "pointer": selection.pointer,
                "digest": selection.digest,
                "directory": str(selection.directory),
                "manifestSha256": selection.manifest_sha256,
                "verification": "LOCAL_ARTIFACT_BYTES_ONLY",
                "inferenceReady": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
