from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Optional

DEFAULT_MODEL_SOURCE = "server"
DEFAULT_MODEL_PROFILE = "runtime"
DEFAULT_MODEL_POINTER = "anomalyclip/prod"
DEFAULT_MODEL_PRESIGN_TTL_SECONDS = 300
DEFAULT_MODEL_SERVER_BASE_URL = "https://api.nuvion-dev.plaidai.io"
DEFAULT_MODEL_GCS_POINTER_URI = "gs://nuv-model/pointers/anomalyclip/prod.json"

_PROFILE_KEYS: dict[str, list[str]] = {
    "runtime": ["text_features", "plan", "triton_config", "manifest"],
    "light": ["text_features", "manifest"],
    "full": [
        "onnx",
        "text_features",
        "pytorch",
        "plan",
        "triton_config",
        "onnx_eval",
        "triton_eval",
        "manifest",
    ],
}

_DEFAULT_LOCAL_PATHS = {
    "text_features": "onnx/text_features.npy",
    "onnx": "onnx/image_encoder_simplified.onnx",
    "pytorch": "pytorch/epoch_15.pth",
    "plan": "triton/model_repository/image_encoder/1/model.plan",
    "triton_config": "triton/model_repository/image_encoder/config.pbtxt",
    "onnx_eval": "metadata/onnx_eval_results.txt",
    "triton_eval": "metadata/triton_eval_results.txt",
    "manifest": "metadata/gcs_manifest.json",
}


def resolve_default_model_dir(identifier: str) -> Path:
    root = Path(os.getenv("NUVION_MODEL_DIR", "~/.cache/nuvion/models")).expanduser()
    safe_id = identifier.replace("/", "__").replace(":", "_")
    return (root / safe_id).resolve()


def _resolve_local_dir(identifier: str, local_dir: Optional[str]) -> Path:
    if local_dir:
        return Path(local_dir).expanduser().resolve()
    return resolve_default_model_dir(identifier)


def _ensure_profile(profile: str) -> None:
    valid = set(_PROFILE_KEYS.keys())
    if profile not in valid:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"Unsupported profile '{profile}'. Expected one of: {allowed}")


def _run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(detail) from exc


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    trimmed = uri[5:]
    bucket, _, object_path = trimmed.partition("/")
    if not bucket:
        raise ValueError(f"Invalid gs:// URI (missing bucket): {uri}")
    return bucket, object_path


def _gcs_uri(bucket: str, object_path: str) -> str:
    object_path = object_path.lstrip("/")
    return f"gs://{bucket}/{object_path}"


def _gcs_cat_json(uri: str) -> dict[str, Any]:
    result = _run_command(["gcloud", "storage", "cat", uri], capture_output=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in pointer: {uri}") from exc


def _copy_gcs_object(src_uri: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["gcloud", "storage", "cp", src_uri, str(dst_path)])


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _http_json(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc.reason}") from exc


def _extract_api_data(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def _login_for_access_token(server_base_url: str, username: str, password: str) -> str:
    if not username or not password:
        raise RuntimeError(
            "Access token is missing. Provide --access-token or set device credentials "
            "(NUVION_DEVICE_USERNAME/NUVION_DEVICE_PASSWORD)."
        )

    login_url = f"{_normalize_base_url(server_base_url)}/auth/login"
    response = _http_json("POST", login_url, payload={"username": username, "password": password})
    data = _extract_api_data(response)
    token = data.get("accessToken")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("Login succeeded but accessToken is missing in response.")
    return token.strip()


def _resolve_profile_keys(pointer: dict[str, Any], profile: str) -> list[str]:
    profiles = pointer.get("profiles")
    if isinstance(profiles, dict):
        candidate = profiles.get(profile)
        if isinstance(candidate, list):
            keys = [str(item).strip() for item in candidate if str(item).strip()]
            if keys:
                return keys
    return list(_PROFILE_KEYS[profile])


def _resolve_local_rel_path(key: str, path_hint: Optional[str] = None) -> str:
    preset = _DEFAULT_LOCAL_PATHS.get(key)
    if preset:
        return preset

    if path_hint:
        filename = PurePosixPath(path_hint).name
        if filename:
            return f"extras/{key}/{filename}"

    return f"extras/{key}"


def _artifact_path_from_pointer(artifact: Any, key: str) -> tuple[str, Optional[str], Optional[int]]:
    if isinstance(artifact, str):
        path = artifact.strip()
        if not path:
            raise RuntimeError(f"Artifact '{key}' path is empty")
        return path, None, None

    if isinstance(artifact, dict):
        path = str(artifact.get("path") or "").strip()
        if not path:
            raise RuntimeError(f"Artifact '{key}' path is empty")
        sha256 = artifact.get("sha256")
        size_bytes = artifact.get("sizeBytes")
        normalized_sha = str(sha256).strip() if sha256 is not None else None
        normalized_size = size_bytes if isinstance(size_bytes, int) else None
        return path, normalized_sha, normalized_size

    raise RuntimeError(f"Artifact '{key}' format is invalid")


def _validate_download_integrity(path: Path, expected_sha256: Optional[str], expected_size_bytes: Optional[int], key: str) -> None:
    if expected_size_bytes is not None:
        actual_size = path.stat().st_size
        if actual_size != expected_size_bytes:
            raise RuntimeError(
                f"Artifact size mismatch for '{key}'. expected={expected_size_bytes} actual={actual_size}"
            )

    if expected_sha256:
        actual_sha256 = _sha256_file(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Artifact sha256 mismatch for '{key}'. expected={expected_sha256} actual={actual_sha256}"
            )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _download_http_file(url: str, dst_path: Path, timeout: int = 120, max_retries: int = 3) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dst_path.with_suffix(dst_path.suffix + ".part")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                if status < 200 or status >= 300:
                    raise RuntimeError(f"HTTP status={status}")
                with part_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            part_path.replace(dst_path)
            return
        except Exception as exc:
            last_error = exc
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))

    raise RuntimeError(f"Download failed after {max_retries} attempts: {url} ({last_error})")


def pull_model_from_gcs(
    pointer_uri: str,
    local_dir: Optional[str] = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> tuple[Path, dict[str, Any]]:
    _ensure_profile(profile)

    bucket, _ = _parse_gs_uri(pointer_uri)
    pointer = _gcs_cat_json(pointer_uri)

    artifacts = pointer.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Pointer JSON must include 'artifacts' object")

    runtime_layout = pointer.get("runtime_layout")
    local_paths: dict[str, str] = {}
    if isinstance(runtime_layout, dict):
        lp = runtime_layout.get("local_paths")
        if isinstance(lp, dict):
            local_paths = {str(k): str(v) for k, v in lp.items()}

    target_dir = _resolve_local_dir(identifier=pointer_uri, local_dir=local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    required_keys = _resolve_profile_keys(pointer, profile)
    missing = [key for key in required_keys if key not in artifacts]
    if missing:
        raise RuntimeError(f"Pointer is missing required artifacts for profile '{profile}': {', '.join(missing)}")

    downloaded: list[dict[str, Any]] = []
    for key in required_keys:
        src_obj, expected_sha256, expected_size = _artifact_path_from_pointer(artifacts[key], key)
        src_uri = _gcs_uri(bucket, src_obj)
        local_rel = local_paths.get(key, _resolve_local_rel_path(key, src_obj))
        dst = (target_dir / local_rel).resolve()
        _copy_gcs_object(src_uri=src_uri, dst_path=dst)
        _validate_download_integrity(dst, expected_sha256, expected_size, key)
        downloaded.append({
            "key": key,
            "src": src_uri,
            "dst": str(dst),
            "sha256": expected_sha256,
            "sizeBytes": expected_size,
        })

    metadata_dir = (target_dir / "metadata").resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_dir / "gcs_pointer.json", pointer)
    _write_json(metadata_dir / "downloaded_from_gcs.json", downloaded)

    return target_dir, pointer


def pull_model_from_server(
    server_base_url: str,
    pointer: str = DEFAULT_MODEL_POINTER,
    profile: str = DEFAULT_MODEL_PROFILE,
    local_dir: Optional[str] = None,
    ttl_seconds: int = DEFAULT_MODEL_PRESIGN_TTL_SECONDS,
    access_token: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> tuple[Path, dict[str, Any]]:
    _ensure_profile(profile)

    normalized_pointer = pointer.strip()
    if not normalized_pointer:
        raise ValueError("pointer is required for source=server")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    base_url = _normalize_base_url(server_base_url.strip())
    if not base_url:
        raise ValueError("server_base_url is required for source=server")

    token = (access_token or "").strip()
    if not token:
        token = _login_for_access_token(
            server_base_url=base_url,
            username=(username or "").strip(),
            password=(password or "").strip(),
        )

    request_payload = {
        "pointer": normalized_pointer,
        "profile": profile,
        "ttlSeconds": int(ttl_seconds),
    }
    headers = {"Authorization": f"Bearer {token}"}
    response_payload = _http_json(
        "POST",
        f"{base_url}/devices/models/presign",
        payload=request_payload,
        headers=headers,
    )
    data = _extract_api_data(response_payload)

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Server response must include artifacts array")

    artifact_by_key: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if key:
            artifact_by_key[key] = item

    required_keys = _PROFILE_KEYS[profile]
    missing = [key for key in required_keys if key not in artifact_by_key]
    if missing:
        raise RuntimeError(f"Presign response is missing required artifacts for profile '{profile}': {', '.join(missing)}")

    identifier = f"server:{normalized_pointer}:{profile}"
    target_dir = _resolve_local_dir(identifier=identifier, local_dir=local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    for key in required_keys:
        artifact = artifact_by_key[key]
        url = str(artifact.get("url") or "").strip()
        if not url:
            raise RuntimeError(f"Presign artifact '{key}' is missing url")

        path_hint = str(artifact.get("path") or "").strip() or None
        expected_sha256 = str(artifact.get("sha256") or "").strip()
        if not expected_sha256:
            raise RuntimeError(f"Presign artifact '{key}' is missing sha256")

        expected_size: Optional[int] = None
        size_bytes = artifact.get("sizeBytes")
        if isinstance(size_bytes, int):
            expected_size = size_bytes

        local_rel = _resolve_local_rel_path(key, path_hint)
        dst = (target_dir / local_rel).resolve()

        _download_http_file(url=url, dst_path=dst)
        _validate_download_integrity(dst, expected_sha256, expected_size, key)

        downloaded.append(
            {
                "key": key,
                "url": url,
                "dst": str(dst),
                "path": path_hint,
                "sha256": expected_sha256,
                "sizeBytes": expected_size,
                "expiresAt": artifact.get("expiresAt"),
            }
        )

    metadata_dir = (target_dir / "metadata").resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_dir / "server_presign_response.json", data)
    _write_json(metadata_dir / "downloaded_from_server.json", downloaded)

    return target_dir, data


def anomalyclip_text_features_path(model_dir: Path) -> Path:
    return model_dir / "onnx" / "text_features.npy"


def anomalyclip_triton_repository_path(model_dir: Path) -> Path:
    return model_dir / "triton" / "model_repository"
