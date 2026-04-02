from __future__ import annotations

import argparse
import os
import sys

from nuvion_app.config import (
    DEFAULT_PORT,
    load_env,
    load_template,
    read_env,
    resolve_config_path,
    setup_config,
    write_env,
)
from nuvion_app.model_store import (
    DEFAULT_MODEL_POINTER,
    DEFAULT_MODEL_PRESIGN_TTL_SECONDS,
    DEFAULT_MODEL_SERVER_BASE_URL,
    DEFAULT_MODEL_PROFILE,
    anomalyclip_text_features_path,
    anomalyclip_triton_repository_path,
    pull_model_from_server,
    resolve_default_model_dir,
)
from nuvion_app.runtime.config_guard import ensure_runtime_config, guard_config, print_report
from nuvion_app.runtime.inference_mode import normalize_backend, normalize_siglip_device


_BACKEND_CHOICES = ("triton", "siglip", "mps", "none")
_SIGLIP_DEVICE_CHOICES = ("auto", "mps", "cuda", "cpu")


def _merge_template_defaults(existing: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    lines, fields = load_template()
    merged = dict(existing)
    for field in fields:
        key = field["key"]
        if key not in merged:
            merged[key] = field["default"]
    return lines, merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuv-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Configure agent settings")
    setup_parser.add_argument("--config", help="Path to config env file")
    setup_parser.add_argument("--web", action="store_true", help="Force web setup")
    setup_parser.add_argument("--cli", action="store_true", help="Force CLI setup")
    setup_parser.add_argument("--qr", action="store_true", help="Use QR pairing (headless)")
    setup_parser.add_argument("--host", default="127.0.0.1", help="Web UI bind address")
    setup_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web UI port")
    setup_parser.add_argument("--no-open", action="store_true", help="Do not open browser")
    setup_parser.add_argument("--advanced", action="store_true", help="Prompt all fields")

    run_parser = subparsers.add_parser("run", help="Run inference service")
    run_parser.add_argument("--config", help="Path to config env file")
    run_parser.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        help="Override backend for this run: triton|siglip|mps(alias for siglip+mps)|none",
    )
    run_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode using the built-in MVTec slideshow source",
    )
    run_parser.add_argument(
        "--siglip-device",
        choices=_SIGLIP_DEVICE_CHOICES,
        help="SigLIP device preference when backend is siglip: auto|mps|cuda|cpu",
    )

    pull_parser = subparsers.add_parser(
        "pull-model",
        help="Download model artifacts for Triton/AnomalyCLIP runtime via server presigned URLs",
    )
    pull_parser.add_argument("--config", help="Path to config env file")
    pull_parser.add_argument(
        "--pointer",
        default=os.getenv("NUVION_MODEL_POINTER", DEFAULT_MODEL_POINTER),
        help="Pointer identifier resolved by the server, e.g. anomalyclip/prod",
    )
    pull_parser.add_argument(
        "--server-base-url",
        default=os.getenv("NUVION_MODEL_SERVER_BASE_URL", os.getenv("NUVION_SERVER_BASE_URL", DEFAULT_MODEL_SERVER_BASE_URL)),
        help="NUV-BE base URL for model presign API (used when --source server)",
    )
    pull_parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=int(os.getenv("NUVION_MODEL_PRESIGN_TTL_SECONDS", str(DEFAULT_MODEL_PRESIGN_TTL_SECONDS))),
        help="Requested signed URL TTL seconds (used when --source server)",
    )
    pull_parser.add_argument(
        "--access-token",
        default=os.getenv("NUVION_MODEL_SERVER_ACCESS_TOKEN", ""),
        help="Optional bearer token for presign API (used when --source server)",
    )
    pull_parser.add_argument(
        "--username",
        default=os.getenv("NUVION_DEVICE_USERNAME", ""),
        help="Device username for /auth/login fallback (used when --source server and no token)",
    )
    pull_parser.add_argument(
        "--password",
        default=os.getenv("NUVION_DEVICE_PASSWORD", ""),
        help="Device password for /auth/login fallback (used when --source server and no token)",
    )
    pull_parser.add_argument(
        "--local-dir",
        default=os.getenv("NUVION_MODEL_LOCAL_DIR", ""),
        help="Destination directory (default: ~/.cache/nuvion/models/<pointer>)",
    )
    pull_parser.add_argument(
        "--profile",
        choices=("full", "runtime", "light"),
        default=os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
        help="Download profile. 'runtime' is enough for Triton + text features",
    )

    path_parser = subparsers.add_parser("config-path", help="Print resolved config path")
    path_parser.add_argument("--config", help="Path to config env file")

    inference_parser = subparsers.add_parser("set-inference", help="Save inference backend/device into config")
    inference_parser.add_argument("--config", help="Path to config env file")
    inference_parser.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        required=True,
        help="triton|siglip|mps(alias for siglip+mps)|none",
    )
    inference_parser.add_argument(
        "--siglip-device",
        choices=_SIGLIP_DEVICE_CHOICES,
        help="SigLIP device preference: auto|mps|cuda|cpu",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Validate/migrate agent config")
    doctor_parser.add_argument("--config", help="Path to config env file")
    doctor_parser.add_argument("--fix", action="store_true", help="Apply automatic migration fixes")

    return parser


def main() -> None:
    if sys.version_info < (3, 10):
        sys.stderr.write("Python 3.10+ is required.\n")
        sys.exit(2)

    # Load default env early so parser defaults can resolve from config file.
    try:
        load_env()
    except Exception:
        pass

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "setup":
        if args.web and args.cli:
            parser.error("--web and --cli are mutually exclusive")
        if args.qr and args.web:
            parser.error("--qr and --web are mutually exclusive")
        use_web = None
        qr = args.qr
        if args.web:
            use_web = True
        if args.cli:
            use_web = False
            qr = False
        setup_config(
            config_path=args.config,
            use_web=use_web,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            advanced=args.advanced,
            qr=qr,
        )
        return

    if args.command == "run":
        config_path = resolve_config_path(args.config)
        load_env(str(config_path))
        if args.demo:
            os.environ["NUVION_DEMO_MODE"] = "true"
        if args.backend:
            raw_backend = args.backend.strip().lower()
            os.environ["NUVION_ZSAD_BACKEND"] = normalize_backend(raw_backend, default="triton")
            if raw_backend == "mps" and not args.siglip_device:
                os.environ["NUVION_ZERO_SHOT_DEVICE"] = "mps"
        if args.siglip_device:
            os.environ["NUVION_ZERO_SHOT_DEVICE"] = normalize_siglip_device(args.siglip_device, default="auto")

        try:
            ensure_runtime_config(config_path=config_path, stage="run", apply_fixes=True)
        except Exception as exc:
            sys.stderr.write(f"Config preflight failed: {exc}\n")
            sys.stderr.write("Run `nuv-agent doctor --fix` and retry.\n")
            sys.exit(2)

        from nuvion_app.runtime.gstreamer_env import configure_gstreamer_environment

        configure_gstreamer_environment()
        from nuvion_app.inference.main import main as run_main

        run_main()
        return

    if args.command == "pull-model":
        load_env(args.config)
        try:
            pointer = (args.pointer or DEFAULT_MODEL_POINTER).strip()
            local_dir = args.local_dir.strip() or str(resolve_default_model_dir(f"server:{pointer}:{args.profile}"))
            server_base_url = (args.server_base_url or os.getenv("NUVION_SERVER_BASE_URL", "")).strip()
            access_token = (args.access_token or "").strip() or None
            username = (args.username or "").strip() or None
            password = (args.password or "").strip() or None

            model_dir, _ = pull_model_from_server(
                server_base_url=server_base_url,
                pointer=pointer,
                profile=args.profile,
                local_dir=local_dir,
                ttl_seconds=args.ttl_seconds,
                access_token=access_token,
                username=username,
                password=password,
            )
        except Exception as exc:
            sys.stderr.write(f"Failed to pull model artifacts: {exc}\n")
            sys.exit(1)

        text_features = anomalyclip_text_features_path(model_dir)
        triton_repo = anomalyclip_triton_repository_path(model_dir)

        sys.stdout.write(f"Model artifacts downloaded to: {model_dir}\n")
        sys.stdout.write("Source: server\n")
        if text_features.exists():
            sys.stdout.write("Suggested env for AnomalyCLIP Triton backend:\n")
            sys.stdout.write("  NUVION_ZSAD_BACKEND=triton\n")
            sys.stdout.write("  NUVION_TRITON_MODE=anomalyclip\n")
            sys.stdout.write("  NUVION_TRITON_MODEL=image_encoder\n")
            sys.stdout.write("  NUVION_TRITON_INPUT=image\n")
            sys.stdout.write("  NUVION_TRITON_IMAGE_FEATURES_OUTPUT=image_features\n")
            sys.stdout.write(f"  NUVION_TRITON_TEXT_FEATURES={text_features}\n")
            if triton_repo.exists():
                sys.stdout.write(f"  # Triton model repository: {triton_repo}\n")
        else:
            sys.stdout.write(
                "Downloaded profile does not include onnx/text_features.npy. "
                "Use --profile runtime or --profile full.\n"
            )
        return

    if args.command == "config-path":
        path = resolve_config_path(args.config)
        sys.stdout.write(str(path))
        sys.stdout.write("\n")
        return

    if args.command == "set-inference":
        config_path = resolve_config_path(args.config)
        existing = read_env(config_path)
        lines, values = _merge_template_defaults(existing)

        raw_backend = args.backend.strip().lower()
        backend = normalize_backend(raw_backend, default="triton")
        values["NUVION_ZSAD_BACKEND"] = backend

        if args.siglip_device:
            values["NUVION_ZERO_SHOT_DEVICE"] = normalize_siglip_device(args.siglip_device, default="auto")
        elif raw_backend == "mps":
            values["NUVION_ZERO_SHOT_DEVICE"] = "mps"
        else:
            values["NUVION_ZERO_SHOT_DEVICE"] = normalize_siglip_device(
                values.get("NUVION_ZERO_SHOT_DEVICE", "auto"),
                default="auto",
            )

        write_env(config_path, lines, values)
        sys.stdout.write(f"Saved inference config: {config_path}\n")
        sys.stdout.write(f"  NUVION_ZSAD_BACKEND={values['NUVION_ZSAD_BACKEND']}\n")
        sys.stdout.write(f"  NUVION_ZERO_SHOT_DEVICE={values['NUVION_ZERO_SHOT_DEVICE']}\n")
        return

    if args.command == "doctor":
        config_path = resolve_config_path(args.config)
        report = guard_config(config_path=config_path, apply_fixes=args.fix)
        print_report(report)
        if report.ok:
            sys.stdout.write("[DOCTOR] result: OK\n")
            return
        sys.stderr.write("[DOCTOR] result: FAILED\n")
        sys.exit(2)

    parser.print_help()


if __name__ == "__main__":
    main()
