from __future__ import annotations

import argparse
import os
import sys

from nuvion_app.config import DEFAULT_PORT, load_env, resolve_config_path, setup_config
from nuvion_app.model_store import (
    DEFAULT_MODEL_PROFILE,
    DEFAULT_MODEL_REPO_ID,
    anomalyclip_text_features_path,
    anomalyclip_triton_repository_path,
    pull_model_snapshot,
    resolve_default_model_dir,
)


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

    pull_parser = subparsers.add_parser(
        "pull-model",
        help="Download model artifacts from Hugging Face for Triton/AnomalyCLIP runtime",
    )
    pull_parser.add_argument(
        "--repo-id",
        default=os.getenv("NUVION_MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID),
        help="Hugging Face model repo id",
    )
    pull_parser.add_argument(
        "--revision",
        default=os.getenv("NUVION_MODEL_REVISION", ""),
        help="Optional git revision (branch/tag/commit)",
    )
    pull_parser.add_argument(
        "--local-dir",
        default=os.getenv("NUVION_MODEL_LOCAL_DIR", ""),
        help="Destination directory (default: ~/.cache/nuvion/models/<repo>)",
    )
    pull_parser.add_argument(
        "--profile",
        choices=("full", "runtime", "light"),
        default=os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
        help="Download profile. 'runtime' is enough for Triton + text features",
    )
    pull_parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
        help="Hugging Face token (optional if already logged in)",
    )

    path_parser = subparsers.add_parser("config-path", help="Print resolved config path")
    path_parser.add_argument("--config", help="Path to config env file")

    return parser


def main() -> None:
    if sys.version_info < (3, 10):
        sys.stderr.write("Python 3.10+ is required.\n")
        sys.exit(2)
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
        load_env(args.config)
        from nuvion_app.inference.main import main as run_main

        run_main()
        return

    if args.command == "pull-model":
        revision = args.revision.strip() or None
        local_dir = args.local_dir.strip() or str(resolve_default_model_dir(args.repo_id))
        token = args.token.strip() or None
        try:
            model_dir = pull_model_snapshot(
                repo_id=args.repo_id,
                revision=revision,
                local_dir=local_dir,
                token=token,
                profile=args.profile,
            )
        except Exception as exc:
            sys.stderr.write(f"Failed to pull model artifacts: {exc}\n")
            sys.exit(1)

        text_features = anomalyclip_text_features_path(model_dir)
        triton_repo = anomalyclip_triton_repository_path(model_dir)

        sys.stdout.write(f"Model artifacts downloaded to: {model_dir}\n")
        if text_features.exists():
            sys.stdout.write("Suggested env for AnomalyCLIP Triton backend:\n")
            sys.stdout.write("  NUVION_ZSAD_BACKEND=triton\n")
            sys.stdout.write("  NUVION_TRITON_MODE=anomalyclip\n")
            sys.stdout.write("  NUVION_TRITON_MODEL=image_encoder\n")
            sys.stdout.write("  NUVION_TRITON_INPUT=images\n")
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

    parser.print_help()


if __name__ == "__main__":
    main()
