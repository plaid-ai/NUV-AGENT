from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "packaging/dev/run-macos-mps-qualification.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_macos_mps_qualification", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def reference_payload() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "componentSha": "a" * 40,
        "agentVersion": "0.1.121",
        "modelRevision": RUNNER.MODEL_REVISION,
        "labels": list(RUNNER.LABELS),
        "scores": [0.125, 0.875],
    }


def mps_proof_payload() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "componentSha": "a" * 40,
        "agentVersion": "0.1.121",
        "modelRevision": RUNNER.MODEL_REVISION,
        "labels": list(RUNNER.LABELS),
        "cpuScores": [0.125, 0.875],
        "mpsScores": [0.126, 0.874],
        "offline": True,
        "mpsAvailable": True,
        "device": "mps",
        "dtype": "float16",
        "visionModelClass": "SiglipVisionModel",
        "visionParameterDevices": ["mps"],
        "visionParameterDtypes": ["float16"],
        "persistentStatePacked": True,
        "firstInference": True,
        "repeatCount": 16,
        "physicalMemoryBytes": 16 * 1024**3,
        "recommendedMaxMemoryBytes": 10 * 1024**3,
        "stableAllocatedBytes": 300 * 1024**2,
        "finalAllocatedBytes": 308 * 1024**2,
        "driverAllocatedBytes": 700 * 1024**2,
    }


class MacosMpsQualificationValidationTests(unittest.TestCase):
    def test_cpu_reference_schema_is_exact_and_rejects_nonfinite_scores(self) -> None:
        payload = reference_payload()
        validated = RUNNER.validate_reference(
            payload,
            component_sha="a" * 40,
            agent_version="0.1.121",
        )
        self.assertEqual(validated, payload)

        unexpected = {**payload, "extra": True}
        with self.assertRaisesRegex(
            RUNNER.QualificationError, "schema keys are invalid"
        ):
            RUNNER.validate_reference(
                unexpected,
                component_sha="a" * 40,
                agent_version="0.1.121",
            )

        for invalid_score in (True, float("nan"), float("inf"), -0.01, 1.01):
            invalid = copy.deepcopy(payload)
            invalid["scores"][0] = invalid_score
            with self.assertRaises(RUNNER.QualificationError):
                RUNNER.validate_reference(
                    invalid,
                    component_sha="a" * 40,
                    agent_version="0.1.121",
                )

    def test_formula_receipt_requires_candidate_tap_and_source_build(self) -> None:
        receipt = {
            "built_as_bottle": False,
            "poured_from_bottle": False,
            "loaded_from_api": False,
            "arch": "arm64",
            "built_on": {"os": "Macintosh"},
            "source": {
                "tap": "nuvion/release-gate",
                "spec": "stable",
                "versions": {"stable": "0.1.121"},
            },
        }
        self.assertEqual(
            RUNNER.validate_install_receipt(
                receipt,
                formula="nuvion/release-gate/nuv-agent",
                agent_version="0.1.121",
            ),
            {
                "arch": "arm64",
                "sourceBuilt": True,
                "tap": "nuvion/release-gate",
                "version": "0.1.121",
            },
        )

        for field in ("built_as_bottle", "poured_from_bottle", "loaded_from_api"):
            invalid = copy.deepcopy(receipt)
            invalid[field] = True
            with self.assertRaises(RUNNER.QualificationError):
                RUNNER.validate_install_receipt(
                    invalid,
                    formula="nuvion/release-gate/nuv-agent",
                    agent_version="0.1.121",
                )

        wrong_tap = copy.deepcopy(receipt)
        wrong_tap["source"]["tap"] = "someone/else"
        with self.assertRaisesRegex(
            RUNNER.QualificationError, "tap differs"
        ):
            RUNNER.validate_install_receipt(
                wrong_tap,
                formula="nuvion/release-gate/nuv-agent",
                agent_version="0.1.121",
            )

    def test_mps_proof_enforces_parity_repeats_and_memory_bounds(self) -> None:
        payload = mps_proof_payload()
        self.assertEqual(
            RUNNER.validate_mps_proof(
                payload,
                component_sha="a" * 40,
                agent_version="0.1.121",
            ),
            payload,
        )

        mutations = (
            ("mpsScores", [0.9, 0.1]),
            ("repeatCount", 15),
            ("physicalMemoryBytes", RUNNER.PHYSICAL_MEMORY_FLOOR_BYTES - 1),
            (
                "recommendedMaxMemoryBytes",
                RUNNER.RECOMMENDED_MEMORY_FLOOR_BYTES - 1,
            ),
            (
                "finalAllocatedBytes",
                payload["stableAllocatedBytes"] + RUNNER.GROWTH_LIMIT_BYTES + 1,
            ),
            ("driverAllocatedBytes", RUNNER.DRIVER_MEMORY_LIMIT_BYTES),
            ("visionParameterDevices", ["cpu", "mps"]),
            ("visionParameterDtypes", ["float16", "float32"]),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                invalid = copy.deepcopy(payload)
                invalid[field] = value
                with self.assertRaises(RUNNER.QualificationError):
                    RUNNER.validate_mps_proof(
                        invalid,
                        component_sha="a" * 40,
                        agent_version="0.1.121",
                    )

    def test_success_and_failure_result_schemas_are_canonical(self) -> None:
        proof = RUNNER.validate_mps_proof(
            mps_proof_payload(),
            component_sha="a" * 40,
            agent_version="0.1.121",
        )
        result = RUNNER.build_success_result(
            component_sha="a" * 40,
            agent_version="0.1.121",
            formula="nuvion/release-gate/nuv-agent",
            formula_prefix=Path("/opt/homebrew/Cellar/nuv-agent/0.1.121"),
            proof=proof,
        )
        self.assertEqual(RUNNER.validate_result(result), result)
        encoded = RUNNER.canonical_json(result)
        self.assertEqual(encoded, RUNNER.canonical_json(json.loads(encoded)))
        self.assertNotIn("NaN", encoded)
        self.assertNotIn('": ', encoded)
        self.assertNotIn('", "', encoded)

        failure = RUNNER.build_failure_result(
            RUNNER.QualificationError("formula-missing", "  install\nfirst  ")
        )
        self.assertEqual(
            failure,
            {
                "schemaVersion": 1,
                "qualification": "macos-mps",
                "status": "failed",
                "error": {
                    "code": "formula-missing",
                    "message": "install first",
                },
            },
        )
        invalid = {**failure, "unexpected": True}
        with self.assertRaises(RUNNER.QualificationError):
            RUNNER.validate_result(invalid)

    def test_minimal_worker_environment_does_not_forward_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            with mock.patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "secret",
                    "GITHUB_TOKEN": "secret",
                    "AWS_SECRET_ACCESS_KEY": "secret",
                    "PYTHONPATH": "/untrusted/source",
                },
                clear=False,
            ):
                environment = RUNNER._safe_environment(
                    temp_root,
                    hf_home=temp_root / "huggingface",
                    offline=True,
                )
        self.assertEqual(environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        for forbidden in (
            "HF_TOKEN",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "PYTHONPATH",
            "PYTHONHOME",
        ):
            self.assertNotIn(forbidden, environment)

    def test_help_does_not_require_macos_or_formula_runtime(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            cwd=ROOT,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C",
                "LC_ALL": "C",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("already installed, source-built Homebrew candidate", completed.stdout)
        self.assertIn("--formula", completed.stdout)
        self.assertIn("--timeout-seconds", completed.stdout)

        no_tomllib = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import builtins,runpy,sys;"
                    "real_import=builtins.__import__;"
                    "builtins.__import__=lambda name,*args,**kwargs: "
                    "(_ for _ in ()).throw(ModuleNotFoundError(name)) "
                    "if name=='tomllib' else real_import(name,*args,**kwargs);"
                    f"sys.argv=[{str(RUNNER_PATH)!r},'--help'];"
                    f"runpy.run_path({str(RUNNER_PATH)!r},run_name='__main__')"
                ),
            ],
            cwd=ROOT,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C",
                "LC_ALL": "C",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(no_tomllib.returncode, 0, no_tomllib.stderr)
        self.assertIn("--formula", no_tomllib.stdout)


class MacosMpsQualificationStaticProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")

    def test_locks_exact_candidate_formula_and_source_isolation(self) -> None:
        for required in (
            '[git, "-C", str(repo_root), "rev-parse", "HEAD"]',
            '"--porcelain=v1", "--untracked-files=all"',
            "build_info.COMPONENT_SHA == os.environ[\"CANDIDATE_SHA\"]",
            "build_info.AGENT_VERSION == os.environ[\"CANDIDATE_VERSION\"]",
            '"INSTALL_RECEIPT.json"',
            'value["poured_from_bottle"] is False',
            'value["loaded_from_api"] is False',
            '[str(formula_python), "-I", "-c", source]',
            'Path(sys.prefix).resolve(strict=True) == libexec',
            "package_path.is_relative_to(libexec)",
            '[str(formula_python), "-I", "-m", "pip", "check"]',
        ):
            self.assertIn(required, self.source)
        self.assertNotIn("brew install", self.source)

    def test_locks_reviewed_snapshot_and_offline_oracles(self) -> None:
        self.assertEqual(
            RUNNER.MODEL_REVISION,
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        )
        for required in (
            'MODEL_REPO = "google/siglip2-base-patch16-224"',
            "revision=revision,",
            "token=False,",
            "local_files_only=True,",
            'environment["HF_HUB_OFFLINE"] = "1"',
            'environment["TRANSFORMERS_OFFLINE"] = "1"',
            'assert os.environ["HF_HUB_OFFLINE"] == "1"',
            'assert os.environ["TRANSFORMERS_OFFLINE"] == "1"',
            "AutoModel.from_pretrained(",
            'device_preference="mps"',
            "torch.allclose(observed_scores, reference_scores, rtol=5e-2, atol=1e-5)",
        ):
            self.assertIn(required, self.source)

    def test_locks_real_mps_tensor_repeat_and_memory_proof(self) -> None:
        for required in (
            "assert torch.backends.mps.is_available()",
            'detector._device == "mps"',
            "detector._inference_dtype == torch.float16",
            'detector._model.__class__.__name__ == "SiglipVisionModel"',
            'vision_parameter_devices == ["mps"]',
            'vision_parameter_dtypes == ["float16"]',
            "detector._mps_text_features._base is detector._mps_persistent_state",
            "first = detector.classify(frame)",
            "for _ in range(16):",
            '["/usr/sbin/sysctl", "-n", "hw.memsize"]',
            "torch.mps.recommended_max_memory()",
            "physical_memory >= int(os.environ[\"PHYSICAL_MEMORY_FLOOR_BYTES\"])",
            "recommended_memory >= int(os.environ[\"RECOMMENDED_MEMORY_FLOOR_BYTES\"])",
            "final_bytes <= stable_bytes + 16 * 1024**2",
            "final_bytes < 512 * 1024**2",
            "driver_bytes < 1024 * 1024**2",
        ):
            self.assertIn(required, self.source)

    def test_runner_has_no_github_runner_or_camera_soak_dependency(self) -> None:
        for forbidden in (
            "actions/checkout",
            "runs-on:",
            "self-hosted",
            "macos-14",
            "VideoCapture",
            "camera",
            "soak",
        ):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
