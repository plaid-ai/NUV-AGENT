from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nuvion_app.runtime import visualad_htp as htp


class VisualADHTPContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        graph = self.root / "visualad.onnx"
        graph.write_bytes(b"unit test artifact; not an ONNX model")
        self.manifest = {
            "schemaVersion": 1,
            "sourceCommit": htp.SOURCE_COMMIT,
            "checkpointSha256": htp.CHECKPOINT_SHA256,
            "backboneSha256": htp.BACKBONE_SHA256,
            "lnPostPolicy": htp.LN_POST_POLICY,
            "graph": "visualad.onnx",
            "opset": 20,
            "input": htp._INPUT,
            "output": htp._OUTPUT,
            "artifacts": [
                {
                    "path": graph.name,
                    "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                    "size": graph.stat().st_size,
                }
            ],
        }

    def save_manifest(self):
        path = self.root / "manifest.json"
        raw = json.dumps(self.manifest).encode()
        path.write_bytes(raw)
        return str(path), hashlib.sha256(raw).hexdigest()

    def test_manifest_and_all_artifact_bytes_are_pinned(self):
        args = self.save_manifest()
        manifest, graph = htp.verify_manifest(*args)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(graph, self.root / "visualad.onnx")
        graph.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            htp.verify_manifest(*args)

    def test_external_manifest_pin_is_required(self):
        path, _ = self.save_manifest()
        for bad_hash in (None, "", "0" * 64):
            with self.subTest(value=bad_hash), self.assertRaises(ValueError):
                htp.verify_manifest(path, bad_hash)

    def test_official_identity_and_shape_cannot_be_silently_changed(self):
        for key, value in (
            ("sourceCommit", "0" * 40),
            ("checkpointSha256", "0" * 64),
            ("backboneSha256", "0" * 64),
            ("lnPostPolicy", "trained"),
            ("opset", 17),
            ("schemaVersion", True),
            ("output", {"name": "score", "shape": [1], "dtype": "float32"}),
        ):
            old = self.manifest[key]
            self.manifest[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "contract mismatch"),
            ):
                htp.verify_manifest(*self.save_manifest())
            self.manifest[key] = old

    def test_artifact_traversal_duplicates_and_symlinks_are_rejected(self):
        original = self.manifest["artifacts"][0]
        for name in ("../visualad.onnx", "/visualad.onnx", "sub/file", "a\\b", "."):
            self.manifest["artifacts"] = [{**original, "path": name}]
            with self.subTest(name=name), self.assertRaises(ValueError):
                htp.verify_manifest(*self.save_manifest())
        self.manifest["artifacts"] = [original, original]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            htp.verify_manifest(*self.save_manifest())
        self.manifest["artifacts"] = [original]
        graph = self.root / "visualad.onnx"
        graph.rename(self.root / "real.onnx")
        graph.symlink_to(self.root / "real.onnx")
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            htp.verify_manifest(*self.save_manifest())

    def test_group_writable_model_is_not_accepted(self):
        (self.root / "visualad.onnx").chmod(0o664)
        with self.assertRaisesRegex(ValueError, "non-group/world-writable"):
            htp.verify_manifest(*self.save_manifest())

    def test_load_holds_verified_graph_inode_until_session_creation(self):
        path, digest = self.save_manifest()
        detector = htp.VisualADHTPAnomalyDetector(True, path, digest, str(self.root))
        graph = self.root / "visualad.onnx"
        original = graph.read_bytes()

        def create_session(pinned):
            graph.rename(self.root / "original.onnx")
            graph.write_bytes(b"unverified replacement")
            self.assertEqual(Path(pinned).read_bytes(), original)

        # The descriptor retains the verified bytes; the metadata check also
        # rejects the changed inode before accepting the loaded session.
        with (
            mock.patch.object(htp, "_require_embedded_graph") as inspect,
            mock.patch.object(detector, "_load_session", side_effect=create_session),
            self.assertRaisesRegex(ValueError, "changed while reading/loading"),
        ):
            detector._load()
        self.assertFalse(Path(inspect.call_args.args[0]).exists())

    def test_nested_external_data_is_rejected_before_ort_loading(self):
        class Tensor:
            EXTERNAL = 1

            def __init__(self, location=0, entries=()):
                self.data_location = location
                self.external_data = entries

            def ListFields(self):
                return []

        class Message:
            def __init__(self, children):
                self.children = children

            def ListFields(self):
                return [(SimpleNamespace(message_type=Tensor, is_repeated=True), self.children)]

        for tensor, valid in (
            (Tensor(), True),
            (Tensor(location=Tensor.EXTERNAL), False),
            (Tensor(entries=("unlisted.data",)), False),
        ):
            onnx = SimpleNamespace(
                TensorProto=Tensor,
                load=mock.Mock(return_value=Message([Message([tensor])])),
            )
            with mock.patch.object(htp.importlib, "import_module", return_value=onnx):
                if valid:
                    htp._require_embedded_graph("/proc/self/fd/17")
                else:
                    with self.assertRaisesRegex(ValueError, "external-data"):
                        htp._require_embedded_graph("/proc/self/fd/17")
            onnx.load.assert_called_once_with("/proc/self/fd/17", load_external_data=False)

    def test_profile_must_prove_exclusively_qnn_node_execution(self):
        profile = self.root / "profile.json"
        for providers, valid in (
            ([], False),
            (["CPUExecutionProvider"], False),
            (["QNNExecutionProvider", "CPUExecutionProvider"], False),
            (["QNNExecutionProvider"], True),
        ):
            profile.write_text(
                json.dumps([{"args": {"provider": provider}} for provider in providers])
            )
            if valid:
                self.assertEqual(htp._require_qnn_only_profile(str(profile)), providers)
            else:
                with self.assertRaises(RuntimeError):
                    htp._require_qnn_only_profile(str(profile))

    def test_disabled_detector_does_not_import_or_execute_cpu_model(self):
        detector = htp.VisualADHTPAnomalyDetector(False, "", "", "")
        with mock.patch.object(
            htp.importlib,
            "import_module",
            side_effect=AssertionError("must not import"),
        ):
            self.assertEqual(detector.is_anomaly(object()), (False, None))
        self.assertFalse(detector.ready)

    def test_loading_failure_is_sticky_and_never_normal(self):
        detector = htp.VisualADHTPAnomalyDetector(True, "", "", "")
        with mock.patch.object(
            detector, "_load", side_effect=RuntimeError("HTP unavailable")
        ) as load:
            self.assertEqual(detector.is_anomaly(object()), (False, None))
            self.assertEqual(detector.is_anomaly(object()), (False, None))
        load.assert_called_once()
        self.assertFalse(detector.ready)
        self.assertIn("HTP unavailable", detector.last_error)

    def test_failure_clears_previously_loaded_graph_identity(self):
        detector = htp.VisualADHTPAnomalyDetector(True, "", "", "")
        detector.ready = True
        detector.graph_sha256 = "a" * 64
        detector.loaded_manifest_sha256 = "b" * 64
        detector._fail(RuntimeError("lost HTP session"))
        self.assertIsNone(detector.graph_sha256)
        self.assertIsNone(detector.loaded_manifest_sha256)
        self.assertFalse(detector.ready)

    def test_preparation_compiles_once_but_does_not_claim_inference_health(self):
        detector = htp.VisualADHTPAnomalyDetector(True, "", "", "")
        session = mock.Mock()

        def load():
            detector._session = session

        with mock.patch.object(detector, "_load", side_effect=load) as prepare:
            self.assertTrue(detector.prepare())
            self.assertTrue(detector.prepare())
        prepare.assert_called_once()
        session.run.assert_not_called()
        self.assertFalse(detector.ready)
        self.assertIsNone(detector.graph_sha256)
        self.assertEqual(detector.inference_count, 0)

    def test_session_is_npu_only_and_disables_both_kinds_of_cpu_fallback(self):
        path, digest = self.save_manifest()
        detector = htp.VisualADHTPAnomalyDetector(True, path, digest, str(self.root))
        plugin = self.root / "libonnxruntime_providers_qnn.so"
        backend = self.root / "libQnnHtp.so"
        plugin.touch()
        backend.touch()
        options = mock.Mock()
        session = mock.Mock()
        session.get_providers.return_value = [
            "QNNExecutionProvider",
            "CPUExecutionProvider",
        ]
        session.get_inputs.return_value = [
            SimpleNamespace(name="image", shape=[1, 3, 518, 518], type="tensor(float)")
        ]
        session.get_outputs.return_value = [
            SimpleNamespace(
                name="patch_maps", shape=[1, 4, 37, 37], type="tensor(float)"
            )
        ]
        npu = SimpleNamespace(
            ep_name="QNNExecutionProvider", device=SimpleNamespace(type="NPU")
        )
        cpu = SimpleNamespace(
            ep_name="QNNExecutionProvider", device=SimpleNamespace(type="CPU")
        )
        ort = SimpleNamespace(
            __version__="1.26.0",
            register_execution_provider_library=mock.Mock(),
            get_ep_devices=lambda: [cpu, npu],
            OrtHardwareDeviceType=SimpleNamespace(NPU="NPU"),
            SessionOptions=lambda: options,
            InferenceSession=mock.Mock(return_value=session),
        )
        qnn = SimpleNamespace(
            __version__="2.5.0",
            get_library_path=lambda: str(plugin),
            get_qnn_htp_path=lambda: str(backend),
        )
        with (
            mock.patch.object(htp, "_registered_library", None),
            mock.patch.object(htp, "_require_one_htp_bundle"),
            mock.patch.object(htp, "_require_embedded_graph"),
            mock.patch.object(
                htp.importlib,
                "import_module",
                side_effect=lambda name: {"onnxruntime": ort, "onnxruntime_qnn": qnn}[
                    name
                ],
            ),
            mock.patch.dict(
                os.environ,
                {
                    "LD_LIBRARY_PATH": str(self.root),
                    "ADSP_LIBRARY_PATH": str(self.root),
                },
            ),
        ):
            detector._load()
        options.add_session_config_entry.assert_called_once_with(
            "session.disable_cpu_ep_fallback", "1"
        )
        devices, provider_options = options.add_provider_for_devices.call_args.args
        self.assertEqual(devices, [npu])
        self.assertEqual(provider_options["offload_graph_io_quantization"], "0")
        self.assertEqual(provider_options["backend_path"], str(backend))
        session.disable_fallback.assert_called_once()
        session.run.assert_not_called()


try:
    import numpy as np
    import scipy.ndimage  # noqa: F401 - optional numerical dependency check.
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional numpy/scipy postprocessing tests")
class VisualADHTPPostprocessTests(unittest.TestCase):
    def test_constant_maps_preserve_raw_score_without_probability_normalization(self):
        for value in (-0.25, 0.25):
            score, result = htp.postprocess_patch_maps(
                np.full((1, 4, 37, 37), value, dtype=np.float32)
            )
            self.assertAlmostEqual(score, value * 4, places=6)
            self.assertEqual(result.shape, (518, 518))

    def test_bad_map_never_becomes_an_inspection(self):
        for maps in (
            np.zeros((1, 37, 37), dtype=np.float32),
            np.zeros((1, 4, 37, 37), dtype=np.float64),
            np.full((1, 4, 37, 37), np.nan, dtype=np.float32),
            np.full((1, 4, 37, 37), 3.0, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                htp.postprocess_patch_maps(maps)


if __name__ == "__main__":
    unittest.main()
