import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pro_readiness', ROOT / 'packaging/release/verify-release-readiness.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ProReadinessTests(unittest.TestCase):
    def evidence(self):
        return {'schemaVersion': 1, 'target': 'pro-demo', 'version': '0.1.136', 'componentSha': 'a' * 40,
            'qualifiedCapabilities': ['package-install', 'local-demo', 'qnn-htp-inference', 'video-loop', 'service-restart'],
            'excludedCapabilities': ['iq9075-ota', 'camera-csi', 'openrb-motor', 'production-accuracy'],
            'device': {'architecture': 'aarch64', 'profile': 'ventuno-q-prototype'},
            'package': {'name': 'nuv-agent_0.1.136_arm64.deb', 'installedVersion': '0.1.136', 'installedComponentSha': 'a'*40, 'size': 100, 'sha256': 'b'*64},
            'bundleSha256': 'c'*64,
            'runnerSha256': hashlib.sha256((ROOT / 'packaging/release/qualify-pro-demo.py').read_bytes()).hexdigest(),
            'runtimeFiles': {'/opt/nuv-agent/overlays/test.py': 'd'*64},
            'observations': [{'profile': profile, 'durationSeconds': 40,
                'frames': [{'httpStatus': 200, 'sequence': n, 'sha256': f'{n:064x}'} for n in range(20)],
                'inferenceSeconds': [0.45]*30, 'provider': 'QNN/HTP', 'processStable': True, 'runtimeErrors': [], 'loops': 3}
                for profile in ['mvtec', 'video-1', 'video-2']],
            'restart': {'pidChanged': True, 'httpStatus': 200, 'selectedProfile': 'video-1', 'inferenceResumed': True},
            'recordedAt': '2026-10-29T01:00:00+00:00'}

    def verify(self, evidence):
        return module._validate_pro_demo_document(evidence, '0.1.136', 'a'*40)

    def test_passes_bound_demo_stack(self):
        self.assertEqual(self.verify(self.evidence())['qualification_target'], 'pro-demo')

    def test_rejects_invalid_observations(self):
        for field, value in [('provider', 'CPU'), ('loops', 0), ('processStable', False),
                             ('inferenceSeconds', [3.0]*30), ('durationSeconds', 5), ('runtimeErrors', ['Traceback'])]:
            with self.subTest(field=field):
                evidence = self.evidence()
                evidence['observations'][1][field] = value
                with self.assertRaises(module.ReadinessError): self.verify(evidence)

    def test_rejects_stale_frames(self):
        evidence = self.evidence()
        evidence['observations'][0]['frames'][1]['sequence'] = 0
        with self.assertRaises(module.ReadinessError): self.verify(evidence)

    def test_rejects_wrong_source_and_runner(self):
        for key in ('componentSha', 'runnerSha256'):
            evidence = self.evidence()
            evidence[key] = '0'*len(evidence[key])
            with self.assertRaises(module.ReadinessError): self.verify(evidence)

    def test_rejects_claimed_motor_support(self):
        evidence = self.evidence()
        evidence['excludedCapabilities'].remove('openrb-motor')
        with self.assertRaises(module.ReadinessError): self.verify(evidence)

    def test_rejects_missing_profile_and_restart(self):
        evidence = self.evidence()
        evidence['observations'].pop()
        with self.assertRaises(module.ReadinessError): self.verify(evidence)
        evidence = self.evidence()
        evidence['restart']['inferenceResumed'] = False
        with self.assertRaises(module.ReadinessError): self.verify(evidence)

    def test_pro_evidence_cannot_authorize_iq9075(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.json'
            path.write_text(json.dumps({'schemaVersion': 2, 'releases': {'0.1.136': {
                'status': 'READY', 'blockers': [], 'evidence': {'proDemoRuntime': {}}}}}))
            with self.assertRaisesRegex(module.ReadinessError, 'requested target'):
                module.verify_readiness(path, version='0.1.136')

if __name__ == '__main__': unittest.main()
