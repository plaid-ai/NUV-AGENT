from __future__ import annotations

import re
import unittest
from pathlib import Path

from nuvion_app import build_info

ROOT = Path(__file__).resolve().parents[2]


class ReleaseGateTest(unittest.TestCase):
    def test_release_tests_source_and_installed_sdist_before_publish(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-publish.yml").read_text(encoding="utf-8")

        source_test = workflow.index("- name: Test source in clean environment")
        build = workflow.index("- name: Build sdist")
        smoke = workflow.index("- name: Install and smoke-test built sdist")
        publish = workflow.index("- name: Create GitHub release")

        self.assertLess(source_test, build)
        self.assertLess(build, smoke)
        self.assertLess(smoke, publish)
        self.assertIn("-m unittest discover -s tests -p 'test_*.py'", workflow)
        self.assertIn("pip install --no-cache-dir \"$TARBALL\"", workflow)
        self.assertIn("stamp-build-info.py --sha \"$COMPONENT_SHA\" --version \"$VERSION\"", workflow)
        self.assertIn("REQUESTED_TAG: ${{ inputs.tag }}", workflow)
        self.assertIn('[[ ! "$TAG" =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+$ ]]', workflow)
        self.assertNotIn('TAG="${{ inputs.tag }}"', workflow)
        self.assertIn("COMPONENT_SHA=$(git rev-parse HEAD)", workflow)
        self.assertGreaterEqual(workflow.count("stamp-build-info.py"), 2)
        self.assertGreaterEqual(workflow.count("NUV_AGENT_CONFIG: ${{ runner.temp }}/nuv-agent-release-test-"), 2)

    def test_release_candidate_version_is_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        version = match.group(1)

        deb = (ROOT / "packaging" / "deb" / "build-deb.sh").read_text(encoding="utf-8")
        homebrew = (ROOT / "packaging" / "homebrew" / "nuv-agent.rb").read_text(encoding="utf-8")

        self.assertEqual(build_info.AGENT_VERSION, version)
        self.assertIn(f'VERSION="${{VERSION:-{version}}}"', deb)
        self.assertIn(f'version "{version}"', homebrew)


if __name__ == "__main__":
    unittest.main()
