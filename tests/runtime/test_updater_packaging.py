from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class UpdaterPackagingTest(unittest.TestCase):
    def test_agent_has_no_root_equivalent_docker_group_or_in_place_venv_clear(self) -> None:
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        unit = (ROOT / "packaging/systemd/nuv-agent.service").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("usermod -a -G docker nuvion", postinst)
        self.assertNotIn("venv --clear", postinst)
        self.assertIn("gpasswd -d nuvion docker", postinst)
        self.assertNotIn("gpasswd -d nuvion docker >/dev/null 2>&1 || true", postinst)
        self.assertIn("nuvion still has the root-equivalent docker group", postinst)
        self.assertIn(
            "systemctl stop nuv-agent-updater.socket nuv-agent-updater.service",
            postinst,
        )
        self.assertIn("systemctl is-active --quiet nuv-agent.service", postinst)
        self.assertNotIn("systemctl stop nuv-agent.service || true", postinst)
        self.assertIn("systemctl restart nuv-agent.service", postinst)
        self.assertNotIn("systemctl start nuv-agent.service", postinst)
        self.assertIn('BOOTSTRAP_ROOT="$INSTALL_ROOT/bootstrap"', postinst)
        self.assertIn("atomic_slot_link current", postinst)
        self.assertNotIn("docker.service", unit)
        self.assertNotIn("PermissionsStartOnly=true", unit)
        self.assertIn("ExecStart=/usr/bin/nuv-agent run", unit)

    def test_boot_guard_and_restart_policy_are_slot_aware(self) -> None:
        unit = (ROOT / "packaging/systemd/nuv-agent.service").read_text(
            encoding="utf-8"
        )
        bundle = (ROOT / "packaging/release/build-agent-bundle.sh").read_text(
            encoding="utf-8"
        )
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")

        self.assertNotIn("ExecStartPre=/opt/nuv-agent/venv/bin/python", unit)
        self.assertEqual(
            unit.count("ExecStartPre=/opt/nuv-agent/current/venv/bin/python"),
            2,
        )
        self.assertIn("nuvion_app.runtime.settings_boot_guard", unit)
        self.assertIn("nuvion_app.runtime.bootstrap", unit)
        self.assertIn("StartLimitIntervalSec=300", unit)
        self.assertIn("StartLimitBurst=3", unit)
        self.assertIn("Restart=always", unit)
        self.assertIn(
            'chmod 0755 "$slot_root/bin/nuv-agent" "$slot_root/venv/bin/python"',
            bundle,
        )
        self.assertIn('readonly BOOTSTRAP_BUNDLE="$INSTALL_ROOT/share/', postinst)
        self.assertIn("sha256sum --check --strict --status", postinst)
        self.assertNotIn("pip install", postinst)
        self.assertNotIn("python3 -m venv", postinst)

    def test_updater_is_root_owned_socket_activated_and_outside_agent_slot(self) -> None:
        service = (ROOT / "packaging/systemd/nuv-agent-updater.service").read_text(
            encoding="utf-8"
        )
        socket_unit = (
            ROOT / "packaging/systemd/nuv-agent-updater.socket"
        ).read_text(encoding="utf-8")
        build = (ROOT / "packaging/deb/build-deb.sh").read_text(encoding="utf-8")
        protocol = (ROOT / "nuvion_updater/protocol.py").read_text(encoding="utf-8")

        self.assertIn("User=root", service)
        self.assertIn("ProtectSystem=strict", service)
        self.assertIn("ReadWritePaths=/opt/nuv-agent /var/lib/nuvion-updater", service)
        self.assertIn("ListenStream=/run/nuvion-updater/control.sock", socket_unit)
        self.assertIn("SocketGroup=nuvion", socket_unit)
        self.assertIn("SocketMode=0660", socket_unit)
        self.assertIn("SO_PEERCRED", protocol)
        self.assertIn("$PKG_DIR/usr/lib/nuvion-updater", build)
        self.assertIn("python3-cryptography", build)
        self.assertIn("test-iq9075.sh", build)
        self.assertIn("PrivateDevices=false", service)
        self.assertNotIn("PrivateDevices=true", service)
        self.assertIn("dpkg-deb --root-owner-group --build", build)
        self.assertIn("bootstrap-agent-bundle.tar.gz", build)
        self.assertIn("requirements-agent-bundle-arm64.txt", build)

    def test_dpkg_root_owner_group_normalizes_unprivileged_builder_ids(self) -> None:
        dpkg_deb = shutil.which("dpkg-deb")
        if dpkg_deb is None:
            self.skipTest("dpkg-deb is unavailable on this host")
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            package = root / "package"
            (package / "DEBIAN").mkdir(parents=True)
            (package / "usr/lib/nuvion-updater").mkdir(parents=True)
            (package / "DEBIAN/control").write_text(
                "Package: ownership-test\n"
                "Version: 1.0.0\n"
                "Architecture: all\n"
                "Maintainer: test <test@example.invalid>\n"
                "Description: ownership normalization test\n",
                encoding="utf-8",
            )
            (package / "usr/lib/nuvion-updater/helper.py").write_text(
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            output = root / "ownership-test.deb"
            subprocess.run(
                [dpkg_deb, "--root-owner-group", "--build", str(package), str(output)],
                check=True,
                capture_output=True,
            )
            filesystem_tar = subprocess.run(
                [dpkg_deb, "--fsys-tarfile", str(output)],
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(filesystem_tar), mode="r:*") as archive:
                privileged = archive.getmember("./usr/lib/nuvion-updater/helper.py")
            self.assertEqual((privileged.uid, privileged.gid), (0, 0))

    def test_agent_state_databases_remain_outside_release_slots(self) -> None:
        unit = (ROOT / "packaging/systemd/nuv-agent.service").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "NUVION_EVENT_OUTBOX_PATH=/var/lib/nuv-agent/events.sqlite3", unit
        )
        self.assertIn(
            "NUVION_COMMAND_INBOX_PATH=/var/lib/nuv-agent/commands.sqlite3", unit
        )
        self.assertNotIn("/opt/nuv-agent/current/var", unit)

        wrapper = (ROOT / "packaging/systemd/nuv-agent-current").read_text(
            encoding="utf-8"
        )
        self.assertIn('export NUVION_ACTIVE_SLOT="$active_slot"', wrapper)
        self.assertIn("NUVION_RELEASE_BOM_PATH", wrapper)
        self.assertIn("NUVION_EXPECTED_BOM_DIGEST", wrapper)

    def test_publish_pairs_exact_artifact_signature_and_bom_in_digest_directory(self) -> None:
        publish = (ROOT / "packaging/apt/publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('CONTENT_BOM_DIR="$PUBLIC_DIR/releases/by-bom-sha256/$BOM_DIGEST"', publish)
        self.assertIn('$release_dir/release-bom.json"', publish)
        self.assertIn('$release_dir/release-bom.json.sig"', publish)
        self.assertIn('$(basename "$BOM_ARTIFACT_PATH")', publish)
        self.assertIn("Refusing to overwrite immutable release bytes", publish)
        self.assertIn("RELEASE_KEYRING_PATH is required", publish)
        self.assertIn('cp -n "$published_release" "$remote_path"', publish)
        self.assertIn("-x '^(releases/|pool/)'", publish)
        self.assertNotIn("setmeta", publish)
        self.assertIn('find "$PUBLIC_DIR/pool" -type f', publish)

    def test_packaging_shell_files_parse(self) -> None:
        paths = sorted((ROOT / "packaging").rglob("*.sh")) + [
            ROOT / "packaging/deb/postinst",
            ROOT / "packaging/deb/prerm",
            ROOT / "packaging/systemd/nuv-agent-current",
            ROOT / "packaging/systemd/nuv-agent-slot-entrypoint",
        ]
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                subprocess.run(
                    ["bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
