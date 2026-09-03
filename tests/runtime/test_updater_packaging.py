from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class UpdaterPackagingTest(unittest.TestCase):
    def test_postrm_rejects_unsafe_marker_before_restoring_legacy_tool(self) -> None:
        postrm = (ROOT / "packaging/deb/postrm").read_text(encoding="utf-8")
        program = postrm.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        blocker = (
            b"#!/usr/bin/python3\n"
            b"import sys\n"
            b"sys.stderr.write('Fleet E2E is blocked during package maintenance\\n')\n"
            b"raise SystemExit(1)\n"
        )
        legacy = b"#!/usr/bin/python3\nraise SystemExit(0)\n"
        real_lstat = os.lstat

        def root_owned_lstat(path: os.PathLike[str] | str) -> os.stat_result:
            fields = list(real_lstat(path))
            fields[4] = 0
            fields[5] = 0
            return os.stat_result(fields)

        for marker_kind, expected_error in (
            ("symlink", "package rollback endpoint is unsafe"),
            ("corrupt", "package maintenance rollback marker is invalid"),
        ):
            with self.subTest(marker=marker_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tool = root / "iq9075-board-e2e.py"
                backup = root / "legacy-board-tool.preinst"
                marker = root / "package-maintenance.json"
                tool.write_bytes(blocker)
                tool.chmod(0o755)
                backup.write_bytes(legacy)
                backup.chmod(0o700)
                if marker_kind == "symlink":
                    target = root / "marker-target"
                    target.write_bytes(b"corrupt\n")
                    marker.symlink_to(target)
                else:
                    marker.write_bytes(b"{}\n")
                    marker.chmod(0o600)
                argv = ["postrm-rollback", str(tool), str(backup), str(marker)]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(os, "lstat", side_effect=root_owned_lstat),
                    self.assertRaisesRegex(SystemExit, expected_error),
                ):
                    exec(compile(program, "postrm-rollback", "exec"), {})
                self.assertEqual(tool.read_bytes(), blocker)
                self.assertEqual(backup.read_bytes(), legacy)

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
        self.assertIn("MemoryHigh=50%", unit)
        self.assertIn("MemoryMax=60%", unit)
        self.assertIn("MemorySwapMax=0", unit)
        self.assertIn("LimitCORE=0", unit)
        self.assertIn("OOMPolicy=stop", unit)
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
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        tmpfiles = (
            ROOT / "packaging/tmpfiles/nuvion-updater.conf"
        ).read_text(encoding="utf-8")

        self.assertIn("User=root", service)
        self.assertIn("LimitCORE=0", service)
        self.assertIn("ProtectSystem=strict", service)
        self.assertIn("ReadWritePaths=/opt/nuv-agent /var/lib/nuvion-updater", service)
        self.assertIn("ListenStream=/run/nuvion-updater/control.sock", socket_unit)
        self.assertIn("SocketGroup=nuvion", socket_unit)
        self.assertIn("SocketMode=0660", socket_unit)
        self.assertIn("SO_PEERCRED", protocol)
        self.assertIn("$PKG_DIR/usr/lib/nuvion-updater", build)
        self.assertIn("python3-cryptography", build)
        self.assertIn("test-iq9075.sh", build)
        self.assertIn("probe-iq9075-oak.sh", build)
        self.assertIn("PrivateDevices=false", service)
        self.assertNotIn("PrivateDevices=true", service)
        self.assertIn("dpkg-deb --root-owner-group --build", build)
        self.assertIn("bootstrap-agent-bundle.tar.gz", build)
        self.assertIn("requirements-agent-bundle-arm64.txt", build)
        self.assertIn("health-attestation-keyring.json", postinst)
        self.assertEqual(tmpfiles, "d /run/nuvion-updater 0750 root nuvion -\n")
        self.assertIn("$PKG_DIR/usr/lib/tmpfiles.d", build)
        self.assertIn("packaging/tmpfiles/nuvion-updater.conf", build)
        self.assertIn(
            "/usr/bin/systemd-tmpfiles --create "
            "/usr/lib/tmpfiles.d/nuvion-updater.conf",
            postinst,
        )
        self.assertIn(
            "Unsafe updater runtime directory identity", postinst
        )
        self.assertLess(
            postinst.index("systemd-tmpfiles --create"),
            postinst.index("systemctl enable --now nuv-agent-updater.socket"),
        )
        self.assertIn("After=systemd-tmpfiles-setup.service", socket_unit)
        self.assertIn("Before=nuv-agent.service", socket_unit)
        self.assertLess(
            postinst.index("systemctl enable --now nuv-agent-updater.socket"),
            postinst.index("systemctl restart nuv-agent.service"),
        )

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

    def test_fleet_boot_reconciler_gates_all_runtime_endpoints(self) -> None:
        reconcile = (
            ROOT / "packaging/systemd/nuvion-fleet-e2e-reconcile.service"
        ).read_text(encoding="utf-8")
        protected = [
            ROOT / "packaging/systemd/nuv-agent.service",
            ROOT / "packaging/systemd/nuv-agent-updater.service",
            ROOT / "packaging/systemd/nuv-agent-updater.socket",
        ]
        build = (ROOT / "packaging/deb/build-deb.sh").read_text(encoding="utf-8")
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        prerm = (ROOT / "packaging/deb/prerm").read_text(encoding="utf-8")
        preinst = (ROOT / "packaging/deb/preinst").read_text(encoding="utf-8")
        postrm = (ROOT / "packaging/deb/postrm").read_text(encoding="utf-8")

        self.assertIn("Type=oneshot", reconcile)
        self.assertIn("LimitCORE=0", reconcile)
        self.assertIn("ProtectSystem=strict", reconcile)
        self.assertIn("NoNewPrivileges=yes", reconcile)
        self.assertIn("RestrictAddressFamilies=AF_UNIX", reconcile)
        self.assertIn(
            "ExecStart=/usr/bin/python3 -I /usr/local/libexec/nuvion/iq9075-board-e2e.py boot-reconcile",
            reconcile,
        )
        self.assertNotIn("/bin/sh", reconcile)
        for path in protected:
            unit = path.read_text(encoding="utf-8")
            self.assertIn("Requires=nuvion-fleet-e2e-reconcile.service", unit)
            self.assertIn("After=nuvion-fleet-e2e-reconcile.service", unit)
        self.assertIn("nuvion-fleet-e2e-reconcile.service", build)
        self.assertIn("/var/lib/nuvion-fleet-e2e/runs", postinst)
        self.assertLess(
            postinst.index("systemctl restart nuvion-fleet-e2e-reconcile.service"),
            postinst.index("systemctl enable --now nuv-agent-updater.socket"),
        )
        self.assertLess(
            postinst.index("systemctl restart nuvion-fleet-e2e-reconcile.service"),
            postinst.index("systemctl restart nuv-agent.service"),
        )
        self.assertIn("iq9075-board-e2e.py", prerm)
        self.assertIn("boot-reconcile --package-maintenance", prerm)
        self.assertIn("test ! -e /var/lib/nuvion-fleet-e2e/active-run.json", prerm)
        self.assertIn("packaging/deb/preinst", build)
        self.assertIn("resource.setrlimit(resource.RLIMIT_CORE, (0, 0))", preinst)
        self.assertIn('if [ -e "$ACTIVE_RUN" ] || [ -L "$ACTIVE_RUN" ]', preinst)
        self.assertIn(
            "Active Fleet E2E recovery must be cleaned before package upgrade",
            preinst,
        )
        self.assertIn("legacy-board-tool.preinst", preinst)
        self.assertIn("Fleet E2E is blocked during package maintenance", preinst)
        self.assertNotIn('"boot-reconcile",', preinst)
        self.assertNotIn('"cleanup",', preinst)
        self.assertLess(preinst.index("flock --exclusive"), preinst.index("os.replace"))
        self.assertLess(
            preinst.index("os.replace(backup_temporary, backup)"),
            preinst.index("os.replace(blocker_temporary, tool)"),
        )
        self.assertLess(
            preinst.index("os.replace(blocker_temporary, tool)"),
            preinst.index("os.replace(temporary, path)"),
        )
        self.assertLess(preinst.index("systemctl stop"), preinst.index("os.replace"))
        for script in (preinst, postinst, prerm, postrm):
            self.assertIn("ulimit -S -c 0", script)
            self.assertIn("ulimit -H -c 0", script)
        self.assertGreater(
            postinst.index("boot-reconcile --package-maintenance"),
            postinst.index("runuser --user nuvion"),
        )
        self.assertLess(
            postinst.index("os.unlink(marker)"),
            postinst.index("systemctl restart nuvion-fleet-e2e-reconcile.service"),
        )
        self.assertLess(
            postinst.index("systemctl restart nuvion-fleet-e2e-reconcile.service"),
            postinst.index("systemctl enable --now nuv-agent-updater.socket"),
        )
        self.assertLess(
            postinst.index("systemctl enable --now nuv-agent-updater.socket"),
            postinst.index('flock --unlock "$package_lock_fd"'),
        )
        self.assertIn("trap stop_partial_package_activation EXIT", postinst)
        self.assertIn("trap 'exit 143' TERM", postinst)
        self.assertIn("ensure_fleet_maintenance_marker", postinst)
        self.assertIn("legacy-board-tool.preinst", postinst)
        self.assertLess(
            postinst.index("systemctl stop"),
            postinst.index("ensure_fleet_maintenance_marker\nflock --unlock"),
        )
        self.assertIn("abort-install|abort-upgrade|disappear", postrm)
        self.assertIn("cannot prove restored Fleet recovery tool identity", postrm)
        self.assertIn("os.replace(restore_temporary, tool)", postrm)
        self.assertIn(
            "if backup_content is not None and tool_content in {None, blocker}",
            postrm,
        )
        self.assertIn("if tool_changed:", postrm)
        self.assertIn("systemctl daemon-reload || true", postrm)
        self.assertIn("partial_stop_failed=false", postinst)
        self.assertIn("--kill-who=all --signal=SIGKILL", postinst)
        self.assertIn("Protected runtime remained active", postinst)

    def test_publish_pairs_exact_artifact_signature_and_bom_in_digest_directory(self) -> None:
        publish = (ROOT / "packaging/apt/publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('CONTENT_BOM_DIR="$PUBLIC_DIR/releases/by-bom-sha256/$BOM_DIGEST"', publish)
        self.assertIn('content_bom="$CONTENT_BOM_DIR/release-bom.json"', publish)
        self.assertIn('content_signature="$CONTENT_BOM_DIR/release-bom.json.sig"', publish)
        self.assertIn('$(basename "$BOM_ARTIFACT_PATH")', publish)
        self.assertIn("Refusing to overwrite immutable release bytes", publish)
        self.assertIn("RELEASE_KEYRING_PATH is required", publish)
        self.assertIn("--if-generation-match=0", publish)
        self.assertIn('gcloud storage cat "$remote_path" | cmp -s', publish)
        self.assertNotIn("gsutil", publish)
        self.assertIn("APT_BY_HASH_PATHS", publish)
        self.assertIn('upload_mutable_metadata "$APT_DISCOVERY_PATH"', publish)
        self.assertNotIn("setmeta", publish)
        self.assertIn('find "$PUBLIC_DIR/pool" -type f', publish)

    def test_packaging_shell_files_parse(self) -> None:
        paths = sorted((ROOT / "packaging").rglob("*.sh")) + [
            ROOT / "packaging/deb/postinst",
            ROOT / "packaging/deb/preinst",
            ROOT / "packaging/deb/prerm",
            ROOT / "packaging/deb/postrm",
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
