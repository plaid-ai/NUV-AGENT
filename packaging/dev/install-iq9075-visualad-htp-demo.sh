#!/usr/bin/env bash
# Exact-board, reversible private experimental deployment. Not Fleet OTA.
set -euo pipefail
readonly demo_root=/opt/nuvion-demo/20260910-visualad-htp
readonly probe_root=/home/plaid/nuvion-demo/20260910-visualad-htp-probe
readonly dropin=/etc/systemd/system/nuv-agent.service.d/91-iq9075-visualad-htp-demo.conf
readonly source_archive="$probe_root/experimental-source-001.tar.gz"
readonly manifest_pin=89114e2159e4e6080971d5fcbe4782829daad78406f043eb877d603885fc4795
readonly graph_pin=1ad0cd1a216ba3bf89267e18bd7f7260bae48290067e3ab60d8cd8392fdeacb8
[ "$(id -u)" = 0 ]
[ "$(hostname)" = iq9075 ]
[ "$(readlink /opt/nuv-agent/current)" = releases/26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1 ]

import_check() {
  /usr/sbin/runuser -u nuvion -- env NUVION_VISUALAD_HTP_MANIFEST_SHA256="$manifest_pin" \
    "$demo_root/src/packaging/dev/run-iq9075-visualad-htp-demo.sh" --import-check
}

case "${1:-}" in
  stage)
    [[ "${2:-}" =~ ^[0-9a-f]{64}$ ]]
    [ ! -e "$demo_root" ] && [ ! -L "$demo_root" ]
    [ -f "$source_archive" ] && [ ! -L "$source_archive" ]
    source_actual=$(sha256sum "$source_archive")
    [ "${source_actual%% *}" = "$2" ]
    install -d -m 0755 "$demo_root/src" "$demo_root/model" "$demo_root/evidence"
    tar -xzf "$source_archive" -C "$demo_root/src" --no-same-owner
    cp -a "$probe_root/venv" "$demo_root/python"
    install -m 0644 "$probe_root/exports/export-001-89114e21/manifest.json" "$demo_root/model/manifest.json"
    install -m 0644 "$probe_root/exports/export-001-89114e21/visualad.onnx" "$demo_root/model/visualad.onnx"
    manifest_actual=$(sha256sum "$demo_root/model/manifest.json")
    graph_actual=$(sha256sum "$demo_root/model/visualad.onnx")
    [ "${manifest_actual%% *}" = "$manifest_pin" ]
    [ "${graph_actual%% *}" = "$graph_pin" ]
    install -m 0600 "$demo_root/src/packaging/dev/iq9075-visualad-htp-deployment.env" "$demo_root/deployment.env"
    install -m 0644 "$probe_root/htp-install-report.json" "$demo_root/evidence/htp-install-report.json"
    install -m 0644 "$probe_root/opencv-install-report.json" "$demo_root/evidence/opencv-install-report.json"
    install -m 0644 "$probe_root/postprocess-install-report.json" "$demo_root/evidence/postprocess-install-report.json"
    sha256sum "$source_archive" > "$demo_root/evidence/source-archive.sha256"
    "$probe_root/venv/bin/python" -m pip freeze > "$demo_root/evidence/requirements.freeze.txt"
    /usr/bin/python3 --version > "$demo_root/evidence/python-version.txt"
    dpkg-query -W python3 libglib2.0-0t64 gstreamer1.0-tools > "$demo_root/evidence/os-packages.txt"
    readlink /opt/nuv-agent/current > "$demo_root/evidence/signed-current-before.txt"
    sha256sum /etc/systemd/system/nuv-agent.service.d/90-iq9075-visualad-demo.conf > "$demo_root/evidence/previous-dropin.sha256"
    chown -R root:root "$demo_root"
    chmod -R go-w "$demo_root"
    chmod 0755 "$demo_root/src/packaging/dev/run-iq9075-visualad-htp-demo.sh"
    install -d -o nuvion -g nuvion -m 0700 /var/lib/nuv-agent/visualad-htp /var/lib/nuv-agent/visualad-htp-settings
    find "$demo_root/src" "$demo_root/python" "$demo_root/model" -type f -exec sha256sum {} + > "$demo_root/evidence/staged-files.sha256"
    import_check
    echo 'EXPERIMENTAL_STAGED; parity remains FAILED; existing Agent unchanged.'
    ;;
  activate)
    [ ! -e "$dropin" ] && [ ! -L "$dropin" ]
    [ -f /etc/systemd/system/nuv-agent.service.d/90-iq9075-visualad-demo.conf ]
    sha256sum -c "$demo_root/evidence/previous-dropin.sha256"
    sha256sum -c "$demo_root/evidence/staged-files.sha256" --status
    import_check
    install -m 0644 "$demo_root/src/packaging/dev/91-iq9075-visualad-htp-demo.conf" "$dropin"
    systemctl daemon-reload
    systemctl restart nuv-agent.service
    systemctl is-active nuv-agent.service
    echo 'EXPERIMENTAL_PROCESS_STARTED; verify actual HTP inference separately.'
    ;;
  rollback)
    [ -f "$dropin" ] && [ ! -L "$dropin" ]
    rollback_dir=$(mktemp -d "$demo_root/evidence/rollback.XXXXXXXX")
    mv "$dropin" "$rollback_dir/91-iq9075-visualad-htp-demo.conf"
    systemctl daemon-reload
    systemctl reset-failed nuv-agent.service
    systemctl restart nuv-agent.service
    systemctl is-active nuv-agent.service
    echo 'Restored previous 90 video-only dev runtime; evidence retained, not signed OTA rollback.'
    ;;
  *) echo 'usage: install-iq9075-visualad-htp-demo.sh stage SOURCE_SHA256|activate|rollback' >&2; exit 2 ;;
esac
