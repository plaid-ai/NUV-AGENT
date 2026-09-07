#!/usr/bin/env bash
# This operator-only helper stages a reversible IQ9075 development runtime.
# It intentionally does NOT modify /opt/nuv-agent/current or publish a BOM.
set -euo pipefail
readonly demo_root=/opt/nuvion-demo/20260910-visualad
readonly probe_root=/home/plaid/nuvion-demo/20260910-visualad-probe
readonly runtime_probe=/home/plaid/nuvion-demo/20260910-model-probe
readonly dropin=/etc/systemd/system/nuv-agent.service.d/90-iq9075-visualad-demo.conf
readonly htp_dropin=/etc/systemd/system/nuv-agent.service.d/91-iq9075-visualad-htp-demo.conf
readonly baseline_target=releases/26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1

[ "$(id -u)" = 0 ] || { echo 'Run this scoped helper with sudo.' >&2; exit 1; }
[ "$(hostname)" = iq9075 ]
[ "$(readlink /opt/nuv-agent/current)" = "$baseline_target" ]

case "${1:-}" in
  stage)
    [ ! -e "$demo_root" ] || { echo 'Demo directory already exists; refusing overwrite.' >&2; exit 1; }
    [ -f "$probe_root/demo-source.tar.gz" ]
    [ -f "$probe_root/demo-source.sha256" ]
    (cd "$probe_root" && sha256sum -c demo-source.sha256)
    [ "$(stat -c %s "$probe_root/weights/openai_vit_l14_336_open_clip_model.safetensors")" = 1711828444 ]
    actual_model_sha=$(sha256sum "$probe_root/weights/openai_vit_l14_336_open_clip_model.safetensors")
    [ "${actual_model_sha%% *}" = fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0 ]
    [ "$(stat -c %s "$probe_root/weights/visualad_train_on_visa_CLIP.pth")" = 126119090 ]
    actual_checkpoint_sha=$(sha256sum "$probe_root/weights/visualad_train_on_visa_CLIP.pth")
    [ "${actual_checkpoint_sha%% *}" = fed8ed5e0973e9adb53a2c91e066eb5ba3f9a42fbf80f115a26998629d1f0a5b ]
    [ "$(git -c safe.directory="$probe_root/source/VisualAD" -C "$probe_root/source/VisualAD" rev-parse HEAD)" = 97eb5f88a44f27c644ea7ed4ecac5e35ddef18bb ]
    install -d -m 0755 "$demo_root/src" "$demo_root/weights" "$demo_root/evidence" "$demo_root/external"
    tar -xzf "$probe_root/demo-source.tar.gz" -C "$demo_root/src" --no-same-owner
    cp -a "$runtime_probe/venv" "$demo_root/python"
    cp -a "$probe_root/source/VisualAD" "$demo_root/external/VisualAD"
    install -m 0644 "$probe_root/weights/openai_vit_l14_336_open_clip_model.safetensors" "$demo_root/weights/open_clip_model.safetensors"
    install -m 0644 "$probe_root/weights/visualad_train_on_visa_CLIP.pth" "$demo_root/weights/visualad_train_on_visa_CLIP.pth"
    install -m 0644 "$runtime_probe/requirements.freeze.txt" "$demo_root/evidence/requirements.freeze.txt"
    install -m 0644 "$probe_root/demo-source.sha256" "$demo_root/evidence/demo-source.sha256"
    chown -R root:root "$demo_root"
    chmod -R go-w "$demo_root"
    chmod 0755 "$demo_root/src/packaging/dev/run-iq9075-visualad-demo.sh"
    /usr/sbin/runuser -u nuvion -- "$demo_root/src/packaging/dev/run-iq9075-visualad-demo.sh" --import-check
    echo 'STAGED; existing Agent is unchanged.'
    ;;
  activate)
    [ -f "$demo_root/src/packaging/dev/90-iq9075-visualad-demo.conf" ]
    [ ! -e "$dropin" ] || { echo 'Drop-in exists; refusing overwrite.' >&2; exit 1; }
    /usr/sbin/runuser -u nuvion -- "$demo_root/src/packaging/dev/run-iq9075-visualad-demo.sh" --import-check
    install -d -o nuvion -g nuvion -m 0700 /var/lib/nuv-agent/visualad-demo-settings
    install -d -m 0755 /etc/systemd/system/nuv-agent.service.d
    install -m 0644 "$demo_root/src/packaging/dev/90-iq9075-visualad-demo.conf" "$dropin"
    systemctl daemon-reload
    systemctl restart nuv-agent.service
    systemctl is-active nuv-agent.service
    echo 'DEVELOPMENT runtime active; signed current pointer preserved.'
    ;;
  rollback)
    if [ -e "$htp_dropin" ] || [ -L "$htp_dropin" ]; then
      echo 'HTP override is present; refusing CPU-only rollback without changing any files.' >&2
      echo 'Follow the dedicated 91-only HTP rollback in docs/IQ9075_VISUALAD_HTP_OPERATIONS.md first.' >&2
      exit 1
    fi
    [ -f "$dropin" ] && [ ! -L "$dropin" ]
    [ ! -e "$demo_root/evidence/rolled-back-demo.conf" ]
    mv "$dropin" "$demo_root/evidence/rolled-back-demo.conf"
    systemctl daemon-reload
    systemctl restart nuv-agent.service
    systemctl is-active nuv-agent.service
    echo 'Original signed runtime restored; all demo files retained.'
    ;;
  *) echo 'usage: install-iq9075-visualad-demo.sh stage|activate|rollback' >&2; exit 2 ;;
esac
