#!/usr/bin/env python3
"""Collect Pro prototype demo evidence on-device. Switches profiles and restarts once.

Run as root after installing the candidate DEB. No motor/camera commands are sent.
Evidence qualifies this recorded QNN/overlay stack, not bare-package inference or OTA.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

BASE = 'http://localhost:8080'

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=4) as response:
        return response.status, response.headers, response.read()

def pid():
    return int(command('systemctl', 'show', 'nuv-agent', '-p', 'MainPID', '--value'))

def wait_profile(profile):
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        try:
            if json.loads(get('/api/profiles')[2])['activeId'] == profile and get('/api/frame.jpg')[0] == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError('profile failed to become ready: ' + profile)

def select(profile):
    req = urllib.request.Request(BASE + '/api/profile', json.dumps({'id': profile}).encode(),
        headers={'Origin': BASE, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=5) as response:
        response.read()
    wait_profile(profile)

def logs(process, since):
    return command('journalctl', '-u', 'nuv-agent', '_PID=' + str(process), '--since', '@' + str(since), '-o', 'cat', '--no-pager')

def inference(log):
    return [float(v) for v in re.findall(r'provider=QNN/HTP[^\n]*?inferenceSeconds=([0-9.]+)', log)]

def observe(profile):
    select(profile)
    process = pid()
    start = time.time()
    frames = []
    for _ in range(20):
        status, headers, image = get('/api/frame.jpg')
        frames.append({'httpStatus': status, 'sequence': int(headers['X-Frame-Sequence']),
                       'sha256': hashlib.sha256(image).hexdigest()})
        time.sleep(2)
    text = logs(process, start)
    return {'profile': profile, 'durationSeconds': time.time() - start, 'frames': frames,
            'provider': 'QNN/HTP' if 'provider=QNN/HTP' in text else 'unknown',
            'inferenceSeconds': inference(text), 'loops': text.count('Local video rewound; live stream clock preserved'),
            'processStable': process > 0 and pid() == process,
            'runtimeErrors': [line for line in text.splitlines() if re.search(r'Traceback|segmentation fault|DeviceReset|GStreamer.*ERROR', line, re.I)]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deb', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    version = command('dpkg-query', '-W', '-f=${Version}', 'nuv-agent')
    slot = Path('/opt/nuv-agent/current').resolve()
    build = slot / 'venv/lib/python3.12/site-packages/nuvion_app/build_info.py'
    source = re.search(r'COMPONENT_SHA\s*=\s*[\"\x27]([a-f0-9]{40})', build.read_text()).group(1)
    runtime = {}
    # Only code/static inputs, never agent.env or credentials.
    for root in (slot / 'venv/lib/python3.12/site-packages/nuvion_app',
                 Path('/opt/nuv-agent/overlays/local-demo-profiles-20261029'),
                 Path('/opt/nuv-agent/share/local-demo')):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.html', '.js', '.css'}:
                runtime[str(path)] = digest(path)
    observations = []
    try:
        for profile in ('mvtec', 'video-1', 'video-2'):
            observations.append(observe(profile))
            print('qualified observation:', profile, flush=True)
    finally:
        select('video-1')
    before = pid()
    start = time.time()
    command('systemctl', 'restart', 'nuv-agent')
    wait_profile('video-1')
    time.sleep(5)
    after = pid()
    result = {'schemaVersion': 1, 'target': 'pro-demo', 'version': version, 'componentSha': source,
              'qualifiedCapabilities': ['package-install', 'local-demo', 'qnn-htp-inference', 'video-loop', 'service-restart'],
              'excludedCapabilities': ['iq9075-ota', 'camera-csi', 'openrb-motor', 'production-accuracy'],
              'device': {'profile': 'ventuno-q-prototype', 'architecture': command('uname', '-m'),
                         'kernel': command('uname', '-r')},
              'package': {'name': args.deb.name, 'sha256': digest(args.deb), 'size': args.deb.stat().st_size,
                          'installedVersion': version, 'installedComponentSha': source},
              'bundleSha256': args.bundle_sha256, 'runnerSha256': digest(__file__),
              'runtimeFiles': runtime, 'observations': observations,
              'restart': {'pidChanged': before != after and after > 0, 'httpStatus': get('/api/frame.jpg')[0],
                          'selectedProfile': json.loads(get('/api/profiles')[2])['activeId'],
                          'inferenceResumed': bool(inference(logs(after, start)))},
              'recordedAt': dt.datetime.now(dt.timezone.utc).isoformat()}
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    print('Evidence saved:', args.output)

if __name__ == '__main__':
    main()
