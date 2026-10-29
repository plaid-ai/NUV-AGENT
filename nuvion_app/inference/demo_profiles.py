"""Local, allowlisted demo selection. Media never comes from request paths."""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path

class DemoProfiles:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.state_path = self.root / 'selected.json'
        self.lock = threading.Lock()
        self.profiles = {'mvtec': {'id': 'mvtec', 'name': 'MVTec 불량 검사', 'kind': 'dataset'}}
        manifest = self.root / 'profiles.json'
        if manifest.is_file():
            for item in json.loads(manifest.read_text()):
                key = item['id']
                if key in self.profiles or not key.replace('-', '').isalnum():
                    raise ValueError('Invalid demo profile ID')
                path = (self.root / item['file']).resolve()
                if not path.is_relative_to(self.root) or path.suffix.lower() != '.mp4':
                    raise ValueError('Invalid demo media path')
                self.profiles[key] = {'id': key, 'name': item['name'], 'kind': 'video', 'path': path}
        self.active_id = 'mvtec'
        try:
            selected = json.loads(self.state_path.read_text())['id']
            self.validate(selected)
            self.active_id = selected
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.pending_id = None

    def validate(self, key):
        if not isinstance(key, str) or key not in self.profiles:
            raise ValueError('Unknown demo profile')
        p = self.profiles[key]
        if p['kind'] == 'video' and not p['path'].is_file():
            raise ValueError('Demo video not installed')
        return p

    def snapshot(self):
        return {'activeId': self.active_id, 'pendingId': self.pending_id,
                'profiles': [{k: p[k] for k in ('id', 'name', 'kind')} | {'available': p['kind'] == 'dataset' or p['path'].is_file()} for p in self.profiles.values()]}

    def select(self, key, restart):
        with self.lock:
            self.validate(key)
            if self.pending_id is not None:
                raise RuntimeError('Demo switch already pending')
            if key == self.active_id:
                return self.snapshot()
            self.root.mkdir(parents=True, exist_ok=True)
            old = self.state_path.read_bytes() if self.state_path.exists() else None
            tmp = self.state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'id': key}))
            os.replace(tmp, self.state_path)
            self.pending_id = key
            try:
                if not restart():
                    raise RuntimeError('Demo restart unavailable')
            except Exception:
                self.pending_id = None
                if old is None:
                    self.state_path.unlink(missing_ok=True)
                else:
                    tmp.write_bytes(old)
                    os.replace(tmp, self.state_path)
                raise
            return self.snapshot()
