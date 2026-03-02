from __future__ import annotations

import html
import json
import os
import platform
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import dotenv_values, load_dotenv


DEFAULT_PORT = 8088
SECRET_KEY_MARKERS = ("PASSWORD",)
DEVICE_TYPE = "NUV_AGENT"
PAIRING_POLL_INTERVAL_SEC = int(os.getenv("NUVION_PAIRING_POLL_INTERVAL_SEC", "5"))
PAIRING_TIMEOUT_SEC = int(os.getenv("NUVION_PAIRING_TIMEOUT_SEC", "600"))
PROVISION_ENDPOINT = os.getenv("NUVION_DEVICE_PROVISION_ENDPOINT", "/devices/provision")
PAIRING_INIT_ENDPOINT = os.getenv("NUVION_PAIRING_INIT_ENDPOINT", "/devices/pairings/init")
PAIRING_STATUS_ENDPOINT = os.getenv("NUVION_PAIRING_STATUS_ENDPOINT", "/devices/pairings/{pairing_id}")
REQUIRED_KEYS = {
    "NUVION_SERVER_BASE_URL",
    "NUVION_DEVICE_USERNAME",
    "NUVION_DEVICE_PASSWORD",
    "NUVION_RTP_REMOTE_IP",
}
PLACEHOLDER_VALUES = {"***"}

_LOADED = False


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if stripped in PLACEHOLDER_VALUES:
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return False


def _is_secret_key(key: str) -> bool:
    return any(marker in key for marker in SECRET_KEY_MARKERS)


def template_path() -> Path:
    path = Path(__file__).resolve().parent / "config_template.env"
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[1] / ".env.example"
    return fallback


def load_template() -> Tuple[List[str], List[Dict[str, str]]]:
    lines = template_path().read_text().splitlines()
    fields: List[Dict[str, str]] = []
    pending_comment: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pending_comment = None
            continue
        if stripped.startswith("#"):
            pending_comment = stripped.lstrip("#").strip()
            continue
        if "=" not in line:
            pending_comment = None
            continue
        key, default = line.split("=", 1)
        fields.append(
            {
                "key": key.strip(),
                "default": default.strip(),
                "comment": pending_comment or "",
            }
        )
        pending_comment = None
    return lines, fields


def _find_repo_env(start: Path) -> Optional[Path]:
    for parent in [start, *start.parents]:
        if (parent / ".env").exists() and (parent / "nuvion_app").is_dir():
            return parent / ".env"
    return None


def _default_system_paths() -> List[Path]:
    if sys.platform == "darwin":
        paths: List[Path] = []
        opt_homebrew = Path("/opt/homebrew/etc/nuv-agent/agent.env")
        usr_local = Path("/usr/local/etc/nuv-agent/agent.env")
        if opt_homebrew.exists() or Path("/opt/homebrew").exists():
            paths.append(opt_homebrew)
            paths.append(usr_local)
        else:
            paths.append(usr_local)
            paths.append(opt_homebrew)
        paths.append(Path("/etc/nuv-agent/agent.env"))
        return paths
    return [Path("/etc/nuv-agent/agent.env")]


def resolve_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.getenv("NUV_AGENT_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    repo_env = _find_repo_env(Path.cwd())
    if repo_env:
        return repo_env

    candidates = _default_system_paths()
    for path in candidates:
        try:
            if path.exists():
                return path
        except PermissionError:
            continue
    return candidates[0]


def load_env(path: Optional[str] = None) -> Path:
    global _LOADED
    if _LOADED:
        return resolve_config_path(path)
    config_path = resolve_config_path(path)
    os.environ.setdefault("NUV_AGENT_CONFIG", str(config_path))
    load_dotenv(config_path, override=False)
    _LOADED = True
    return config_path


def read_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {key: value for key, value in values.items() if value is not None}


def render_env(lines: List[str], values: Dict[str, str]) -> str:
    rendered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        value = values.get(key, "")
        rendered.append(f"{key}={value}")
    return "\n".join(rendered) + "\n"


def write_env(path: Path, lines: List[str], values: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_env(lines, values)
    path.write_text(content)


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _request_json(
    url: str,
    method: str = "POST",
    payload: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
) -> Dict[str, object] | None:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        return {"error": f"{exc.code} {exc.reason}", "details": detail}
    except Exception as exc:
        return {"error": str(exc)}


def _extract_data(response: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not response:
        return None
    if "data" in response and isinstance(response["data"], dict):
        return response["data"]  # type: ignore[return-value]
    return response


def _extract_list(response: Optional[Dict[str, object]] | List[object]) -> Optional[List[object]]:
    if not response:
        return None
    if isinstance(response, list):
        return response
    data = response.get("data")
    if isinstance(data, list):
        return data
    return None


def _login_user(server_base_url: str, username: str, password: str) -> Optional[str]:
    url = f"{_normalize_base_url(server_base_url)}/auth/login"
    payload = {"username": username, "password": password}
    response = _request_json(url, payload=payload)
    data = _extract_data(response)
    if not data:
        return None
    token = data.get("accessToken") or data.get("token")
    return token if isinstance(token, str) else None


def _provision_device(
    server_base_url: str,
    username: str,
    password: str,
    space_id: str,
    device_name: str,
) -> Optional[Dict[str, object]]:
    token = _login_user(server_base_url, username, password)
    if not token:
        return None
    payload: Dict[str, object] = {
        "spaceId": space_id,
        "deviceName": device_name,
        "deviceType": DEVICE_TYPE,
        "model": platform.machine(),
        "os": platform.system(),
    }
    url = f"{_normalize_base_url(server_base_url)}{PROVISION_ENDPOINT}"
    response = _request_json(url, payload=payload, headers={"Authorization": f"Bearer {token}"})
    return _extract_data(response)


def _fetch_spaces(
    server_base_url: str,
    username: str,
    password: str,
) -> Optional[List[object]]:
    token = _login_user(server_base_url, username, password)
    if not token:
        return None
    url = f"{_normalize_base_url(server_base_url)}/spaces/me"
    response = _request_json(url, method="GET", headers={"Authorization": f"Bearer {token}"})
    return _extract_list(response)


def _init_pairing(server_base_url: str, device_name: str) -> Optional[Dict[str, object]]:
    payload: Dict[str, object] = {
        "deviceName": device_name,
        "deviceType": DEVICE_TYPE,
        "model": platform.machine(),
        "os": platform.system(),
    }
    base_url = _normalize_base_url(server_base_url)
    url = f"{base_url}{PAIRING_INIT_ENDPOINT}"
    response = _request_json(url, payload=payload)
    return _extract_data(response)


def _wait_for_pairing(
    server_base_url: str,
    pairing_id: str,
    pairing_secret: Optional[str],
) -> Optional[Dict[str, object]]:
    base_url = _normalize_base_url(server_base_url)
    deadline = time.time() + PAIRING_TIMEOUT_SEC
    headers = {}
    if pairing_secret:
        headers["X-Pairing-Secret"] = pairing_secret
    status_url = f"{base_url}{PAIRING_STATUS_ENDPOINT.format(pairing_id=pairing_id)}"
    while time.time() < deadline:
        response = _request_json(status_url, method="GET", headers=headers)
        data = _extract_data(response)
        if data:
            status = str(data.get("status") or data.get("state") or "").upper()
            if status in {"ISSUED", "READY", "APPROVED", "ACTIVE"}:
                return data
            if status in {"EXPIRED", "REJECTED"}:
                return None
        time.sleep(PAIRING_POLL_INTERVAL_SEC)
    return None


def _print_qr(url: str) -> None:
    print("Pairing URL:", url)
    try:
        import qrcode  # type: ignore

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        print("Install 'qrcode' to render QR in terminal.")


def _merge_defaults(fields: List[Dict[str, str]], existing: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = dict(existing)
    for field in fields:
        key = field["key"]
        if key not in merged:
            merged[key] = field["default"]
    return merged


def _validate_required(values: Dict[str, str]) -> List[str]:
    missing = []
    for key in REQUIRED_KEYS:
        if _is_placeholder(values.get(key)):
            missing.append(key)
    return missing


def prompt_cli(fields: List[Dict[str, str]], existing: Dict[str, str], advanced: bool) -> Dict[str, str]:
    values = _merge_defaults(fields, existing)
    for field in fields:
        key = field["key"]
        default = values.get(key, "")
        required = key in REQUIRED_KEYS
        if not advanced and not required:
            continue

        label = field["comment"] or key
        prompt = f"{label} ({key})"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        while True:
            if _is_secret_key(key):
                import getpass

                entered = getpass.getpass(prompt)
            else:
                entered = input(prompt)

            if not entered:
                entered = default
            values[key] = entered

            if required and _is_placeholder(values.get(key)):
                print("A value is required.")
                continue
            break

    return values


def _has_display() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def _render_form(
    fields: List[Dict[str, str]],
    values: Dict[str, str],
    missing: List[str],
    device_name: str,
) -> str:
    rows: List[str] = []
    for field in fields:
        key = field["key"]
        comment = field["comment"] or key
        value = values.get(key, field["default"])
        is_secret = _is_secret_key(key)
        input_type = "password" if is_secret else "text"
        placeholder = ""
        note = ""
        if is_secret and values.get(key):
            value = ""
            note = "<div class=\"note\">Leave blank to keep current value.</div>"
        required_attr = "required" if (key in REQUIRED_KEYS and _is_placeholder(values.get(key))) else ""
        if key in missing:
            placeholder = " required"
        rows.append(
            """
            <div class="field">
              <label>{label}<span class="key">{key}</span></label>
              <input type="{input_type}" name="{key}" value="{value}" {required} placeholder="{placeholder}">
              {note}
            </div>
            """.format(
                label=html.escape(comment),
                key=html.escape(key),
                input_type=input_type,
                value=html.escape(value or ""),
                required=required_attr,
                placeholder=html.escape(placeholder.strip()),
                note=note,
            )
        )

    error_block = ""
    if missing:
        error_items = " ".join(html.escape(key) for key in missing)
        error_block = f"<div class=\"error\">Missing required values: {error_items}</div>"

    provision_block = """
          <div class="card">
            <h2>Auto Provision (recommended)</h2>
            <p class="muted">
              Login with a space owner/admin account to create a device credential.
              Your account is not stored on the device.
            </p>
            <div class="grid">
              <div class="field">
                <label>Account username</label>
                <input type="text" id="prov-username" autocomplete="username">
              </div>
              <div class="field">
                <label>Account password</label>
                <input type="password" id="prov-password" autocomplete="current-password">
              </div>
              <div class="field">
                <label>Space</label>
                <select id="prov-space-select" disabled>
                  <option value="">Login to load spaces</option>
                </select>
              </div>
              <div class="field">
                <label>Device name</label>
                <input type="text" id="prov-device" value="{device_name}">
              </div>
            </div>
            <div class="actions">
              <button type="button" id="prov-login" onclick="loadSpaces()">Login & Load Spaces</button>
              <button type="button" id="prov-create" onclick="provisionDevice()" disabled>Create Device</button>
            </div>
            <div id="provision-status" class="status"></div>
          </div>
    """.format(device_name=html.escape(device_name))

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Nuvion Agent Setup</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f6f4f0;
            --card: #ffffff;
            --ink: #1a1a1a;
            --muted: #606060;
            --accent: #2f6b4f;
            --border: #e3ded7;
          }}
          body {{
            margin: 0;
            font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--ink);
          }}
          .wrap {{
            max-width: 820px;
            margin: 40px auto 80px;
            padding: 0 20px;
          }}
          header h1 {{
            font-size: 28px;
            margin-bottom: 6px;
          }}
          header p {{
            color: var(--muted);
            margin-top: 0;
          }}
          .card {{
            background: var(--card);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
          }}
          .card + .card {{
            margin-top: 20px;
          }}
          .field {{
            display: flex;
            flex-direction: column;
            margin-bottom: 18px;
          }}
          label {{
            font-weight: 600;
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
            gap: 12px;
          }}
          .key {{
            font-size: 12px;
            color: var(--muted);
            font-weight: 400;
          }}
          input {{
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
          }}
          select {{
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background: white;
          }}
          .note {{
            font-size: 12px;
            color: var(--muted);
            margin-top: 6px;
          }}
          .actions {{
            display: flex;
            justify-content: flex-end;
            margin-top: 18px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
          }}
          .muted {{
            color: var(--muted);
            margin-top: 4px;
            margin-bottom: 16px;
          }}
          .status {{
            margin-top: 12px;
            font-size: 13px;
            color: var(--muted);
          }}
          button {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
          }}
          .error {{
            background: #ffe3e3;
            color: #8a1f1f;
            padding: 12px 14px;
            border-radius: 8px;
            margin-bottom: 18px;
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <header>
            <h1>Nuvion Agent Setup</h1>
            <p>Enter device settings and save.</p>
          </header>
          {provision_block}
          <div class="card">
            {error_block}
            <form method="post" action="/save">
              {rows}
              <div class="actions">
                <button type="submit">Save</button>
              </div>
            </form>
          </div>
        </div>
        <script>
          async function loadSpaces() {{
            const statusEl = document.getElementById("provision-status");
            const loginBtn = document.getElementById("prov-login");
            const createBtn = document.getElementById("prov-create");
            const spaceSelect = document.getElementById("prov-space-select");
            const serverBaseUrl = document.querySelector('input[name="NUVION_SERVER_BASE_URL"]').value.trim();
            const username = document.getElementById("prov-username").value.trim();
            const password = document.getElementById("prov-password").value;
            if (!serverBaseUrl || !username || !password) {{
              statusEl.textContent = "Server URL, username, and password are required.";
              return;
            }}
            loginBtn.disabled = true;
            createBtn.disabled = true;
            statusEl.textContent = "Loading spaces...";
            spaceSelect.innerHTML = "<option value=''>Loading...</option>";
            spaceSelect.disabled = true;
            try {{
              const resp = await fetch("/api/spaces", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ serverBaseUrl, username, password }})
              }});
              const data = await resp.json();
              if (!resp.ok || data.error) {{
                statusEl.textContent = data.error || "Failed to load spaces.";
                return;
              }}
              const spaces = (data.spaces || data || []);
              spaceSelect.innerHTML = "";
              if (!spaces.length) {{
                spaceSelect.innerHTML = "<option value=''>No spaces found</option>";
                statusEl.textContent = "No spaces found for this account.";
                return;
              }}
              spaceSelect.innerHTML = "<option value=''>Select a space</option>";
              spaces.forEach((space) => {{
                const option = document.createElement("option");
                option.value = space.id;
                option.textContent = `${{space.name || "Space"}} (#${{space.id}})`;
                spaceSelect.appendChild(option);
              }});
              spaceSelect.disabled = false;
              createBtn.disabled = false;
              statusEl.textContent = "Spaces loaded. Select a space to provision.";
            }} catch (err) {{
              statusEl.textContent = "Failed to load spaces: " + err;
            }} finally {{
              loginBtn.disabled = false;
            }}
          }}

          async function provisionDevice() {{
            const statusEl = document.getElementById("provision-status");
            const spaceSelect = document.getElementById("prov-space-select");
            const spaceId = spaceSelect.value;
            if (!spaceId) {{
              statusEl.textContent = "Please select a space.";
              return;
            }}
            statusEl.textContent = "Provisioning device credentials...";
            const payload = {{
              serverBaseUrl: document.querySelector('input[name="NUVION_SERVER_BASE_URL"]').value.trim(),
              username: document.getElementById("prov-username").value.trim(),
              password: document.getElementById("prov-password").value,
              spaceId: spaceId,
              deviceName: document.getElementById("prov-device").value.trim()
            }};
            try {{
              const resp = await fetch("/api/provision", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(payload)
              }});
              const data = await resp.json();
              if (!resp.ok || data.error) {{
                statusEl.textContent = data.error || "Provisioning failed.";
                return;
              }}
              const deviceUsername = data.deviceUsername || data.username || "";
              const devicePassword = data.devicePassword || data.password || data.deviceSecret || "";
              const rtpIp = data.rtpRemoteIp || data.rtpIp || "";
              if (deviceUsername) {{
                document.querySelector('input[name="NUVION_DEVICE_USERNAME"]').value = deviceUsername;
              }}
              if (devicePassword) {{
                document.querySelector('input[name="NUVION_DEVICE_PASSWORD"]').value = devicePassword;
              }}
              if (rtpIp) {{
                document.querySelector('input[name="NUVION_RTP_REMOTE_IP"]').value = rtpIp;
              }}
              statusEl.textContent = "Device credentials created. Review and click Save.";
            }} catch (err) {{
              statusEl.textContent = "Provisioning error: " + err;
            }}
          }}
        </script>
      </body>
    </html>
    """.format(error_block=error_block, rows="\n".join(rows), provision_block=provision_block)


def run_web_setup(
    config_path: Path,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    lines, fields = load_template()
    existing = _merge_defaults(fields, read_env(config_path))
    device_name = socket.gethostname()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, status: HTTPStatus, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: HTTPStatus, body: Dict[str, object]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                missing = _validate_required(existing)
                body = _render_form(fields, existing, missing, device_name)
                self._send_html(HTTPStatus.OK, body)
                return
            if self.path == "/health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/spaces":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                server_base_url = str(payload.get("serverBaseUrl") or existing.get("NUVION_SERVER_BASE_URL") or "").strip()
                username = str(payload.get("username") or "").strip()
                password = str(payload.get("password") or "")
                if not server_base_url or not username or not password:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing credentials or server base URL."})
                    return
                spaces = _fetch_spaces(server_base_url, username, password)
                if spaces is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Failed to load spaces."})
                    return
                self._send_json(HTTPStatus.OK, {"spaces": spaces})
                return

            if self.path == "/api/provision":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                server_base_url = str(payload.get("serverBaseUrl") or existing.get("NUVION_SERVER_BASE_URL") or "").strip()
                username = str(payload.get("username") or "").strip()
                password = str(payload.get("password") or "")
                space_id = str(payload.get("spaceId") or "").strip()
                device_name_local = str(payload.get("deviceName") or device_name).strip()

                if not server_base_url or not username or not password or not space_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required provisioning fields."})
                    return

                data = _provision_device(server_base_url, username, password, space_id, device_name_local)
                if not data:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Provisioning failed."})
                    return
                self._send_json(HTTPStatus.OK, data)
                return

            if self.path != "/save":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            parsed = urllib.parse.parse_qs(body)
            values = dict(existing)

            for field in fields:
                key = field["key"]
                posted = parsed.get(key, [""])[0].strip()
                if not posted and _is_secret_key(key) and existing.get(key):
                    values[key] = existing[key]
                else:
                    values[key] = posted

            missing = _validate_required(values)
            if missing:
                body = _render_form(fields, values, missing, device_name)
                self._send_html(HTTPStatus.BAD_REQUEST, body)
                return

            write_env(config_path, lines, values)
            success = """
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <title>Saved</title>
              </head>
              <body>
                <h2>Saved</h2>
                <p>Settings saved. Restart the service to apply.</p>
              </body>
            </html>
            """
            self._send_html(HTTPStatus.OK, success)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer((host, port), Handler)
    address = f"http://{host}:{server.server_address[1]}"
    print(f"Setup page: {address}")
    if open_browser:
        browser_target = address
        if host == "0.0.0.0":
            browser_target = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            webbrowser.open(browser_target)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Setup server stopped.")
    finally:
        server.server_close()


def run_qr_setup(config_path: Path, advanced: bool) -> None:
    lines, fields = load_template()
    existing = _merge_defaults(fields, read_env(config_path))
    server_base_url = str(existing.get("NUVION_SERVER_BASE_URL") or "").strip()
    if not server_base_url:
        raise RuntimeError("NUVION_SERVER_BASE_URL is required for QR setup.")
    device_name = socket.gethostname()

    pairing = _init_pairing(server_base_url, device_name)
    if not pairing:
        raise RuntimeError("Failed to initiate pairing. Check server URL and network.")

    pairing_url = str(
        pairing.get("pairingUrl")
        or pairing.get("url")
        or pairing.get("pairingURL")
        or ""
    )
    pairing_code = str(pairing.get("pairingCode") or pairing.get("code") or "").strip()
    pairing_id = str(pairing.get("pairingId") or pairing.get("id") or "").strip()
    pairing_secret = str(pairing.get("pairingSecret") or pairing.get("secret") or "").strip() or None

    if pairing_code:
        print("Pairing code:", pairing_code)
    if pairing_url:
        _print_qr(pairing_url)

    if not pairing_id:
        raise RuntimeError("Pairing response missing pairingId.")

    print("Waiting for pairing approval...")
    result = _wait_for_pairing(server_base_url, pairing_id, pairing_secret)
    if not result:
        raise RuntimeError("Pairing not approved or expired.")

    values = dict(existing)
    device_username = (
        result.get("deviceUsername")
        or result.get("username")
        or result.get("deviceId")
    )
    device_password = (
        result.get("devicePassword")
        or result.get("password")
        or result.get("deviceSecret")
        or result.get("secret")
    )
    rtp_ip = result.get("rtpRemoteIp") or result.get("rtpIp")

    if device_username:
        values["NUVION_DEVICE_USERNAME"] = str(device_username)
    if device_password:
        values["NUVION_DEVICE_PASSWORD"] = str(device_password)
    if rtp_ip:
        values["NUVION_RTP_REMOTE_IP"] = str(rtp_ip)

    missing = _validate_required(values)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing required values after pairing: {missing_str}")

    write_env(config_path, lines, values)
    print(f"Saved: {config_path}")


def setup_config(
    config_path: Optional[str] = None,
    use_web: Optional[bool] = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    advanced: bool = False,
    qr: bool = False,
) -> Path:
    path = resolve_config_path(config_path)

    if qr:
        use_web = False

    if use_web is None:
        use_web = _has_display()

    if use_web:
        run_web_setup(path, host=host, port=port, open_browser=open_browser)
    else:
        if not qr and not _has_display():
            qr = True
        if qr:
            run_qr_setup(path, advanced=advanced)
        else:
            lines, fields = load_template()
            existing = read_env(path)
            values = prompt_cli(fields, existing, advanced=advanced)
            missing = _validate_required(values)
            if missing:
                missing_str = ", ".join(missing)
                raise RuntimeError(f"Missing required values: {missing_str}")
            write_env(path, lines, values)
            print(f"Saved: {path}")

    try:
        os.environ["NUV_AGENT_CONFIG"] = str(path)
        for key, value in read_env(path).items():
            os.environ[key] = value

        from nuvion_app.runtime.bootstrap import ensure_ready

        ready = ensure_ready(stage="setup")
        if ready:
            print("Runtime bootstrap: ready")
        else:
            print("Runtime bootstrap: degraded (backend switched to none for this session)")
    except Exception as exc:
        print(f"Runtime bootstrap skipped: {exc}")

    return path
