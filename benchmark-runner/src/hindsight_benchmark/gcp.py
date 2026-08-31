"""GCP auth helpers: a local proxy that forwards OpenAI-style requests to a
Vertex AI endpoint with a fresh OAuth bearer token per request.

Vertex access tokens expire hourly; benchmark runs outlast that. The proxy
keeps credentials out of every config file and client: callers point their
OpenAI-compatible base_url at localhost and never see a token.
"""

import http.server
import subprocess
import threading
import time
import urllib.error
import urllib.request


def access_token() -> str:
    out = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"could not mint access token: {out.stderr[-500:]}")
    return out.stdout.strip()


class TokenCache:
    def __init__(self):
        self._token = ""
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            # gcloud caches tokens itself, so a "fresh" token may arrive with
            # only minutes of life left. The short TTL narrows that window;
            # invalidate() closes it when a 401 proves the token dead.
            if not self._token or time.time() - self._fetched_at > 5 * 60:
                self._token = access_token()
                self._fetched_at = time.time()
            return self._token

    def invalidate(self):
        with self._lock:
            self._token = ""


def start_token_proxy(upstream_base: str, port: int) -> http.server.ThreadingHTTPServer:
    """Forward requests to upstream_base with a fresh bearer token.

    Clients speak /v1/chat/completions; the upstream path already carries its
    own /v1(beta1) prefix, so the incoming /v1 is stripped. Runs in a daemon
    thread; dies with the process.

    An upstream ending in a ``:verb`` (``:predict``) is a complete URL and is
    used as-is: not every endpoint exposes the OpenAI-style
    ``/endpoints/ID/chat/completions`` passthrough, but ``:predict`` forwards
    the body to the container's own route and returns its reply unchanged.
    """
    cache = TokenCache()
    verb_upstream = upstream_base.rsplit("/", 1)[-1].startswith(":") or ":" in upstream_base.rsplit("/", 1)[-1]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            path = self.path.removeprefix("/v1")
            url = upstream_base if verb_upstream else upstream_base + path
            try:
                for attempt in range(2):
                    req = urllib.request.Request(url, data=body, method="POST")
                    req.add_header("Content-Type", "application/json")
                    req.add_header("Authorization", f"Bearer {cache.get()}")
                    try:
                        with urllib.request.urlopen(req, timeout=600) as resp:
                            payload = resp.read()
                            self.send_response(resp.status)
                            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                            self.send_header("Content-Length", str(len(payload)))
                            self.end_headers()
                            self.wfile.write(payload)
                            return
                    except urllib.error.HTTPError as e:
                        # A 401 means the cached token died early; refresh and
                        # retry once before surfacing the error.
                        if e.code == 401 and attempt == 0:
                            cache.invalidate()
                            continue
                        payload = e.read()
                        self.send_response(e.code)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
            except (BrokenPipeError, ConnectionResetError):
                # The client gave up (request timeout) before the upstream
                # answered. Its request already counts as failed on its side.
                pass

        def do_GET(self):
            self.do_POST()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Token proxy on http://127.0.0.1:{port} -> {upstream_base}", flush=True)
    return server


def vertex_openai_upstream(project: str, location: str = "global") -> str:
    """OpenAI-compatible chat-completions base for Google-hosted models on
    Vertex (Gemini). Model names are passed as 'google/<model>'."""
    host = "aiplatform.googleapis.com" if location == "global" \
        else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"
