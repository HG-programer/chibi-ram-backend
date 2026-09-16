import json
import threading
import time
import unittest
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chibi_ram_backend as backend


class BackendServerMixin:
    def setUp(self):
        self.server = backend.create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._wait_for_server()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _wait_for_server(self):
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1):
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("Server did not start in time.")


class HealthEndpointTest(BackendServerMixin, unittest.TestCase):
    def test_health_endpoint_returns_service_status(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as response:
            self.assertEqual(response.status, 200)
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload, {"ok": True, "service": "chibi-ram-backend"})


if __name__ == "__main__":
    unittest.main()
