import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chibi_ram_backend as backend


class BackendServerMixin:
    def setUp(self):
        self._original_config = backend.CONFIG
        backend.CONFIG = backend.AppConfig(
            gemini_api_key="dummy-gemini-key",
            fish_audio_api_key="dummy-fish-key",
            fish_audio_ram_model_id="dummy-fish-model",
            chat_api_token="shared-test-token",
            host="127.0.0.1",
            port=0,
            debug=False,
        )
        self.server = backend.create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._wait_for_server()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        backend.CONFIG = self._original_config

    def _wait_for_server(self):
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1):
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("Server did not start in time.")


class ChatAuthTest(BackendServerMixin, unittest.TestCase):
    def test_chat_post_requires_shared_token(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/chat",
            data=json.dumps({"message": "こんにちは"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(context.exception.code, 401)
        context.exception.close()

    def test_chat_get_accepts_shared_token(self):
        with patch.object(
            backend,
            "process_prompt",
            return_value={"status": "processing"},
        ):
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/chat?q=hello",
                headers={"X-Chibi-Ram-Token": "shared-test-token"},
            )

            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload, {"status": "processing", "message": "hello"})


if __name__ == "__main__":
    unittest.main()