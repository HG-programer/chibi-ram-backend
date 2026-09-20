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


class VoiceChatEndpointTest(BackendServerMixin, unittest.TestCase):
    def test_voice_chat_requires_token(self):
        fake_wav = b"RIFF....WAVEfmt ...."
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/voice-chat",
            data=fake_wav,
            method="POST",
            headers={"Content-Type": "audio/wav"},
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(ctx.exception.code, 401)
        ctx.exception.close()

    def test_voice_chat_rejects_invalid_token(self):
        fake_wav = b"RIFF....WAVEfmt ...."
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/voice-chat",
            data=fake_wav,
            method="POST",
            headers={
                "Content-Type": "audio/wav",
                "X-Chibi-Ram-Token": "wrong-token",
            },
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(ctx.exception.code, 401)
        ctx.exception.close()

    def test_voice_chat_accepts_binary_wav_with_valid_token(self):
        fake_wav = b"RIFF44WAVEfmt 16000data1234"
        expected_result = {
            "ok": True,
            "seq": 1,
            "transcript": "hello Ram",
            "reply": "こんにちは、ハル。",
            "text": "こんにちは、ハル。",
            "audio_url": "/ram_speech.mp3?seq=1",
        }

        with patch.object(backend, "process_voice_prompt", return_value=expected_result) as mock_process:
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/voice-chat",
                data=fake_wav,
                method="POST",
                headers={
                    "Content-Type": "audio/wav",
                    "X-Chibi-Ram-Token": "shared-test-token",
                },
            )

            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read().decode("utf-8"))

            mock_process.assert_called_once_with(fake_wav)
            self.assertEqual(payload, expected_result)

    def test_voice_chat_accepts_multipart_form_data(self):
        boundary = "----TestBoundary123456"
        file_content = b"WAV_DUMMY_AUDIO_DATA_BYTES"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="my_voice.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + file_content + f"\r\n--{boundary}--\r\n".encode("utf-8")

        expected_result = {
            "ok": True,
            "seq": 2,
            "transcript": "wake up Ram",
            "reply": "起きているわよ、ハル。",
            "text": "起きているわよ、ハル。",
            "audio_url": "/ram_speech.mp3?seq=2",
        }

        with patch.object(backend, "process_voice_prompt", return_value=expected_result) as mock_process:
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/voice-chat",
                data=body,
                method="POST",
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "X-Chibi-Ram-Token": "shared-test-token",
                },
            )

            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read().decode("utf-8"))

            mock_process.assert_called_once_with(file_content)
            self.assertEqual(payload, expected_result)

    def test_voice_chat_rejects_empty_payload(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/voice-chat",
            data=b"",
            method="POST",
            headers={
                "Content-Type": "audio/wav",
                "X-Chibi-Ram-Token": "shared-test-token",
            },
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(ctx.exception.code, 400)
        ctx.exception.close()


class TranscribeAudioTest(unittest.TestCase):
    def setUp(self):
        self._original_config = backend.CONFIG
        backend.CONFIG = backend.AppConfig(
            gemini_api_key="test-gemini-key",
            fish_audio_api_key="test-fish-key",
            fish_audio_ram_model_id="test-fish-model",
            chat_api_token="test-token",
            host="127.0.0.1",
            port=0,
            debug=False,
        )

    def tearDown(self):
        backend.CONFIG = self._original_config

    def test_transcribe_audio_empty_bytes_raises(self):
        with self.assertRaises(ValueError):
            backend.transcribe_audio(b"")

    def test_transcribe_audio_no_api_key_raises(self):
        backend.CONFIG = backend.AppConfig(
            gemini_api_key="",
            fish_audio_api_key="test-fish-key",
            fish_audio_ram_model_id="test-fish-model",
            chat_api_token="test-token",
            host="127.0.0.1",
            port=0,
            debug=False,
        )
        with self.assertRaises(RuntimeError):
            backend.transcribe_audio(b"RIFF....")

    @patch("google.genai.Client")
    def test_transcribe_audio_primary_success(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = unittest.mock.MagicMock(
            text="Hello Ram, how are you?",
            candidates=[],
        )

        result = backend.transcribe_audio(b"RIFF_FAKE_AUDIO_DATA")
        self.assertEqual(result, "Hello Ram, how are you?")
        mock_client.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_transcribe_audio_fallback_when_first_model_fails(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.side_effect = [
            Exception("model overloaded"),
            unittest.mock.MagicMock(text="Fallback transcribed text", candidates=[]),
        ]

        result = backend.transcribe_audio(b"RIFF_FAKE_AUDIO_DATA")
        self.assertEqual(result, "Fallback transcribed text")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    @patch("google.genai.Client")
    def test_transcribe_audio_all_models_fail_returns_empty_string(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.side_effect = Exception("503 unavailable across all models")

        result = backend.transcribe_audio(b"RIFF_FAKE_AUDIO_DATA")
        self.assertEqual(result, "")


class ProcessVoicePromptTest(unittest.TestCase):
    def setUp(self):
        self._original_config = backend.CONFIG
        backend.CONFIG = backend.AppConfig(
            gemini_api_key="test-gemini-key",
            fish_audio_api_key="test-fish-key",
            fish_audio_ram_model_id="test-fish-model",
            chat_api_token="test-token",
            host="127.0.0.1",
            port=0,
            debug=False,
        )

    def tearDown(self):
        backend.CONFIG = self._original_config

    @patch.object(backend, "transcribe_audio", return_value="おはよう、ラム")
    @patch.object(backend, "call_gemini", return_value="……うるさいわね、ハル。")
    @patch.object(backend, "call_fish_audio")
    def test_process_voice_prompt_flow(self, mock_fish, mock_gemini, mock_transcribe):
        result = backend.process_voice_prompt(b"FAKE_WAV_BYTES")

        self.assertTrue(result["ok"])
        self.assertEqual(result["transcript"], "おはよう、ラム")
        self.assertEqual(result["reply"], "……うるさいわね、ハル。")
        self.assertEqual(result["text"], "……うるさいわね、ハル。")
        self.assertIn("ram_speech.mp3", result["audio_url"])
        mock_transcribe.assert_called_once_with(b"FAKE_WAV_BYTES")
        mock_gemini.assert_called_once_with("おはよう、ラム")
        mock_fish.assert_called_once_with("……うるさいわね、ハル。", emotion="gentle")

    @patch.object(backend, "transcribe_audio", return_value="")
    @patch.object(backend, "call_gemini", return_value="何かしらハル、聞こえなかったわ。")
    @patch.object(backend, "call_fish_audio")
    def test_process_voice_prompt_empty_transcript_uses_fallback(self, mock_fish, mock_gemini, mock_transcribe):
        result = backend.process_voice_prompt(b"SILENT_WAV_BYTES")

        self.assertTrue(result["ok"])
        self.assertEqual(result["transcript"], "（ハル様がお話しになりました）")
        self.assertEqual(result["reply"], "何かしらハル、聞こえなかったわ。")
        mock_gemini.assert_called_once_with("（ハル様がお話しになりました）")
        mock_fish.assert_called_once_with("何かしらハル、聞こえなかったわ。", emotion="gentle")


if __name__ == "__main__":
    unittest.main()
