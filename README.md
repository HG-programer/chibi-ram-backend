# Chibi Ram Backend

A small Python backend for the Chibi Ram robotics project. It keeps the current working architecture intact: text goes to Gemini, Gemini returns short Japanese speech in Ram's personality, Fish Audio turns that text into MP3, and the ESP32 polls the HTTP API for the latest audio.

## What it exposes

- `GET /health`
- `GET /status`
- `GET /ack`
- `GET /ram_speech.mp3`
- `GET /chat?q=...` for legacy compatibility
- `POST /chat` with JSON `{ "message": "..." }`
- `POST /process` as an alias for the chat processor

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your real credentials locally.

```powershell
copy .env.example .env
```

Run the backend:

```powershell
python chibi_ram_backend.py
```

Then test:

- `http://localhost:8000/health`
- `http://localhost:8000/status`

## Environment variables

Required for chat generation:

- `GEMINI_API_KEY`
- `FISH_AUDIO_API_KEY`
- `FISH_AUDIO_RAM_MODEL_ID`
- `CHAT_API_TOKEN`

Deployment port handling:

- `PORT` is preferred in production
- `SERVER_PORT` is kept as a local fallback

## Render deployment

1. Push this repository to GitHub.
2. Create a new Render Web Service.
3. Connect the GitHub repository.
4. Use Python as the runtime.
5. Install dependencies with `pip install -r requirements.txt`.
6. Start the app with `python chibi_ram_backend.py`.
7. Set these environment variables on Render:
   - `GEMINI_API_KEY`
   - `FISH_AUDIO_API_KEY`
   - `FISH_AUDIO_RAM_MODEL_ID`
   - `CHAT_API_TOKEN`
8. Deploy the service.
9. Verify `https://SERVICE.onrender.com/health`.
10. Verify `https://SERVICE.onrender.com/status`.

Render injects `PORT` automatically; the server already reads it, so you do not need to set that manually.

For `/chat` and `/process`, send the shared token in `X-Chibi-Ram-Token` or `Authorization: Bearer ...`.

The ESP32 should keep polling the stable MP3 URL at `/ram_speech.mp3?seq=...` and continue using the existing ack/status flow.

## Notes

- The backend listens on `0.0.0.0` and respects Render's `PORT` variable.
- The generated audio file is replaced atomically so a new request never corrupts the last valid MP3.
- No secrets are committed to the repository.
