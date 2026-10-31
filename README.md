# Synapse AI API

## Let a colleague run a local copy for free

1. Send the project source files to your colleague as a ZIP or through a private Git repository.
2. Do not include `.venv`, `.env`, `synapse.db`, or `.data`. These contain local packages, secrets, account data, or recordings. The `.gitignore` already excludes them from Git, but check manual ZIPs yourself.
3. On Windows, install Python 3.10 or newer, extract the project folder, and double-click `START_SYNAPSE.bat`.
4. On first launch the script creates a local virtual environment and installs the dependencies. It then opens `http://127.0.0.1:8000` in the browser.
5. To use AI analysis and transcription, the colleague must add their own `OPENAI_API_KEY` to their private `.env` file. Do not send your key. OpenAI API usage may be billed to the key owner.

The application, local SQLite database, and hosting on the colleague's own computer have no hosting fee. This does not make third-party AI usage free. Each local copy has its own accounts, database, and recordings; this method does not share one common history between computers.

To use the same installation from phones or multiple computers, the app must instead be hosted on an HTTPS server. Free hosting plans may have limits or sleep, and AI usage may still cost money.

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\malgi\.vscode-shared\SYNAPSE AI - moja apka"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set an OpenAI key in the current PowerShell session to enable analysis and transcription:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Alternatively, copy `.env.example` to `.env` and set `OPENAI_API_KEY` there. The application loads `.env` on startup; `.env` is ignored by Git and must never be committed.

Run the API:

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation and `http://127.0.0.1:8000/health` for the health check. SQLite creates `synapse.db` on first startup.

## Install on a phone

Synapse AI is a Progressive Web App (PWA). It must be hosted over HTTPS before it can be installed from a phone; `127.0.0.1` is only for local computer testing.

- iPhone: open the HTTPS app address in Safari, tap Share, then choose Add to Home Screen.
- Android: open the HTTPS app address in Chrome and use Install app or Add to Home screen.
- Sign in to the installed app as usual. API requests, account data, recordings, and transcripts are not cached for offline use.

## Included

- Account registration, login, logout, and authenticated user identity.
- After registration, accept the one-time content-analysis consent at `POST /api/v1/auth/analysis-consent`; its version and timestamp are stored on the account. Content uploads and audio sessions are unavailable until accepted.
- Multilingual model configurations for Polish (`pl`), English (`en`), French (`fr`), and Spanish (`es`) are seeded in the database for analysis and transcription.
- Use `GET /api/v1/languages` to list enabled language/model configurations and `PUT /api/v1/me/language` with `{"language":"fr"}` to set the signed-in user's default response and transcription language.
- Analysis uses the requested language, then the source language, then the account preference. Responses are instructed to use natural, grammatical language, preserve the source meaning, and avoid unsupported factual claims.
- Private text sources and analysis history, each analysis linked to its source and creation time.
- User-scoped database queries; clients cannot choose the owner of a record.
- Audio sessions require participant confirmation before recording. This confirmation is checked for the session but is not stored as a separate consent record.
- Start/pause/resume/stop controls and a WebSocket audio stream. Raw audio is saved only when `save_raw_audio` is enabled; there is no separate storage-consent field.
- Optional transcription using OpenAI. Transcripts are saved as user-owned sources.
- Voice-control preference and authenticated voice-origin controls. Speech recognition and recording UI must be provided by a frontend.

Use a strong unique password. For deployment, configure HTTPS and set `COOKIE_SECURE=true`; use a managed database, secret storage, backups, and a retention policy before storing real user data. The SQLite database and audio directory are local-development defaults.
