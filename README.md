# invetflow-agent

LiveKit **Agents** worker that joins interview rooms as the AI interviewer. It loads questions from the Invetflow API and syncs **Candidate** / **AI** lines to the session transcript.

## Prerequisites

- Same [LiveKit](https://livekit.io/) project as `invetflow-server` (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`).
- Register an agent in LiveKit (e.g. Cloud → Agents) with name **`invetflow-agent`**.
- [OpenAI](https://platform.openai.com/) API key for LLM (and optionally STT/TTS) (`OPENAI_API_KEY`).
- Optional: [Deepgram](https://developers.deepgram.com/home) API key for STT (`DEEPGRAM_API_KEY`).
- Optional: [Cartesia](https://docs.cartesia.ai/get-started/overview) API key for TTS (`CARTESIA_API_KEY`).

## Configuration

Create `.env` in this directory (see `.env.example`). Important:

| Variable | Purpose |
|----------|---------|
| `LIVEKIT_AGENT_NAME` | Must be **`invetflow-agent`** and match the server and LiveKit registration (default in code is `invetflow-agent`). |
| `INVETFLOW_API_URL` | Base URL of `invetflow-server`, e.g. `http://127.0.0.1:3001` |
| `AGENT_API_SECRET` | Must match `AGENT_API_SECRET` on the server. |
| `STT_PROVIDER` | STT backend: `openai` (default) or `deepgram`. |
| `TTS_PROVIDER` | TTS backend: `openai` (default) or `cartesia`. |
| `OPENAI_STT_MODEL` | OpenAI STT model (when `STT_PROVIDER=openai`). Defaults to `gpt-4o-transcribe`. |
| `OPENAI_STT_REALTIME` | OpenAI realtime **transcription** mode (when `STT_PROVIDER=openai`). Defaults to `true`. This enables OpenAI server VAD and interim transcript events; it does not switch the agent to a single speech-to-speech realtime LLM model. |
| `OPENAI_STT_LANGUAGE` | Input language code when language detection is off. Defaults to `en`. |
| `OPENAI_STT_DETECT_LANGUAGE` | Set `true` for multilingual or code-switched interviews. Defaults to `false`. |
| `OPENAI_STT_NOISE_REDUCTION` | OpenAI realtime input noise reduction: `near_field` (default), `far_field`, or `off`. This is the single noise-processing stage; do not also enable browser-side or LiveKit noise cancellation. |
| `OPENAI_STT_SILENCE_DURATION_MS` | OpenAI realtime server VAD silence window. Defaults to `1000` ms to tolerate thinking pauses. |
| `OPENAI_STT_PROMPT` | Optional custom transcription vocabulary/context. If unset, the agent derives one from the interview title, role, and questions. |
| `DEEPGRAM_STT_MODEL` | Deepgram model (when `STT_PROVIDER=deepgram`). Defaults to `nova-3`. |
| `DEEPGRAM_STT_LANGUAGE` | Deepgram language (when detect-language is off). Defaults to `en-US`. |
| `OPENAI_TTS_MODEL` / `OPENAI_TTS_VOICE` | OpenAI TTS model/voice (when `TTS_PROVIDER=openai`). Defaults: `gpt-4o-mini-tts` + `ash`. |
| `CARTESIA_TTS_MODEL` / `CARTESIA_TTS_VOICE` | Cartesia model/voice (when `TTS_PROVIDER=cartesia`). Defaults: `sonic-3` + Cartesia default voice id. |
| `LIVEKIT_AGENT_NOISE_CANCELLATION` | Inbound LiveKit noise cancellation: `off` (default), `nc`, `bvc`, or `telephony`. Off by default so OpenAI's STT model handles noise reduction; enable only for genuinely noisy rooms. |
| `SILERO_MIN_SILENCE_DURATION` | Silero end-of-speech window. Defaults to `0.90` s. Used whenever OpenAI realtime STT server VAD is not active (for Deepgram STT and OpenAI non-realtime STT). |
| `AGENT_MIN_ENDPOINTING_DELAY` / `AGENT_MAX_ENDPOINTING_DELAY` | Extra turn-completion guardrails tuned for interview pauses. Defaults to `1.0` / `3.5` seconds. |

The Rust server also needs:

- `LIVEKIT_AGENT_NAME=invetflow-agent` — so `POST /api/interviews/:id/join` dispatches this worker to new rooms.
- `AGENT_API_SECRET` — enables `/api/agent/...` for this worker.

## Run (development)

The virtualenv lives **only** at **`invetflow-agent/.venv`** (gitignored). Do not put a `.venv` at the monorepo root and do not use a root symlink to this folder.

**Editor / Pylance:** The interpreter is only inside that path. For a clean setup that matches the usual “.venv at workspace root” pattern, open **`invetflow-agent/invetflow-agent.code-workspace`**, or **File → Open Folder…** and choose the **`invetflow-agent`** directory (not the whole `invetflow` repo). Then **Python: Select Interpreter** should list `.venv` under Python, or you can enter: `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows). If you keep the monorepo open instead, the repo’s `.vscode` still points to `invetflow-agent/.venv` — use **Enter interpreter path** with the full path to that `python` binary if it doesn’t appear in the list.

```bash
cd invetflow-agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
python main.py dev
```

A `.venv-agent` folder at the **repo root** was a one-off test environment and is not part of this project; safe to delete if you still have it.

For production, follow [LiveKit agent deployment](https://docs.livekit.io/agents/ops/deployment/) (same env vars, command is typically `python main.py start`).

## How it ties to the stack

1. Candidate **joins** an interview → server creates a LiveKit room and **CreateDispatch** for `LIVEKIT_AGENT_NAME` with JSON metadata: `interviewId` (and legacy `sessionId`).
2. This worker receives the job, fetches `GET /api/agent/interviews/{id}/context`, starts a voice `AgentSession`, and posts transcript lines to `POST /api/agent/interviews/{id}/transcript`.
3. When the remote candidate publishes a **microphone** audio track, the agent calls `POST /api/agent/interviews/{id}/candidate-audio-egress` so the server starts **LiveKit Track Egress** to S3 (configure `AWS_*` and `S3_BUCKET` on **invetflow-server**).
4. On job shutdown, the agent calls `POST /api/agent/interviews/{id}/refine-transcript`. The server downloads the object from S3 and runs **non-realtime** `gpt-4o-transcribe`, then merges the result into Mongo (AI/System lines preserved; live **Candidate** lines replaced by one refined block). The same refinement is also queued when the candidate ends the session from the app.
