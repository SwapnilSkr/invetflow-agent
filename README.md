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

## Meeting Agent (`meeting_agent.py`)

A second worker in this repo handles **human-meeting transcription** for the Human Video Interview feature. It joins recruiter-scheduled rooms, transcribes all participants via Deepgram nova-3, and streams live captions to the recruiter UI over a LiveKit DataChannel.

### Prerequisites

- **`DEEPGRAM_API_KEY`** — required (not optional). The meeting agent uses Deepgram exclusively for STT.
- `MEETING_AGENT_NAME` — must match `livekit_meeting_agent_name` in `invetflow-server` config. Default is `invetflow-meeting-agent`.
- `INVETFLOW_API_BASE` or `INVETFLOW_API_URL` — base URL of `invetflow-server`.
- `AGENT_API_SECRET` — same shared secret as the AI interview agent.

### Run (development)

The meeting agent connects to LiveKit Cloud from your local machine — no deployment needed for development. When the Rust server dispatches for `invetflow-meeting-agent`, LiveKit routes the job to your local process.

```bash
# In a second terminal alongside main.py dev
cd invetflow-agent
source .venv/bin/activate
python meeting_agent.py dev
```

Watch for `registered agent: invetflow-meeting-agent` in the logs. Then start a Human Interview session from the recruiter app — the agent should join the room and begin transcribing.

### Production deployment

Run `meeting_agent.py` as a **separate container / service** from `main.py` (Option A — recommended for independent scaling and restart policies):

```bash
docker run -d \
  --name invetflow-meeting-agent \
  --restart unless-stopped \
  -e LIVEKIT_URL=wss://your-project.livekit.cloud \
  -e LIVEKIT_API_KEY=xxx \
  -e LIVEKIT_API_SECRET=xxx \
  -e DEEPGRAM_API_KEY=xxx \
  -e AGENT_API_SECRET=xxx \
  -e INVETFLOW_API_BASE=https://api.invetflow.com \
  -e MEETING_AGENT_NAME=invetflow-meeting-agent \
  invetflow-agent \
  python meeting_agent.py start
```

The default `CMD` in the Dockerfile runs `main.py start` (AI interview agent). Override with `python meeting_agent.py start` when running the meeting agent — this does **not** affect the existing AI interview agent container.

### How it ties to the stack

1. Recruiter clicks **Start session** → server calls `create_agent_dispatch("meet-<oid>", "invetflow-meeting-agent", metadata)`.
2. LiveKit Cloud routes the dispatch to the connected meeting-agent worker.
3. Agent joins the room, subscribes to all participant audio, skips its own `agent-` track.
4. Finals are `POST`ed to `POST /api/agent/human-interviews/transcript` with `X-Invetflow-Agent-Secret`.
5. Live (interim) turns are broadcast over `DataChannel` topic `invetflow-meeting-transcript` — the recruiter UI renders them immediately.
6. On room close, the server transitions `transcript_status` to `Complete` and runs GPT-4o-mini summarization in the background.

## How it ties to the stack (AI interview agent)

1. Candidate **joins** an interview → server creates a LiveKit room and **CreateDispatch** for `LIVEKIT_AGENT_NAME` with JSON metadata: `interviewId` (and legacy `sessionId`).
2. This worker receives the job, fetches `GET /api/agent/interviews/{id}/context`, starts a voice `AgentSession`, and posts transcript lines to `POST /api/agent/interviews/{id}/transcript`.
3. When the remote candidate publishes a **microphone** audio track, the agent calls `POST /api/agent/interviews/{id}/candidate-audio-egress` so the server starts **LiveKit Track Egress** to S3 (configure `AWS_*` and `S3_BUCKET` on **invetflow-server**).
4. On job shutdown, the agent calls `POST /api/agent/interviews/{id}/refine-transcript`. The server downloads the object from S3 and runs **non-realtime** `gpt-4o-transcribe`, then merges the result into Mongo (AI/System lines preserved; live **Candidate** lines replaced by one refined block). The same refinement is also queued when the candidate ends the session from the app.
