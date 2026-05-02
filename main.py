"""
Invetflow AI interviewer — LiveKit Agents worker.

- Register the worker name in LiveKit (Cloud → Agents) as **invetflow-agent** (or set LIVEKIT_AGENT_NAME).
- Set the same value on the Rust server: LIVEKIT_AGENT_NAME=invetflow-agent
- AGENT_API_SECRET must match between this process and invetflow-server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.events import ConversationItemAddedEvent, UserInputTranscribedEvent
from livekit.plugins import openai, silero

try:
    from livekit.plugins import cartesia
except ImportError:  # pragma: no cover - optional runtime dependency
    cartesia = None

try:
    from livekit.plugins import deepgram
except ImportError:  # pragma: no cover - optional runtime dependency
    deepgram = None

try:
    from livekit.plugins import noise_cancellation
except ImportError:  # pragma: no cover - optional runtime dependency
    noise_cancellation = None

load_dotenv()

logger = logging.getLogger("invetflow-agent")

DEFAULT_AGENT_NAME = "invetflow-agent"
DEFAULT_STT_PROVIDER = "openai"
DEFAULT_STT_MODEL = "gpt-4o-transcribe"
DEFAULT_STT_REALTIME = True
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TTS_PROVIDER = "openai"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid float for %s=%r; using %.2f", name, raw, default)
        return default


def _agent_name() -> str:
    return os.environ.get("LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME).strip() or DEFAULT_AGENT_NAME


def _api_base() -> str:
    return os.environ.get("INVETFLOW_API_URL", "http://127.0.0.1:3001").rstrip("/")


def _require_secret() -> str:
    s = os.environ.get("AGENT_API_SECRET", "").strip()
    if not s:
        raise RuntimeError("AGENT_API_SECRET is required (match invetflow-server)")
    return s


# Full job descriptions + question text can be long; model context is large, prior cap was too tight.
_MAX_JOB_DESCRIPTION_CHARS = 24_000


def _build_instructions(ctx: dict[str, Any]) -> str:
    title = ctx.get("title") or "Interview"
    job_title = ctx.get("job_title") or ""
    job_desc = (ctx.get("job_description") or "").strip()
    if len(job_desc) > _MAX_JOB_DESCRIPTION_CHARS:
        job_desc = job_desc[:_MAX_JOB_DESCRIPTION_CHARS]
    duration = ctx.get("duration_minutes")
    try:
        duration_mins = int(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_mins = None
    questions = ctx.get("questions") or []
    parts: list[str] = [
        "You are a professional interviewer for Invetflow.",
        f"Interview: {title}.",
        f"Role: {job_title}.",
        "Ask one main question at a time, wait for the candidate's full answer before the next main "
        "question. Use follow-ups from the list only when they help clarify or go deeper. "
        "Keep your spoken replies concise and natural.",
    ]
    if duration_mins is not None and duration_mins > 0:
        parts.append(
            f"Target interview length: about {duration_mins} minutes — pace main questions and "
            "follow-ups so you stay roughly within that time."
        )
    if job_desc:
        parts.append("Job description and expectations (use this to interpret answers and choose probes):\n")
        parts.append(job_desc)
    else:
        parts.append(
            "No job description was provided. Infer reasonable expectations from the role title and "
            "interview title, and ask relevant technical and behavioral questions for that role."
        )

    if questions:
        parts.append("Planned questions (in order). Cover them in this order unless a follow-up must finish first:")
        for q in questions:
            order = int(q.get("order", 0)) + 1
            main = (q.get("question") or "").strip()
            cat = (q.get("category") or "").strip()
            line = f"{order}. [{cat}] {main}" if cat else f"{order}. {main}"
            parts.append(line)
            tl = q.get("time_limit_seconds")
            if tl is not None:
                try:
                    sec = int(tl)
                    if sec > 0:
                        parts.append(f"   Suggested time for this question: about {sec} seconds.")
                except (TypeError, ValueError):
                    pass
            fps = q.get("follow_up_prompts") or []
            if fps:
                parts.append(f"   Optional follow-ups: {'; '.join(str(f) for f in fps)}")
    else:
        parts.append(
            "No fixed question list was provided. Design a structured interview that fits the role and "
            "any job context above: mix behavioral and role-relevant technical questions, then "
            "probe with short follow-ups as needed."
        )
    return "\n".join(parts)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid integer for %s=%r; using %d", name, raw, default)
        return default


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in allowed:
        logger.warning("invalid value for %s=%r; using %s", name, raw, default)
        return default
    return value


def _build_stt_prompt(ctx: dict[str, Any]) -> str:
    custom = os.environ.get("OPENAI_STT_PROMPT", "").strip()
    if custom:
        return custom

    title = ctx.get("title") or "Interview"
    job_title = ctx.get("job_title") or ""
    questions = ctx.get("questions") or []
    terms: list[str] = []
    for q in questions[:12]:
        for key in ("category", "question"):
            value = q.get(key)
            if value:
                terms.append(str(value))
        for follow_up in (q.get("follow_up_prompts") or [])[:2]:
            terms.append(str(follow_up))

    joined_terms = " ".join(terms)
    if len(joined_terms) > 1200:
        joined_terms = joined_terms[:1200]

    return (
        "This is a job interview. The candidate is answering technical questions. "
        "Transcribe exactly what is spoken. Preserve technical terms, code identifiers, "
        "acronyms, and numbers precisely. Do not correct grammar. Do not omit filler words. "
        "If audio is unclear, transcribe your best interpretation rather than skipping it. "
        f"Interview: {title}. Role: {job_title}. "
        f"Domain vocabulary: {joined_terms}"
    ).strip()


def _build_openai_stt(ctx: dict[str, Any]) -> tuple[Any, bool]:
    detect_language = _env_bool("OPENAI_STT_DETECT_LANGUAGE", False)
    language = os.environ.get("OPENAI_STT_LANGUAGE", "en").strip() or "en"
    model = os.environ.get("OPENAI_STT_MODEL", DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL
    use_realtime = _env_bool("OPENAI_STT_REALTIME", DEFAULT_STT_REALTIME)

    turn_detection = {
        "type": "server_vad",
        "threshold": _env_float("OPENAI_STT_VAD_THRESHOLD", 0.45),
        "prefix_padding_ms": _env_int("OPENAI_STT_PREFIX_PADDING_MS", 500),
        # Interview answers contain thinking pauses; 1000 ms keeps OpenAI's server VAD from
        # closing the turn mid-thought.
        "silence_duration_ms": _env_int("OPENAI_STT_SILENCE_DURATION_MS", 1000),
    }

    kwargs: dict[str, Any] = {
        "model": model,
        "detect_language": detect_language,
        "prompt": _build_stt_prompt(ctx),
        "use_realtime": use_realtime,
    }
    if not detect_language:
        kwargs["language"] = language

    if use_realtime:
        kwargs["turn_detection"] = turn_detection
        noise_reduction = _env_choice(
            "OPENAI_STT_NOISE_REDUCTION",
            "near_field",
            {"near_field", "far_field", "off", "none", "false", "0"},
        )
        if noise_reduction not in {"off", "none", "false", "0"}:
            kwargs["noise_reduction_type"] = noise_reduction

    return openai.STT(**kwargs), use_realtime


def _build_deepgram_stt() -> tuple[Any, bool]:
    if deepgram is None:
        raise RuntimeError(
            "STT_PROVIDER=deepgram requires livekit-plugins-deepgram; install requirements.txt"
        )

    detect_language = _env_bool("DEEPGRAM_STT_DETECT_LANGUAGE", False)
    kwargs: dict[str, Any] = {
        "model": os.environ.get("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3",
        "detect_language": detect_language,
        "interim_results": _env_bool("DEEPGRAM_STT_INTERIM_RESULTS", True),
        "smart_format": _env_bool("DEEPGRAM_STT_SMART_FORMAT", True),
        "endpointing_ms": _env_int("DEEPGRAM_STT_ENDPOINTING_MS", 25),
        "filler_words": _env_bool("DEEPGRAM_STT_FILLER_WORDS", True),
    }
    if not detect_language:
        kwargs["language"] = os.environ.get("DEEPGRAM_STT_LANGUAGE", "en-US").strip() or "en-US"

    return deepgram.STT(**kwargs), False


def _build_stt(ctx: dict[str, Any]) -> tuple[Any, bool]:
    provider = _env_choice("STT_PROVIDER", DEFAULT_STT_PROVIDER, {"openai", "deepgram"})
    if provider == "deepgram":
        return _build_deepgram_stt()
    return _build_openai_stt(ctx)


def _build_llm() -> Any:
    return openai.LLM(
        model=os.environ.get("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    )


def _build_tts() -> Any:
    provider = _env_choice("TTS_PROVIDER", DEFAULT_TTS_PROVIDER, {"openai", "cartesia"})
    if provider == "cartesia":
        if cartesia is None:
            raise RuntimeError(
                "TTS_PROVIDER=cartesia requires livekit-plugins-cartesia; install requirements.txt"
            )
        speed_raw = os.environ.get("CARTESIA_TTS_SPEED", "").strip()
        speed: float | None = None
        if speed_raw:
            try:
                speed = float(speed_raw)
            except ValueError:
                logger.warning("invalid float for CARTESIA_TTS_SPEED=%r; using default", speed_raw)
        return cartesia.TTS(
            model=os.environ.get("CARTESIA_TTS_MODEL", "sonic-3").strip() or "sonic-3",
            voice=(
                os.environ.get("CARTESIA_TTS_VOICE", "f786b574-daa5-4673-aa0c-cbe3e8534c02").strip()
                or "f786b574-daa5-4673-aa0c-cbe3e8534c02"
            ),
            language=os.environ.get("CARTESIA_TTS_LANGUAGE", "en").strip() or "en",
            speed=speed,
        )

    return openai.TTS(
        model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
        voice=os.environ.get("OPENAI_TTS_VOICE", "ash").strip() or "ash",
    )


def _build_vad() -> Any:
    return silero.VAD.load(
        min_speech_duration=_env_float("SILERO_MIN_SPEECH_DURATION", 0.10),
        # Interview answers include thinking pauses; 0.90 s also matches OpenAI server VAD's
        # 1000 ms so Silero never closes the turn before OpenAI does when both are running.
        min_silence_duration=_env_float("SILERO_MIN_SILENCE_DURATION", 0.90),
        prefix_padding_duration=_env_float("SILERO_PREFIX_PADDING_DURATION", 0.50),
        activation_threshold=_env_float("SILERO_ACTIVATION_THRESHOLD", 0.45),
        deactivation_threshold=_env_float("SILERO_DEACTIVATION_THRESHOLD", 0.25),
    )


def _build_room_options() -> room_io.RoomOptions:
    # Stacking LiveKit NC on top of the browser capture filters and OpenAI's near-field
    # noise reduction smears speech and degrades STT accuracy. Default to off so OpenAI's
    # own model-tuned reduction is the single noise stage; opt back in only when the room
    # is genuinely noisy (set LIVEKIT_AGENT_NOISE_CANCELLATION=nc|bvc|telephony).
    mode = os.environ.get("LIVEKIT_AGENT_NOISE_CANCELLATION", "off").strip().lower()
    nc_options: Any = None

    if mode not in {"", "off", "none", "false", "0"}:
        if noise_cancellation is None:
            logger.warning(
                "LIVEKIT_AGENT_NOISE_CANCELLATION=%s but livekit-plugins-noise-cancellation "
                "is not installed; inbound audio will not use enhanced cancellation",
                mode,
            )
        elif mode == "bvc":
            nc_options = noise_cancellation.BVC()
        elif mode == "telephony":
            nc_options = noise_cancellation.BVCTelephony()
        elif mode == "nc":
            nc_options = noise_cancellation.NC()
        else:
            logger.warning(
                "unknown LIVEKIT_AGENT_NOISE_CANCELLATION=%r; expected nc, bvc, telephony, or off",
                mode,
            )

    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(noise_cancellation=nc_options),
    )


async def _fetch_context(session_id: str, secret: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            f"{_api_base()}/api/agent/interviews/{session_id}/context",
            headers={"X-Invetflow-Agent-Secret": secret},
        )
        r.raise_for_status()
        return r.json()


async def _start_candidate_audio_egress(session_id: str, secret: str, track_id: str) -> None:
    """Tell invetflow-server to start LiveKit Track Egress (candidate mic → S3) for post-session STT."""
    if not (track_id or "").strip():
        return
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{_api_base()}/api/agent/interviews/{session_id}/candidate-audio-egress",
                headers={"X-Invetflow-Agent-Secret": secret, "Content-Type": "application/json"},
                json={"trackId": track_id},
            )
        if not r.is_success:
            logger.warning(
                "candidate-audio-egress failed: %s %s", r.status_code, r.text[:500]
            )
    except httpx.HTTPError as e:
        logger.warning("candidate-audio-egress error: %s", e)


def _register_candidate_mic_egress(room: rtc.Room, session_id: str, secret: str) -> None:
    """Start S3 track egress when the remote candidate publishes their microphone (one shot)."""
    state = {"done": False}

    @room.on("track_published")
    def _on_track_published(
        pub: rtc.RemoteTrackPublication, _participant: rtc.RemoteParticipant
    ) -> None:
        if state["done"]:
            return
        if pub.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if pub.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        state["done"] = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            _start_candidate_audio_egress(session_id, secret, pub.sid),
        )


async def _request_batch_transcript_refinement(session_id: str, secret: str) -> None:
    """After the call ends, server polls S3 and runs gpt-4o-transcribe on the merged audio."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{_api_base()}/api/agent/interviews/{session_id}/refine-transcript",
                headers={"X-Invetflow-Agent-Secret": secret},
            )
        if not r.is_success:
            logger.warning("refine-transcript failed: %s %s", r.status_code, r.text[:500])
    except httpx.HTTPError as e:
        logger.warning("refine-transcript error: %s", e)


async def _append_transcript(
    session_id: str,
    secret: str,
    *,
    speaker: str,
    content: str,
    confidence: float | None = None,
) -> None:
    body: dict[str, Any] = {"speaker": speaker, "content": content}
    if confidence is not None:
        body["confidence"] = confidence
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{_api_base()}/api/agent/interviews/{session_id}/transcript",
                headers={"X-Invetflow-Agent-Secret": secret, "Content-Type": "application/json"},
                json=body,
            )
        if not r.is_success:
            logger.warning("append_transcript failed: %s %s", r.status_code, r.text[:500])
    except httpx.HTTPError as e:
        logger.warning("append_transcript error: %s", e)


def _register_transcript_handlers(session: AgentSession, session_id: str, secret: str) -> None:
    def schedule(coro: Any) -> None:
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            logger.exception("transcript handler: no running event loop")

    @session.on("user_input_transcribed")
    def _on_user(ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final or not (ev.transcript or "").strip():
            return
        schedule(
            _append_transcript(
                session_id,
                secret,
                speaker="Candidate",
                content=ev.transcript.strip(),
            )
        )

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage) or item.role != "assistant":
            return
        if item.interrupted:
            return
        text = item.text_content
        if not text or not text.strip():
            return
        schedule(
            _append_transcript(
                session_id,
                secret,
                speaker="AI",
                content=text.strip(),
            )
        )


server = AgentServer()


@server.rtc_session(agent_name=_agent_name())
async def entrypoint(ctx: JobContext) -> None:
    raw = (ctx.job.metadata or "").strip()
    meta: dict[str, Any] = {}
    if raw:
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("job metadata is not valid JSON: %s", raw[:200])

    session_id = (
        meta.get("interviewId")
        or meta.get("sessionId")
        or meta.get("session_id")
        or ""
    ).strip()
    if not session_id:
        raise RuntimeError(
            "Missing interviewId in job metadata. The Invetflow server should dispatch with "
            'JSON: {"interviewId":"...","jobId":"..."}'
        )

    secret = _require_secret()
    ctx_data = await _fetch_context(session_id, secret)
    instructions = _build_instructions(ctx_data)

    await _append_transcript(
        session_id,
        secret,
        speaker="System",
        content="AI interviewer (invetflow-agent) connected.",
    )

    stt_provider = _env_choice("STT_PROVIDER", DEFAULT_STT_PROVIDER, {"openai", "deepgram"})
    tts_provider = _env_choice("TTS_PROVIDER", DEFAULT_TTS_PROVIDER, {"openai", "cartesia"})
    stt_model = (
        os.environ.get("OPENAI_STT_MODEL", DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL
        if stt_provider == "openai"
        else os.environ.get("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3"
    )
    tts_voice = (
        os.environ.get("OPENAI_TTS_VOICE", "ash").strip() or "ash"
        if tts_provider == "openai"
        else (
            os.environ.get("CARTESIA_TTS_VOICE", "f786b574-daa5-4673-aa0c-cbe3e8534c02").strip()
            or "f786b574-daa5-4673-aa0c-cbe3e8534c02"
        )
    )
    logger.info(
        "starting voice session with stt_provider=%s stt_model=%s openai_stt_realtime=%s llm_model=%s "
        "tts_provider=%s tts_voice=%s noise_cancellation=%s",
        stt_provider,
        stt_model,
        _env_bool("OPENAI_STT_REALTIME", DEFAULT_STT_REALTIME) if stt_provider == "openai" else False,
        os.environ.get("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL,
        tts_provider,
        tts_voice,
        os.environ.get("LIVEKIT_AGENT_NOISE_CANCELLATION", "off").strip().lower(),
    )

    stt, stt_uses_server_vad = _build_stt(ctx_data)
    session: AgentSession[None] = AgentSession(
        stt=stt,
        llm=_build_llm(),
        tts=_build_tts(),
        # Avoid dual VAD: OpenAI's realtime STT performs server-side turn detection, and
        # running Silero in parallel can cut a turn before OpenAI does (truncated segments).
        # For non-realtime OpenAI STT and Deepgram STT, keep Silero enabled for turn endpointing.
        vad=None if stt_uses_server_vad else _build_vad(),
        # Interview pauses: 1.0 s minimum and 3.5 s maximum endpointing tolerate candidates
        # thinking mid-answer without leaving long dead air at the upper bound.
        min_endpointing_delay=_env_float("AGENT_MIN_ENDPOINTING_DELAY", 1.0),
        max_endpointing_delay=_env_float("AGENT_MAX_ENDPOINTING_DELAY", 3.5),
        min_interruption_duration=_env_float("AGENT_MIN_INTERRUPTION_DURATION", 0.75),
        min_interruption_words=int(_env_float("AGENT_MIN_INTERRUPTION_WORDS", 2)),
    )
    _register_transcript_handlers(session, session_id, secret)
    _register_candidate_mic_egress(ctx.room, session_id, secret)

    async def _on_job_shutdown(_reason: str) -> None:
        await _request_batch_transcript_refinement(session_id, secret)

    ctx.add_shutdown_callback(_on_job_shutdown)

    await session.start(
        room=ctx.room,
        agent=Agent(instructions=instructions),
        room_options=_build_room_options(),
    )
    await session.generate_reply(
        instructions=(
            "Greet the candidate briefly, then ask the first question from the interview list."
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
