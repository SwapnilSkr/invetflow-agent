"""Invetflow human-meeting transcription worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli
from livekit.plugins import deepgram

load_dotenv()

logger = logging.getLogger("invetflow-meeting-agent")

TRANSCRIPT_TOPIC = "invetflow-meeting-transcript"
DEFAULT_AGENT_NAME = "invetflow-meeting-agent"

# Retry / resilience tuning constants
HTTP_MAX_ATTEMPTS = 3
HTTP_BACKOFF_SECONDS = [0.5, 1.5, 3.0]
STT_MAX_CONSECUTIVE_FAILURES = 5
STT_BACKOFF_SECONDS = [1.0, 2.0, 4.0, 8.0, 8.0]
STT_RECOVERY_WINDOW_SECONDS = 30.0


def _api_base() -> str:
    return (
        os.environ.get("INVETFLOW_API_BASE")
        or os.environ.get("INVETFLOW_API_URL")
        or "http://127.0.0.1:3001"
    ).rstrip("/")


def _agent_secret() -> str:
    secret = os.environ.get("AGENT_API_SECRET", "").strip()
    if not secret:
        raise RuntimeError("AGENT_API_SECRET is required")
    return secret


def _agent_name() -> str:
    return os.environ.get("MEETING_AGENT_NAME", DEFAULT_AGENT_NAME).strip() or DEFAULT_AGENT_NAME


def _speaker_name(participant: rtc.RemoteParticipant) -> str:
    return participant.name or participant.identity or "Participant"


def _event_text(event: Any) -> str:
    alternatives = getattr(event, "alternatives", None) or []
    if not alternatives:
        return ""
    return (getattr(alternatives[0], "text", "") or "").strip()


def _event_confidence(event: Any) -> float:
    alternatives = getattr(event, "alternatives", None) or []
    if not alternatives:
        return 0.0
    try:
        return float(getattr(alternatives[0], "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_final(event: Any) -> bool:
    event_type = getattr(event, "type", None)
    return "FINAL" in str(event_type).upper()


def _jitter(base: float) -> float:
    """Apply uniform jitter multiplier to a base delay."""
    return base * random.uniform(0.5, 1.5)


async def _post_turn(
    client: httpx.AsyncClient,
    session_id: str,
    secret: str,
    payload: dict[str, Any],
) -> None:
    """POST a transcript turn with retry on transport/timeout/5xx errors."""
    last_exc: Exception | None = None
    for attempt in range(HTTP_MAX_ATTEMPTS):
        try:
            response = await client.post(
                f"{_api_base()}/api/agent/meetings/{session_id}/transcript",
                headers={"X-Invetflow-Agent-Secret": secret},
                json=payload,
            )
            if response.is_success:
                return
            if response.status_code >= 500:
                logger.warning(
                    "meeting transcript POST server error (attempt %d/%d): %s %s",
                    attempt + 1,
                    HTTP_MAX_ATTEMPTS,
                    response.status_code,
                    response.text[:500],
                    extra={"session_id": session_id, "attempt": attempt + 1},
                )
                last_exc = None  # will retry
            else:
                # 4xx — non-recoverable
                logger.error(
                    "meeting transcript POST non-recoverable error: %s %s — speaker=%s text=%r",
                    response.status_code,
                    response.text[:500],
                    payload.get("speaker_identity"),
                    payload.get("text", "")[:80],
                    extra={"session_id": session_id, "attempt": attempt + 1},
                )
                return
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            logger.warning(
                "meeting transcript POST transport error (attempt %d/%d): %s",
                attempt + 1,
                HTTP_MAX_ATTEMPTS,
                exc,
                extra={"session_id": session_id, "attempt": attempt + 1},
            )
            last_exc = exc

        # Backoff before next attempt (no sleep after last attempt)
        if attempt < HTTP_MAX_ATTEMPTS - 1:
            delay = _jitter(HTTP_BACKOFF_SECONDS[attempt])
            await asyncio.sleep(delay)

    logger.error(
        "meeting transcript POST failed after %d attempts — speaker=%s text=%r exc=%s",
        HTTP_MAX_ATTEMPTS,
        payload.get("speaker_identity"),
        payload.get("text", "")[:80],
        last_exc,
        extra={"session_id": session_id, "attempt": HTTP_MAX_ATTEMPTS},
    )


async def _publish_turn(room: rtc.Room, payload: dict[str, Any]) -> None:
    await room.local_participant.publish_data(
        json.dumps(payload).encode("utf-8"),
        reliable=True,
        topic=TRANSCRIPT_TOPIC,
    )


async def _publish_status(
    room: rtc.Room,
    participant: rtc.RemoteParticipant,
    status: str,
) -> None:
    """Publish a transcript_status control message over the DataChannel."""
    payload: dict[str, Any] = {
        "type": "transcript_status",
        "speaker_identity": participant.identity,
        "speaker_name": participant.name or participant.identity,
        "status": status,
    }
    await room.local_participant.publish_data(
        json.dumps(payload).encode("utf-8"),
        reliable=True,
        topic=TRANSCRIPT_TOPIC,
    )


async def _transcribe_track(
    room: rtc.Room,
    session_id: str,
    secret: str,
    audio_track: rtc.RemoteAudioTrack,
    participant: rtc.RemoteParticipant,
) -> None:
    identity = participant.identity
    stt = deepgram.STT(
        model=os.environ.get("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3",
        interim_results=True,
        smart_format=True,
        punctuate=True,
        endpointing_ms=300,
    )
    audio_stream = rtc.AudioStream(audio_track)

    consecutive_failures = 0
    started_at = time.monotonic()

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            stt_stream = stt.stream()
            push_task: asyncio.Task[None] | None = None

            async def push_audio() -> None:
                async for frame in audio_stream:
                    stt_stream.push_frame(frame)

            push_task = asyncio.create_task(push_audio())
            stream_start = time.monotonic()
            resumed_published = False

            try:
                async for event in stt_stream:
                    # After a successful reconnect publish "resumed" once
                    if consecutive_failures > 0 and not resumed_published:
                        await _publish_status(room, participant, "resumed")
                        logger.info(
                            "STT stream recovered for participant %s",
                            identity,
                            extra={"participant": identity, "session_id": session_id, "attempt": consecutive_failures},
                        )
                        resumed_published = True

                    # Reset failure counter if stream has been healthy for the recovery window
                    if consecutive_failures > 0 and (time.monotonic() - stream_start) >= STT_RECOVERY_WINDOW_SECONDS:
                        logger.info(
                            "STT stream stable for %.0fs — resetting failure counter for %s",
                            STT_RECOVERY_WINDOW_SECONDS,
                            identity,
                            extra={"participant": identity, "session_id": session_id, "attempt": 0},
                        )
                        consecutive_failures = 0

                    text = _event_text(event)
                    if not text:
                        continue
                    now_ms = int((time.monotonic() - started_at) * 1000)
                    payload: dict[str, Any] = {
                        "type": "transcript_turn",
                        "speaker_identity": identity,
                        "speaker_name": _speaker_name(participant),
                        "text": text,
                        "start_ms": now_ms,
                        "end_ms": now_ms,
                        "confidence": _event_confidence(event),
                        "is_final": _is_final(event),
                    }
                    await _publish_turn(room, payload)
                    if payload["is_final"]:
                        await _post_turn(client, session_id, secret, payload)

                # stt_stream exhausted cleanly — exit the while loop
                break

            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                logger.warning(
                    "STT stream error for participant %s (failure %d/%d): %s: %s",
                    identity,
                    consecutive_failures,
                    STT_MAX_CONSECUTIVE_FAILURES,
                    type(exc).__name__,
                    exc,
                    extra={"participant": identity, "session_id": session_id, "attempt": consecutive_failures},
                )
                await _publish_status(room, participant, "paused")

                if consecutive_failures >= STT_MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "STT stream gave up after %d consecutive failures for participant %s — disabling transcription",
                        STT_MAX_CONSECUTIVE_FAILURES,
                        identity,
                        extra={"participant": identity, "session_id": session_id, "attempt": consecutive_failures},
                    )
                    await _publish_status(room, participant, "disabled")
                    break

                backoff_index = min(consecutive_failures - 1, len(STT_BACKOFF_SECONDS) - 1)
                delay = _jitter(STT_BACKOFF_SECONDS[backoff_index])
                logger.info(
                    "STT retry for %s in %.2fs (attempt %d)",
                    identity,
                    delay,
                    consecutive_failures,
                    extra={"participant": identity, "session_id": session_id, "attempt": consecutive_failures},
                )
                await asyncio.sleep(delay)

            finally:
                if push_task is not None:
                    push_task.cancel()
                    await asyncio.gather(push_task, return_exceptions=True)


server = AgentServer()


@server.rtc_session(agent_name=_agent_name())
async def entrypoint(ctx: JobContext) -> None:
    raw = (ctx.job.metadata or "").strip()
    metadata: dict[str, Any] = json.loads(raw) if raw else {}
    session_id = str(metadata.get("session_id") or metadata.get("sessionId") or "").strip()
    if not session_id:
        raise RuntimeError("meeting agent metadata must include session_id")
    if metadata.get("transcription_enabled") is False:
        logger.info("transcription disabled for session %s", session_id)
        return

    secret = _agent_secret()
    await ctx.connect()

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity.startswith("agent-"):
            return
        if not isinstance(track, rtc.RemoteAudioTrack):
            return
        asyncio.create_task(
            _transcribe_track(ctx.room, session_id, secret, track, participant),
        )

    await ctx.wait_for_disconnect()


if __name__ == "__main__":
    cli.run_app(server)
