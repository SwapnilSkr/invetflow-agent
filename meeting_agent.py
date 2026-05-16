"""Invetflow human-meeting transcription worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


async def _post_turn(
    client: httpx.AsyncClient,
    session_id: str,
    secret: str,
    payload: dict[str, Any],
) -> None:
    try:
        response = await client.post(
            f"{_api_base()}/api/agent/meetings/{session_id}/transcript",
            headers={"X-Invetflow-Agent-Secret": secret},
            json=payload,
        )
        if not response.is_success:
            logger.warning(
                "meeting transcript POST failed: %s %s",
                response.status_code,
                response.text[:500],
            )
    except httpx.HTTPError as exc:
        logger.warning("meeting transcript POST error: %s", exc)


async def _publish_turn(room: rtc.Room, payload: dict[str, Any]) -> None:
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
    stt = deepgram.STT(
        model=os.environ.get("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3",
        interim_results=True,
        smart_format=True,
        punctuate=True,
        endpointing_ms=300,
    )
    audio_stream = rtc.AudioStream(audio_track)
    stt_stream = stt.stream()

    async def push_audio() -> None:
        async for frame in audio_stream:
            stt_stream.push_frame(frame)

    push_task = asyncio.create_task(push_audio())
    started_at = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            async for event in stt_stream:
                text = _event_text(event)
                if not text:
                    continue
                now_ms = int((time.monotonic() - started_at) * 1000)
                payload = {
                    "type": "transcript_turn",
                    "speaker_identity": participant.identity,
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
        finally:
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
