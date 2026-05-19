from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

from livekit import rtc
from PIL import Image

logger = logging.getLogger("invetflow-agent.vision")


def _dct_1d(vec: list[float]) -> list[float]:
    """1-D DCT-II (naive O(n²)) — fine for n=32."""
    import math

    n = len(vec)
    out = [0.0] * n
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += vec[i] * math.cos(math.pi / n * (i + 0.5) * k)
        scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        out[k] = s * scale
    return out


def _phash(img: Image.Image, hash_size: int = 16) -> str:
    """Compute a perceptual hash (simplified DCT-based) of the image.

    Resize → 32×32 grayscale → DCT → keep top-left 16×16 low-freq coeffs →
    median threshold → hex string.
    """
    gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())

    # 2-D DCT via separable 1-D DCT on rows then columns
    rows = [pixels[i * 32 : (i + 1) * 32] for i in range(32)]
    dct_rows = [_dct_1d([float(v) for v in row]) for row in rows]
    cols = [[dct_rows[r][c] for r in range(32)] for c in range(32)]
    dct_cols = [_dct_1d(col) for col in cols]

    # Keep top-left hash_size × hash_size low-frequency coefficients
    low_freq = []
    for c in range(hash_size):
        for r in range(hash_size):
            low_freq.append(dct_cols[c][r])

    # Skip the DC coefficient (index 0) for robustness to brightness changes
    avg = sum(low_freq[1:]) / (len(low_freq) - 1)
    bits = "".join("1" if v > avg else "0" for v in low_freq[1:])

    # Pad to full hash_size*hash_size bits (minus 1 for DC)
    while len(bits) < hash_size * hash_size - 1:
        bits += "0"

    # Convert to hex
    hex_str = hex(int(bits, 2))[2:]
    expected_len = (hash_size * hash_size - 1) // 4
    return hex_str.zfill(expected_len)


def _hamming_distance(a: str, b: str) -> int:
    """Bit-level Hamming distance between two hex strings."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a.zfill(max_len)
        b = b.zfill(max_len)
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class ScreenWatcher:
    """Keeps the freshest frame from the candidate's screen-share track.

    Also computes a perceptual hash on new frames and tracks whether the
    screen has "materially changed" since the last frame that was sent to the
    LLM.  pHash is throttled to at most once per second to avoid blocking the
    frame consumer or concurrent snapshot callers.
    """

    def __init__(self) -> None:
        self._latest: rtc.VideoFrame | None = None
        self._captured_at: float = 0.0
        self._lock = asyncio.Lock()
        self._frame_ready = asyncio.Event()
        self._consumer_task: asyncio.Task[None] | None = None
        # Phase 2: change-gated auto-attach
        self._last_sent_hash: str | None = None
        self._materially_changed: bool = False
        self._last_auto_attach_at: float = 0.0
        self._last_phash_at: float = 0.0

    def attach(self, track: Any) -> None:
        self.detach()
        self._frame_ready.clear()

        stream = rtc.VideoStream(track)

        async def consume() -> None:
            try:
                async for event in stream:
                    async with self._lock:
                        self._latest = event.frame
                        self._captured_at = time.monotonic()
                        self._frame_ready.set()

                    # Phase 2: compute pHash outside the lock so snapshot_jpeg
                    # never waits on DCT work. Throttle to once per second.
                    now = time.monotonic()
                    if now - self._last_phash_at < 1.0:
                        continue
                    self._last_phash_at = now
                    try:
                        img = _frame_to_pil(event.frame)
                        h = _phash(img)
                        async with self._lock:
                            if self._last_sent_hash is not None:
                                if _hamming_distance(h, self._last_sent_hash) > 12:
                                    self._materially_changed = True
                            else:
                                self._materially_changed = True
                    except Exception:
                        logger.debug("pHash computation failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("screen-share frame consumer failed")
            finally:
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("screen-share stream close failed", exc_info=True)

        self._consumer_task = asyncio.create_task(consume())

    def detach(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            self._consumer_task = None
        self._latest = None
        self._captured_at = 0.0
        self._frame_ready.clear()
        self._last_sent_hash = None
        self._materially_changed = False

    @property
    def has_frame(self) -> bool:
        return self._latest is not None and (time.monotonic() - self._captured_at) < 30.0

    @property
    def materially_changed(self) -> bool:
        return self._materially_changed

    def mark_sent(self) -> None:
        """Call after a frame is sent to the LLM to reset change state."""
        self._materially_changed = False
        try:
            if self._latest is not None:
                img = _frame_to_pil(self._latest)
                self._last_sent_hash = _phash(img)
        except Exception:
            logger.debug("mark_sent pHash failed", exc_info=True)
        self._last_auto_attach_at = time.monotonic()

    def can_auto_attach(self, cooldown_seconds: float = 8.0) -> bool:
        """True if the screen has materially changed and cooldown has elapsed."""
        if not self.has_frame:
            return False
        if not self._materially_changed:
            return False
        return (time.monotonic() - self._last_auto_attach_at) >= cooldown_seconds

    async def wait_for_frame(self, timeout: float = 1.0) -> bool:
        if self.has_frame:
            return True
        try:
            await asyncio.wait_for(self._frame_ready.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return self.has_frame

    async def snapshot_jpeg(
        self, *, max_width: int = 1024, quality: int = 70, detail: str = "low"
    ) -> bytes | None:
        async with self._lock:
            frame = self._latest
            if frame is None:
                return None
            image = _frame_to_pil(frame)

        # Phase 3: high detail uses full 1024px, low uses 512px
        effective_max = 1024 if detail == "high" else 512
        if image.width > effective_max:
            ratio = effective_max / image.width
            image = image.resize(
                (effective_max, max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def _frame_to_pil(frame: rtc.VideoFrame) -> Image.Image:
    rgba = frame.convert(rtc.VideoBufferType.RGBA)
    return Image.frombytes("RGBA", (rgba.width, rgba.height), bytes(rgba.data)).convert("RGB")
