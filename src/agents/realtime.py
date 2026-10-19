"""实时语音链路：RTC → ASR → Orchestrator → RAG/Agent → LLM → TTS。

语音 Agent 的难点不在「怎么调模型」，而在**链路与事件**：
一段话进来要走 采集 → 识别 → 路由 → 检索/工具 → 生成 → 合成 → 播放 七个环节，
每一环都可能成为首字延迟的瓶颈。所以这里先把**事件流**打穿：
每个环节都吐一个带序号的事件，上层可以据此算「哪一环慢」，而不是只知道「整体慢」。

实现策略是**端口 + Mock**（诚实边界，见 PROJECT.md 第 8 章）：
- `RTCSession` / `ASRPort` / `TTSPort` 三个 Protocol 定义链路契约；
- Mock 实现把「语音」退化成可离线复现的帧（payload 是文本字节），
  这样无需媒体服务器也能把**事件流与编排接线**验证到底；
- 换成真实供应商（如 RTC SDK + ASR/TTS 云服务）时，只替换端口实现，编排层零改动。

不宣称的部分：不做回声消除、不做 VAD 断句、不做真流式增量识别 —— 这些属于媒体层，
不在本面试项目的范围内。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.infra.core.logging import get_logger
from src.infra.tools.base import ToolContext

from .orchestrator import Orchestrator

logger = get_logger("agents.realtime")


@dataclass(slots=True)
class AudioFrame:
    seq: int
    payload: bytes

    def text(self) -> str:
        return self.payload.decode("utf-8", errors="ignore")


@dataclass(slots=True)
class RealtimeEvent:
    seq: int
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.type, **self.payload}


@dataclass(slots=True)
class RealtimeTurn:
    events: list[RealtimeEvent]
    transcript: str | None
    answer: str | None
    audio_frames: int
    route: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "answer": self.answer,
            "audio_frames": self.audio_frames,
            "route": self.route,
            "events": [event.to_payload() for event in self.events],
        }


# --------------------------------------------------------------------------- #
# 端口（链路契约）
# --------------------------------------------------------------------------- #
class RTCSession(Protocol):
    async def capture(self) -> list[AudioFrame]: ...
    async def play(self, frames: list[AudioFrame]) -> None: ...


class ASRPort(Protocol):
    async def transcribe(self, frames: list[AudioFrame]) -> str: ...


class TTSPort(Protocol):
    async def synthesize(self, text: str) -> list[AudioFrame]: ...


# --------------------------------------------------------------------------- #
# Mock 实现：把语音退化成可离线复现的帧
# --------------------------------------------------------------------------- #
class MockRTCSession:
    """把「客户端说的一句话」按帧下发；播放端只记录收到多少帧。"""

    def __init__(self, *, utterances: list[str] | None = None, frame_size: int = 12) -> None:
        self._pending = list(utterances or [])
        self._frame_size = frame_size
        self.played: int = 0

    def push(self, utterance: str) -> None:
        self._pending.append(utterance)

    async def capture(self) -> list[AudioFrame]:
        if not self._pending:
            return []
        utterance = self._pending.pop(0).encode("utf-8")
        return [
            AudioFrame(seq=index, payload=utterance[index : index + self._frame_size])
            for index in range(0, len(utterance), self._frame_size)
        ]

    async def play(self, frames: list[AudioFrame]) -> None:
        self.played += len(frames)


class MockASR:
    """识别 = 把帧解回文本（模拟供应商返回单段终稿）。"""

    async def transcribe(self, frames: list[AudioFrame]) -> str:
        return "".join(frame.text() for frame in frames)


class MockTTS:
    """合成 = 按固定块长切分文本帧。"""

    def __init__(self, *, chunk_size: int = 10) -> None:
        self._chunk_size = chunk_size

    async def synthesize(self, text: str) -> list[AudioFrame]:
        raw = (text or "").encode("utf-8")
        return [
            AudioFrame(seq=index, payload=raw[index : index + self._chunk_size])
            for index in range(0, len(raw), self._chunk_size)
        ]


# --------------------------------------------------------------------------- #
# 链路编排
# --------------------------------------------------------------------------- #
class RealtimePipeline:
    def __init__(
        self,
        *,
        rtc: RTCSession,
        asr: ASRPort,
        tts: TTSPort,
        orchestrator: Orchestrator,
    ) -> None:
        self._rtc = rtc
        self._asr = asr
        self._tts = tts
        self._orchestrator = orchestrator
        self._seq = 0

    def _emit(self, events: list[RealtimeEvent], kind: str, **payload: Any) -> None:
        self._seq += 1
        events.append(RealtimeEvent(seq=self._seq, type=kind, payload=payload))

    async def handle_turn(self, ctx: ToolContext, *, text: str | None = None) -> RealtimeTurn:
        events: list[RealtimeEvent] = []

        if text is not None:
            transcript = text
            self._emit(events, "asr.skipped", reason="text_input")
        else:
            frames = await self._rtc.capture()
            self._emit(events, "rtc.capture", frames=len(frames))
            transcript = await self._asr.transcribe(frames)
            if not transcript:
                self._emit(events, "turn.end", status="empty_transcript")
                return RealtimeTurn(events=events, transcript=None, answer=None, audio_frames=0)
            self._emit(events, "asr.final", chars=len(transcript))

        result = await self._orchestrator.handle(transcript, ctx)
        self._emit(
            events,
            "orchestrator.done",
            route=(result.route or {}).get("kind"),
            mode=result.mode,
            success=result.success,
        )

        audio = await self._tts.synthesize(result.answer or "")
        await self._rtc.play(audio)
        self._emit(events, "tts.chunk", frames=len(audio))
        self._emit(events, "turn.end", status="ok" if result.success else "error")

        logger.info(
            "agents.realtime.turn",
            extra={
                "event": "realtime",
                "status": "ok" if result.success else "error",
                "duration_ms": round(result.duration_ms, 2),
            },
        )
        return RealtimeTurn(
            events=events,
            transcript=transcript,
            answer=result.answer,
            audio_frames=len(audio),
            route=result.route,
        )


def build_realtime_ports() -> tuple[MockRTCSession, MockASR, MockTTS]:
    return MockRTCSession(), MockASR(), MockTTS()


__all__ = [
    "AudioFrame",
    "RealtimeEvent",
    "RealtimeTurn",
    "RTCSession",
    "ASRPort",
    "TTSPort",
    "MockRTCSession",
    "MockASR",
    "MockTTS",
    "RealtimePipeline",
    "build_realtime_ports",
]
