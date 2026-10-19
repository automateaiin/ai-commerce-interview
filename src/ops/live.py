"""直播智能处理：分段转录 → 重叠去重 → 分批分析 → 切片候选 → FFmpeg 命令构造。

一条长直播为什么不能「整场丢给模型」（面试必答）：
1) **上下文装不下**：2 小时口播远超任何单次调用的上下文窗口；
2) **成本与延迟不可控**：一次调用要吐完整场总结，token 量与等待时间都爆炸；
3) **质量反而更差**：长上下文里关键卖点被稀释，模型抓不住「哪 30 秒值得切片」。

所以拆成：**按时间窗分段转录 → 分批分析 → 从每批里挑出精彩片段**。
分段之间刻意留 `overlap_s` 重叠：不重叠的话，切点正好落在句子中间就会丢字、断句，
导致转录文本在段边界处出现不可读的残句 —— 重叠是修这个的。

重叠带来的副作用是**重复文本**，因此去重是必需的一步（不是可选优化）：
- 主机制是「裁掉上一段尾部与下一段头部重合的部分」（精确匹配，因为重叠区就是同一段音频）；
- 兜底再看整段近重复（字符 bigram Jaccard），防止某个窗口被完全包含。

FFmpeg 的诚实边界（务必主动说明）：
本仓库**只构造命令字符串**，不调用 FFmpeg 二进制、不产出真实视频文件。
缺少二进制时返回 `degraded=true` + `reason=ffmpeg_not_installed`，绝不假装切片已经产出。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.dataplatform.dedupe import jaccard
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage

from .multimodal import AsrPort, OfflineAsr
from .schemas import BatchAnalysis, FfmpegCommand, LiveClip, LiveSegment

logger = get_logger("ops.live")


class _BatchDraft(BaseModel):
    """模型只负责「这批判读出了什么」；批号与段号由程序填，不让模型编索引。"""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=800)
    selling_points: list[str] = Field(default_factory=list, max_length=8)
    highlight_index: int | None = None


def _windows(duration_s: float, window_s: int, overlap_s: int) -> list[tuple[float, float]]:
    """按时间窗 + 重叠切分直播。最后一个窗口不超过直播总时长。"""
    window = max(1.0, float(window_s))
    overlap = max(0.0, min(float(overlap_s), window - 1.0))
    step = window - overlap
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(start + window, duration_s)
        windows.append((start, end))
        if end >= duration_s:
            break
        start += step
    return windows or [(0.0, duration_s)]


def _overlap_len(prev_text: str, cur_text: str, *, min_len: int = 4) -> int:
    """求「上一段尾部 == 下一段头部」的最长长度（重叠正好落在此处）。

    这是**精确匹配**而非相似度：重叠区本来就是同一段音频的转写，字面必须一致。
    """
    limit = min(len(prev_text), len(cur_text))
    for length in range(limit, min_len - 1, -1):
        if prev_text.endswith(cur_text[:length]):
            return length
    return 0


def dedupe_overlaps(
    segments: Sequence[LiveSegment], *, overlap_s: int = 0, threshold: float = 0.85
) -> tuple[list[LiveSegment], dict[str, Any]]:
    """裁掉相邻段的重叠文本；若整段近重复则丢弃（合并时间区间到上一段）。

    两级机制，职责不重叠：
    1) **精确重叠裁剪**（主）：上一段尾部与下一段头部字面重合的部分直接切掉 ——
       这是「含 overlap 的转录」必然产生的重复，不是噪声；
    2) **整段近重复丢弃**（兜底）：字符 bigram Jaccard 超过阈值说明这一窗被上一窗
       完全覆盖了。阈值取 0.85 而非 0.4 一线：直播口播本身就有大量高频口头语，
       阈值放低会把「内容相似」误判成「重复」而丢真内容 —— 宁可少删，不可错删。
    """
    kept: list[LiveSegment] = []
    trimmed_chars = 0
    dropped: list[int] = []
    for segment in segments:
        if not kept:
            kept.append(segment)
            continue
        prev = kept[-1]
        if jaccard(prev.text, segment.text) >= threshold:
            kept[-1] = prev.model_copy(update={"end_s": max(prev.end_s, segment.end_s)})
            dropped.append(segment.index)
            continue
        overlap = _overlap_len(prev.text, segment.text)
        if overlap:
            kept.append(segment.model_copy(update={"text": segment.text[overlap:]}))
            trimmed_chars += overlap
        else:
            kept.append(segment)
    kept = [item.model_copy(update={"index": index}) for index, item in enumerate(kept)]
    report = {
        "input_segments": len(segments),
        "output_segments": len(kept),
        "dropped_segments": dropped,
        "trimmed_chars": trimmed_chars,
        "overlap_s": overlap_s,
    }
    logger.info(
        "ops.live.dedupe",
        extra={
            "event": "ops_live",
            "status": "ok",
            "reason": f"trimmed={trimmed_chars},dropped={len(dropped)}",
            "changed_keys": len(segments),
        },
    )
    return kept, report


@dataclass(slots=True)
class LiveProcessingResult:
    segments: list[LiveSegment]
    analyses: list[BatchAnalysis]
    clips: list[LiveClip]
    ffmpeg: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    dedupe: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "segments": [item.to_payload() for item in self.segments],
            "analyses": [item.to_payload() for item in self.analyses],
            "clips": [item.to_payload() for item in self.clips],
            "ffmpeg": self.ffmpeg,
            "usage": self.usage,
            "dedupe": self.dedupe,
        }


class LiveProcessor:
    """直播处理器：端口（ASR）+ 网关（分批分析）+ FFmpeg 命令构造。"""

    def __init__(
        self,
        *,
        gateway: Any,
        asr: AsrPort | None = None,
        window_s: int = 90,
        overlap_s: int = 15,
        batch_size: int = 3,
        analysis_model: str = "chat-pro",
        clock: Any = None,
    ) -> None:
        self._gateway = gateway
        self._asr = asr or OfflineAsr()
        self._window_s = max(1, int(window_s))
        self._overlap_s = max(0, int(overlap_s))
        self._batch_size = max(1, int(batch_size))
        self._model = analysis_model

    # -- 1) 分段转录 --------------------------------------------------------- #
    async def transcribe(self, source: dict[str, Any], *, duration_s: float) -> list[LiveSegment]:
        segments: list[LiveSegment] = []
        windows = _windows(duration_s, self._window_s, self._overlap_s)
        async with stage("ops.live.transcribe", windows=len(windows)):
            for index, (start, end) in enumerate(windows):
                text = await self._asr.transcribe(source, window=(start, end))
                segments.append(
                    LiveSegment(
                        index=index,
                        start_s=start,
                        end_s=end,
                        text=text,
                        source=getattr(self._asr, "name", "asr"),
                    )
                )
        return segments

    # -- 2) 重叠去重 --------------------------------------------------------- #
    def dedupe_overlaps(self, segments: Sequence[LiveSegment], *, threshold: float = 0.85) -> tuple[list[LiveSegment], dict[str, Any]]:
        """裁掉相邻段的重叠文本；若整段近重复则丢弃（合并时间区间到上一段）。"""
        return dedupe_overlaps(segments, overlap_s=self._overlap_s, threshold=threshold)

    # -- 3) 分批分析 --------------------------------------------------------- #
    async def analyze(self, segments: Sequence[LiveSegment]) -> tuple[list[BatchAnalysis], dict[str, Any]]:
        """按 `batch_size` 分批，每批一次模型调用；不合法则确定性兜底并标记 degraded。"""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        analyses: list[BatchAnalysis] = []
        batches = [list(segments[i : i + self._batch_size]) for i in range(0, len(segments), self._batch_size)]
        async with stage("ops.live.analyze", batches=len(batches)):
            for batch_index, batch in enumerate(batches):
                draft, degraded, batch_usage = await self._analyze_batch(batch_index, batch)
                usage["prompt_tokens"] += batch_usage["prompt_tokens"]
                usage["completion_tokens"] += batch_usage["completion_tokens"]
                usage["cost_usd"] += batch_usage["cost_usd"]
                highlight = draft.highlight_index
                if highlight is not None and not (0 <= highlight < len(batch)):
                    highlight = None  # 模型给的索引越界 → 丢弃，不让它污染切片
                analyses.append(
                    BatchAnalysis(
                        batch_index=batch_index,
                        segment_indexes=[item.index for item in batch],
                        summary=draft.summary,
                        selling_points=list(draft.selling_points),
                        highlight_index=highlight,
                        logical_model=self._model,
                    )
                )
                if degraded:
                    logger.warning(
                        "ops.live.analyze.degraded",
                        extra={"event": "ops_live", "status": "degraded", "degraded": True,
                               "reason": "structured_output_invalid", "changed_keys": batch_index},
                    )
        return analyses, usage

    async def _analyze_batch(self, batch_index: int, batch: Sequence[LiveSegment]) -> tuple[_BatchDraft, bool, dict[str, Any]]:
        """一次批分析。模型输出非 JSON 时走确定性兜底（离线默认即走这条路径）。"""
        transcript = "\n".join(f"[{item.index}] {item.text}" for item in batch)
        messages = [
            {
                "role": "system",
                "content": "你是直播运营分析助手。只返回一个 JSON 对象，字段：summary(str)、selling_points(str[])、highlight_index(int|null)。不要 Markdown 包裹。",
            },
            {"role": "user", "content": f"以下是直播转录分段（方括号内为段号）：\n{transcript}\n\n请总结本批卖点，并指出最值得切片的段号。"},
        ]
        response = await self._gateway.complete(messages, logical_model=self._model, temperature=0.2)
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "cost_usd": response.usage.cost_usd,
        }
        try:
            draft = _BatchDraft.model_validate_json(_strip_fences(response.text or ""))
            return draft, False, usage
        except Exception as exc:  # noqa: BLE001 - 非 JSON 是常态，必须兜底而不是抛错
            logger.warning(
                "ops.live.analyze.invalid",
                extra={"event": "ops_live", "status": "invalid", "reason": f"{type(exc).__name__}",
                       "changed_keys": batch_index},
            )
            return _fallback_draft(batch), True, usage

    # -- 4) 切片候选 --------------------------------------------------------- #
    def build_clips(self, analyses: Sequence[BatchAnalysis], segments: Sequence[LiveSegment]) -> list[LiveClip]:
        """把每批选出的精彩段变成切片候选（只描述时间区间，不做转码）。"""
        by_index = {item.index: item for item in segments}
        clips: list[LiveClip] = []
        for analysis in analyses:
            if analysis.highlight_index is None:
                continue
            segment_index = analysis.segment_indexes[analysis.highlight_index]
            segment = by_index.get(segment_index)
            if segment is None:
                continue
            score = round(min(1.0, 0.5 + 0.1 * len(analysis.selling_points)), 4)
            clips.append(
                LiveClip(
                    clip_id=f"clip_{analysis.batch_index}_{segment_index}",
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    reason=f"本批卖点密度最高：{'、'.join(analysis.selling_points[:2]) or '口播亮点'}",
                    score=score,
                    text=segment.text,
                )
            )
        return clips

    # -- 5) FFmpeg 命令构造（不执行） --------------------------------------- #
    def build_ffmpeg_commands(
        self, clips: Sequence[LiveClip], *, source_path: str = "live.mp4", output_dir: str = "clips"
    ) -> tuple[list[FfmpegCommand], bool, str | None]:
        available = shutil.which("ffmpeg") is not None
        degraded = not available
        reason = None if available else "ffmpeg_not_installed"
        commands = [
            FfmpegCommand(
                clip_id=clip.clip_id,
                command=(
                    f"ffmpeg -ss {clip.start_s:.2f} -to {clip.end_s:.2f} -i {source_path} "
                    f"-c copy -movflags +faststart {output_dir}/{clip.clip_id}.mp4"
                ),
                output=f"{output_dir}/{clip.clip_id}.mp4",
            )
            for clip in clips
        ]
        if degraded:
            logger.warning(
                "ops.live.ffmpeg.degraded",
                extra={"event": "ops_live", "status": "degraded", "degraded": True,
                       "reason": reason, "changed_keys": len(commands)},
            )
        return commands, degraded, reason

    # -- 编排 --------------------------------------------------------------- #
    async def process(
        self, source: dict[str, Any], *, duration_s: float, source_path: str = "live.mp4"
    ) -> LiveProcessingResult:
        segments = await self.transcribe(source, duration_s=duration_s)
        kept, dedupe_report = self.dedupe_overlaps(segments)
        analyses, usage = await self.analyze(kept)
        clips = self.build_clips(analyses, kept)
        commands, degraded, reason = self.build_ffmpeg_commands(clips, source_path=source_path)
        return LiveProcessingResult(
            segments=kept,
            analyses=analyses,
            clips=clips,
            ffmpeg={
                "available": not degraded,
                "degraded": degraded,
                "reason": reason,
                "commands": [item.to_payload() for item in commands],
            },
            usage={**usage, "cost_usd": round(usage["cost_usd"], 6)},
            dedupe=dedupe_report,
        )


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(line for line in stripped.splitlines() if not line.strip().startswith("```")).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    return stripped[start : end + 1] if start != -1 and end > start else stripped


def _fallback_draft(batch: Sequence[LiveSegment]) -> _BatchDraft:
    longest = max(batch, key=lambda item: len(item.text))
    points = [item.text[:24] for item in batch if item.text.strip()][:3]
    return _BatchDraft(
        summary=f"离线兜底：本批 {len(batch)} 段，最长段为第 {longest.index} 段（模型输出未通过结构化校验）",
        selling_points=points,
        highlight_index=list(batch).index(longest),
    )


__all__ = ["LiveProcessingResult", "LiveProcessor", "dedupe_overlaps"]
