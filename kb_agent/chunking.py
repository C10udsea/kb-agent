"""文档读取与切块（chunking）。

为什么需要切块？
1. LLM 的上下文窗口有限，不能把整本书塞进 prompt；
2. 检索需要“细粒度”的片段，块太大会混入无关内容；
3. 向量模型也有最大输入长度（Voyage 通常是 32000 token，但短块更精准）。

为什么不能切太碎？
- 一个完整的观点可能被拦腰截断，检索到的碎片缺少上下文，回答质量下降。

本模块实现一个对 Markdown 友好的朴素切分器：
先按标题/段落切分，再按字符数聚合成块，并保留相邻块之间的重叠
（overlap），让语义跨边界时仍有上下文。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# 纯中文一个字符约等于 1 个 token；英文约 4 个字符 1 个 token。
# 这里用“字符数”做近似，适合教学项目；生产系统会用 tiktoken 等精确计数。
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class RawDocument:
    """从磁盘读取到的一整篇文档。"""

    path: Path
    text: str

    @property
    def source(self) -> str:
        """相对 docs/ 的路径，作为引用来源显示给用户。"""
        return str(self.path)


@dataclass
class Chunk:
    """切好的一个文本块 + 元数据。"""

    text: str
    source: str
    section: str
    index: int
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "section": self.section,
            "chunk_index": self.index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


def read_documents(docs_dir: Path, suffixes: tuple[str, ...] = (".md", ".txt")) -> list[RawDocument]:
    """递归读取 docs/ 下所有 markdown / txt 文档，按文件名排序保证结果稳定。"""
    files = sorted(
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )
    docs: list[RawDocument] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="ignore")
        text = text.strip()
        if text:
            docs.append(RawDocument(path=path.relative_to(docs_dir.parent), text=text))
    return docs


def _heading_level(line: str) -> tuple[int, str] | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def _split_long_text(text: str, max_len: int) -> list[str]:
    """超长段落按 max_len 硬切，尽量在句号/换行处断开。"""
    if len(text) <= max_len:
        return [text]
    pieces: list[str] = []
    rest = text
    while len(rest) > max_len:
        cut = rest[:max_len]
        # 在切割窗口后 1/4 处寻找最接近的天然断点
        for sep in ("。", "！", "？", "\n", "；", ". ", "! ", "? "):
            idx = cut.rfind(sep)
            if idx > max_len * 0.6:
                cut = rest[: idx + 1]
                break
        pieces.append(cut)
        rest = rest[len(cut):].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """把全文拆成 (当前所属章节, 段落) 列表。

    只保留每个段落所属的“最近一级标题”，避免在段落里重复标题文字；
    章节信息放进 metadata，供检索结果展示。
    """
    blocks: list[tuple[str, str]] = []
    section = "（文档开头）"
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        para = "\n".join(current).strip()
        if para:
            blocks.append((section, para))
        current = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _heading_level(line)
        if heading:
            flush()
            section = heading[1]
            continue
        if line.strip() == "":
            flush()
            continue
        current.append(line)
    flush()
    return blocks


def split_text(
    text: str,
    source: str = "unknown.md",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Markdown 感知切块。

    流程：
    Markdown 文本 -> 标题/段落块 -> 超长段硬切 -> 聚合成 chunk_size 左右的块
    -> 每块保留上一块末尾 chunk_overlap 字符作为上下文重叠。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    paragraphs = _split_into_blocks(text)
    chunks: list[Chunk] = []
    index = 0
    buffer_text = ""
    buffer_section = paragraphs[0][0] if paragraphs else "（文档开头）"
    buffer_start = 0

    def emit(buf: str, section: str, start: int) -> None:
        nonlocal index
        buf = buf.strip()
        if not buf:
            return
        chunks.append(
            Chunk(
                text=buf,
                source=source,
                section=section,
                index=index,
                char_start=start,
                char_end=start + len(buf),
            )
        )
        index += 1

    for section, para in paragraphs:
        pieces = _split_long_text(para, chunk_size)
        for piece in pieces:
            if buffer_text:
                candidate = buffer_text + "\n\n" + piece
            else:
                candidate = piece
            if len(candidate) <= chunk_size:
                buffer_text = candidate
            else:
                emit(buffer_text, buffer_section, buffer_start)
                overlap_text = buffer_text[-chunk_overlap:] if chunk_overlap else ""
                buffer_start = buffer_start + len(buffer_text) + 2
                buffer_text = (overlap_text + "\n\n" + piece) if overlap_text else piece
                buffer_section = section
                # 重叠 + 新段落仍可能超过 chunk_size：此时硬切，保证块长上限
                while len(buffer_text) > chunk_size:
                    hard_pieces = _split_long_text(buffer_text, chunk_size)
                    for j, hard in enumerate(hard_pieces):
                        if j < len(hard_pieces) - 1:
                            emit(hard, buffer_section, buffer_start)
                            buffer_start += len(hard) + 2
                        else:
                            buffer_text = hard
    emit(buffer_text, buffer_section, buffer_start)

    # 如果最后一个块因为边界处理为空，删除它
    chunks = [c for c in chunks if c.text.strip()]
    for i, c in enumerate(chunks):
        c.index = i
    return chunks


def chunk_id(source: str, index: int, chunk_size: int, chunk_overlap: int) -> str:
    """稳定的块 ID：同一文档、同样切块参数，重复入库不会产生重复数据。

    用 sha1 而不是完整路径，避免 Chroma 对 ID 长度/特殊字符的限制。
    """
    raw = f"{source}|{index}|{chunk_size}|{chunk_overlap}".encode()
    return hashlib.sha1(raw).hexdigest()[:32]
