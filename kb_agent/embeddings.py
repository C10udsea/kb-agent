"""Embedding：把文本变成向量。

这是整个 RAG 的地基。核心思想：
- 文字不能直接做数学运算，所以先用 Embedding 模型把每段文本映射成
  一个高维向量（例如 512 / 1024 维的小数列表）；
- 语义相近的文本，其向量在空间中距离近（通常用余弦相似度衡量）；
- 于是“找最相关的段落”就变成了“找距离最近的向量”，可以用向量库快速完成。

本项目支持多种实现：
1. OpenAICompatEmbedder —— 国内推荐，使用智谱 GLM 的 Embedding 接口
   （OpenAI 兼容协议；DeepSeek 目前不提供 Embedding）；
2. VoyageEmbedder —— Anthropic 官方推荐的 Embedding，保留可选；
3. LocalHashEmbedder —— 纯本地演示方案（不需要 API Key）。
   它用“字符 n-gram + 特征哈希”构造向量，只能捕捉字面相似度，
   不能真正理解语义。它的作用是在你没有 API Key 时也能跑通全流程，
   但真实知识库请务必使用 GLM 或 Voyage。
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from abc import ABC, abstractmethod

from . import config


class Embedder(ABC):
    """Embedding 服务抽象：入库和查询都通过这两个方法。"""

    name = "embedder"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把“知识库文档片段”编码成向量（input_type=document）。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """把“用户问题”编码成向量（input_type=query）。"""


class VoyageEmbedder(Embedder):
    """调用 Voyage AI 的官方 Python SDK。

    - voyage-3-lite：便宜、快，学习期首选；
    - voyage-3：质量更高，正式使用可切换。
    """

    name = "voyage"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        batch_size: int = 128,
        output_dimension: int | None = None,
    ) -> None:
        try:
            from voyageai import Client
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise RuntimeError(
                "未安装 voyageai，请先运行: pip install voyageai"
            ) from exc
        self.client = Client(api_key=api_key)
        self.model = model or config.voyage_model()
        self.batch_size = batch_size
        self.output_dimension = output_dimension

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        # 分批调用：1) 避免单次请求过大；2) 网络抖动时只重试一小批
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                output_dimension=self.output_dimension,
            )
            vectors.extend(result.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # 英文单词之间保留空格；其余空白折叠
    return re.sub(r"\s+", " ", text).strip()


def _ngrams(text: str) -> list[str]:
    """中文按字 n-gram，英文同时按词 n-gram，尽量覆盖两种语言。"""
    grams: list[str] = []
    compact = text.replace(" ", "")
    for n in (2, 3):
        grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    words = re.findall(r"[a-z0-9_]+", text)
    for word in words:
        if len(word) >= 3:
            grams.append(word)
    return grams


def _hash_sign(token: str, dim: int) -> tuple[int, int]:
    """把一个 n-gram 稳定地映射到 [0, dim) 和 ±1 符号。

    这就是工业界常用的 feature hashing（哈希技巧）的简化版。
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest[:4], "little") % dim
    sign = 1 if digest[4] % 2 == 0 else -1
    return idx, sign


class OpenAICompatEmbedder(Embedder):
    """OpenAI Embeddings 兼容接口。

    国内最常用的是智谱 GLM：
        base_url = https://open.bigmodel.cn/api/paas/v4
        model    = embedding-3（如不可用可改为 embedding-2）

    与 Voyage 的区别：这个协议没有 document/query 输入类型之分，
    两个方法底层一样，但保留两个入口可以让未来平滑扩展。
    """

    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 32) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = max(1, batch_size)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(result.data, key=lambda item: getattr(item, "index", 0))
            vectors.extend([item.embedding for item in ordered])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class LocalHashEmbedder(Embedder):
    """无 Key 演示版：把文本投影到 dim 维稀疏哈希向量。

    注意：它衡量的是“字面重合度”，不是真正的语义相似度。
    例如“苹果手机”和“iPhone”对它来说可能完全不相关。
    """

    name = "local-hash"

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _ngrams(_normalize(text)):
            idx, sign = _hash_sign(token, self.dim)
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedder(batch_size: int = 32) -> Embedder:
    """工厂函数：按 EMBEDDING_PROVIDER 选择真实模型，否则用本地演示版。

    国内推荐组合：LLM=DeepSeek + Embedding=GLM。
    """
    provider = config.resolve_embedding_provider()
    if provider == "voyage":
        settings = config.embedding_settings(provider)
        return VoyageEmbedder(api_key=settings["api_key"], batch_size=batch_size)
    if provider in ("glm", "openai"):
        settings = config.embedding_settings(provider)
        return OpenAICompatEmbedder(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            batch_size=batch_size,
        )
    print("⚠️  未检测到 Embedding API Key（支持 GLM_API_KEY / VOYAGE_API_KEY），将使用本地演示 Embedding。")
    return LocalHashEmbedder()
