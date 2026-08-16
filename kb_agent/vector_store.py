"""向量库封装。

ChromaDB 在内部做了两件事：
1. 持久化保存向量和原文；
2. 用近似最近邻（ANN）索引，在几毫秒内从几万/几十万个向量中
   找出与查询向量最接近的 top-k 个。

本模块把 Chroma 封装成三个操作：upsert / search / stats。
同时提供一个纯 Python 的 JSON 后备实现，用于 ChromaDB 安装失败时
仍能演示完整 RAG 流程（不推荐生产使用）。
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk


@dataclass
class SearchResult:
    text: str
    source: str
    section: str
    chunk_index: int
    score: float
    distance: float | None = None

    def to_prompt_block(self, number: int) -> str:
        return (
            f"【资料{number}】来源: {self.source}（章节: {self.section}，"
            f"片段 {self.chunk_index}，相关度 {self.score:.3f}）\n{self.text}"
        )


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma 只接受 str/int/float/bool 类型的 metadata 值。"""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


class VectorStore:
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], ids: list[str]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int = 5, source: str | None = None) -> list[SearchResult]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def sources(self) -> list[str]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str = "kb_agent") -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # HNSW 空间设为 cosine：距离 = 1 - 余弦相似度，因此 distance 越小越相关
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], ids: list[str]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[_clean_metadata(c.to_metadata()) for c in chunks],
        )

    def search(self, query_vector: list[float], top_k: int = 5, source: str | None = None) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
        }
        if source:
            kwargs["where"] = {"source": source}
        res = self.collection.query(**kwargs)

        results: list[SearchResult] = []
        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None
            if not doc:
                continue
            # cosine 距离转回相似度，越接近 1 越相关
            score = 1.0 - float(distance) if distance is not None else float("nan")
            results.append(
                SearchResult(
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    section=str(meta.get("section", "")),
                    chunk_index=int(meta.get("chunk_index", i)),
                    score=score,
                    distance=float(distance) if distance is not None else None,
                )
            )
        return results

    def count(self) -> int:
        return int(self.collection.count())

    def sources(self) -> list[str]:
        try:
            data = self.collection.get(include=["metadatas"])
            sources = {
                str(m.get("source", "unknown"))
                for m in (data.get("metadatas") or [])
                if m
            }
            return sorted(sources)
        except Exception:
            return []

    def clear(self) -> None:
        # Chroma 没有直接清空集合的 API，删除后重建最简单
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )


class JSONVectorStore(VectorStore):
    """纯 Python 暴力最近邻检索。

    没有 ANN 索引，每次查询都和所有向量算一遍余弦相似度。
    只用于 Chroma 不可用时的教学演示；数据量大了会很慢。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"items": []}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], ids: list[str]) -> None:
        with self._lock:
            by_id = {item["id"]: item for item in self._data["items"]}
            for chunk, vector, chunk_id in zip(chunks, vectors, ids):
                by_id[chunk_id] = {
                    "id": chunk_id,
                    "document": chunk.text,
                    "metadata": _clean_metadata(chunk.to_metadata()),
                    "embedding": vector,
                }
            self._data["items"] = list(by_id.values())
            self._save()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query_vector: list[float], top_k: int = 5, source: str | None = None) -> list[SearchResult]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._data["items"]:
            meta = item.get("metadata") or {}
            if source and meta.get("source") != source:
                continue
            score = self._cosine(query_vector, item.get("embedding", []))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for score, item in scored[:top_k]:
            meta = item.get("metadata") or {}
            results.append(
                SearchResult(
                    text=item.get("document", ""),
                    source=str(meta.get("source", "unknown")),
                    section=str(meta.get("section", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=score,
                )
            )
        return results

    def count(self) -> int:
        return len(self._data["items"])

    def sources(self) -> list[str]:
        sources = {
            str((item.get("metadata") or {}).get("source", "unknown"))
            for item in self._data["items"]
        }
        return sorted(sources)

    def clear(self) -> None:
        self._data["items"] = []
        self._save()


def get_vector_store() -> VectorStore:
    """优先 ChromaDB；若导入失败则退回 JSON 暴力检索。"""
    from . import config

    config.ensure_dirs()
    try:
        store = ChromaVectorStore(config.CHROMA_DIR)
        print(f"✅ 使用 ChromaDB 向量库，存储目录: {config.CHROMA_DIR}")
        return store
    except Exception as exc:
        print(f"⚠️  ChromaDB 初始化失败（{type(exc).__name__}: {exc}），退回 JSON 向量库演示模式。")
        return JSONVectorStore(config.DATA_DIR / "fallback_store.json")
