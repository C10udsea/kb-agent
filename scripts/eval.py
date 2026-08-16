#!/usr/bin/env python3
"""阶段 1 · 第 3 周产出：对比不同 chunk_size / top-k 的检索效果。

评估文件：docs/eval_questions.json
指标：
- Hit@k：标准答案来源文档是否出现在前 k 个检索结果中（命中率）；
- MRR：第一个正确来源出现位置的倒数平均（越接近 1 越好）。

你可以非常直观地看到计划里的核心结论：
检索质量决定回答质量。top-k 太小会漏掉答案，太大又引入噪声。

用法：
    python scripts/eval.py
    python scripts/eval.py --chunk-sizes 400,800,1200 --top-k 3,5,8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent import config
from kb_agent.chunking import read_documents, split_text
from kb_agent.embeddings import get_embedder
from kb_agent.vector_store import (
    ChromaVectorStore,
    JSONVectorStore,
    VectorStore,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估 chunk size 与 top-k 对检索质量的影响")
    p.add_argument("--docs", type=Path, default=config.DOCS_DIR)
    p.add_argument("--questions", type=Path, default=config.DOCS_DIR / "eval_questions.json")
    p.add_argument("--chunk-sizes", type=str, default="400,800,1200", help="逗号分隔的字符数列表")
    p.add_argument("--top-k", type=str, default="3,5,8", help="逗号分隔的 top-k 列表")
    p.add_argument("--overlap-ratio", type=float, default=0.15, help="overlap 占 chunk_size 的比例")
    return p.parse_args(argv)


def _build_store(tmp: Path) -> tuple[VectorStore, Path]:
    try:
        return ChromaVectorStore(tmp / "chroma", collection_name="kb_eval"), tmp / "chroma"
    except Exception:
        return JSONVectorStore(tmp / "store.json"), tmp / "store.json"


def evaluate_config(
    questions: list[dict],
    documents,
    chunk_size: int,
    overlap: int,
    embedder,
    top_k_values: list[int],
) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"kb_eval_{chunk_size}_"))
    try:
        store, _ = _build_store(tmp)
        for doc in documents:
            chunks = split_text(
                doc.text,
                source=doc.source,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            if not chunks:
                continue
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.upsert(
                chunks,
                vectors,
                [f"{doc.source}::{c.index}" for c in chunks],
            )

        if not questions:
            return {"mrr": {"hit@k": 0.0, "keyword_hit": None}}

        hit_ranks: list[int] = []
        keyword_hits: list[int] = []
        reciprocal_ranks: list[float] = []
        max_k = max(q.get("top_k", 5) for q in questions)

        for q in questions:
            query_vec = embedder.embed_query(q["question"])
            results = store.search(query_vec, top_k=max_k)
            expected_sources = set(q.get("sources", []))
            keywords = q.get("keywords", [])

            # 来源命中：正确文档第一次出现在第几位（1-based）
            first_hit = None
            keyword_hit = False
            for rank, r in enumerate(results, 1):
                if first_hit is None and r.source in expected_sources:
                    first_hit = rank
                if any(k.lower() in r.text.lower() for k in keywords):
                    keyword_hit = True

            hit_ranks.append(first_hit if first_hit is not None else 10**6)
            keyword_hits.append(1 if keyword_hit else 0)
            if first_hit is not None:
                reciprocal_ranks.append(1.0 / first_hit)
            else:
                reciprocal_ranks.append(0.0)

        rows = {}
        for k in top_k_values:
            hit_k = sum(1 for rank in hit_ranks if rank <= k) / len(questions)
            # keyword_hit 不受 k 影响，因为我们都取到最大 k 判断
            rows[k] = {
                "hit@k": round(hit_k, 3),
                "keyword_hit": round(sum(keyword_hits) / len(questions), 3),
            }
        rows["mrr"] = {
            "hit@k": round(sum(reciprocal_ranks) / len(questions), 3),
            "keyword_hit": None,
        }
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()

    questions_file: Path = args.questions
    if not questions_file.exists():
        print(f"❌ 评估问题文件不存在: {questions_file}")
        return 1
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    for q in questions:
        q.setdefault("top_k", 5)
        q.setdefault("keywords", [])
        q.setdefault("sources", [])

    documents = read_documents(args.docs)
    if not documents:
        print("❌ docs 目录没有文档。")
        return 1

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    top_k_values = sorted({int(x) for x in args.top_k.split(",") if x.strip()})
    for q in questions:
        q["top_k"] = max(top_k_values) if top_k_values else q["top_k"]

    embedder = get_embedder()
    print("=" * 78)
    print("📊 RAG 检索评估")
    print(f"   文档数: {len(documents)} | 问题数: {len(questions)}")
    print(f"   Embedding: {embedder.name} | 评估 top-k: {top_k_values}")
    print("=" * 78)

    all_rows: list[tuple[int, int, dict]] = []
    for chunk_size in chunk_sizes:
        overlap = min(chunk_size - 1, int(chunk_size * args.overlap_ratio))
        print(f"\n▶ chunk_size={chunk_size}, overlap={overlap}")
        rows = evaluate_config(questions, documents, chunk_size, overlap, embedder, top_k_values)
        header = "  top-k   " + "  ".join(f"k={k:<6}" for k in top_k_values) + "MRR"
        print("  " + header)
        vals = "  Hit@k   " + "  ".join(f"{rows[k]['hit@k']:<8}" for k in top_k_values) + f"{rows['mrr']['hit@k']:<8}"
        print("  " + vals)
        if any(rows[k]["keyword_hit"] is not None for k in top_k_values):
            kw = "  关键词   " + "  ".join(f"{rows[k]['keyword_hit']:<8}" for k in top_k_values)
            print("  " + kw)
        for k in top_k_values:
            all_rows.append((chunk_size, k, rows[k]))

    print("\n" + "=" * 78)
    print("结论提示：")
    print("1. top-k 增大通常能提高 Hit@k，但会把更多噪声塞进 prompt；")
    print("2. chunk_size 需要配合文档风格选择：颗粒太粗/太细都会掉点；")
    print("3. 把最佳组合用于 scripts/ingest.py 的 --chunk-size 参数。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
