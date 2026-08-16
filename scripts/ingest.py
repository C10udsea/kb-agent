#!/usr/bin/env python3
"""阶段 1 入库脚本：文档 -> 切块 -> Embedding -> 向量库。

用法：
    python scripts/ingest.py                 # 按默认参数入库 docs/
    python scripts/ingest.py --chunk-size 500 --chunk-overlap 80
    python scripts/ingest.py --clear         # 清空旧数据后重新入库

运行一次即可；之后 chat.py / agent.py 直接读取持久化的向量库。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从项目根目录运行时能 import kb_agent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent import config
from kb_agent.chunking import chunk_id, read_documents, split_text
from kb_agent.embeddings import get_embedder
from kb_agent.vector_store import get_vector_store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 docs/ 下的 md/txt 文档写入向量库")
    parser.add_argument("--docs", type=Path, default=config.DOCS_DIR, help="文档目录")
    parser.add_argument("--chunk-size", type=int, default=800, help="每个块的字符数（默认 800）")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="相邻块重叠字符数（默认 120）")
    parser.add_argument("--clear", action="store_true", help="入库前清空旧数据")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding 批大小")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()

    print("=" * 64)
    print("📥 kb-agent 文档入库")
    print(f"   文档目录: {args.docs}")
    print(f"   切块参数: chunk_size={args.chunk_size}, overlap={args.chunk_overlap}")
    print("=" * 64)

    if not args.docs.exists():
        print(f"❌ 目录不存在: {args.docs}")
        return 1

    documents = read_documents(args.docs)
    if not documents:
        print(f"⚠️  {args.docs} 下没有找到 .md / .txt 文件。")
        print("   请先放入你的笔记，或运行: mkdir -p docs")
        return 1

    chunks = []
    for doc in documents:
        doc_chunks = split_text(
            doc.text,
            source=doc.source,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        chunks.extend(doc_chunks)
        print(f"   {doc.source}: {len(doc_chunks)} 个片段")

    print(f"\n共 {len(documents)} 个文档，切成 {len(chunks)} 个片段。")

    embedder = get_embedder(batch_size=max(1, args.batch_size))
    store = get_vector_store()
    if args.clear:
        print("🧹 清空旧向量数据...")
        store.clear()

    print(f"🧮 使用 {embedder.name} 生成向量...")
    ids = [
        chunk_id(c.source, c.index, args.chunk_size, args.chunk_overlap)
        for c in chunks
    ]
    # 分批写入：避免一次性把所有文本堆在内存/请求里
    batch = max(1, args.batch_size)
    for start in range(0, len(chunks), batch):
        end = min(start + batch, len(chunks))
        batch_chunks = chunks[start:end]
        vectors = embedder.embed_documents([c.text for c in batch_chunks])
        store.upsert(batch_chunks, vectors, ids[start:end])
        print(f"   ✅ 已写入 {end}/{len(chunks)}")

    print("\n" + "=" * 64)
    print(f"🎉 入库完成！向量库现有 {store.count()} 个片段。")
    print("下一步：")
    print("   python scripts/chat.py   # RAG 问答（阶段1）")
    print("   python scripts/agent.py  # 多步 Agent 问答（阶段2）")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
