#!/usr/bin/env python3
"""阶段 1 · 第 2 周产出：对知识库提问，回答带引用来源。

流程（每次提问）：
    问题 -> Embedding -> 向量库 top-k 检索 -> 原文拼进 prompt -> LLM 回答

用法：
    python scripts/chat.py --top-k 5
命令：
    /topk N   调整检索片段数
    /sources  列出知识库中的文档
    /exit     退出
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent import config
from kb_agent.embeddings import get_embedder
from kb_agent.kb_tools import KnowledgeBase
from kb_agent.llm import get_chat_model
from kb_agent.rag import ask
from kb_agent.vector_store import get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kb-agent 知识库问答")
    parser.add_argument("--top-k", type=int, default=5, help="检索片段数量（默认 5）")
    parser.add_argument("--question", "-q", type=str, default=None, help="直接提问一次后退出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ensure_dirs()

    store = get_vector_store()
    if store.count() == 0:
        print("❌ 知识库为空。请先运行: python scripts/ingest.py")
        return 1

    embedder = get_embedder()
    kb = KnowledgeBase(store, embedder)
    model = get_chat_model()
    top_k = max(1, args.top_k)

    print("=" * 64)
    print("📚 kb-agent 知识库问答（RAG）")
    print(f"   片段数: {store.count()} | top-k: {top_k} | 模型: {model.name}")
    print("   输入 /topk N 调整检索数，/sources 查看文档，/exit 退出")
    print("=" * 64)

    if args.question:
        ask(args.question, kb, model=model, top_k=top_k)
        print()
        return 0

    while True:
        try:
            question = input("\n🧑 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            return 0

        if not question:
            continue
        if question.lower() in ("/exit", "/quit"):
            print("👋 再见！")
            return 0
        if question.lower().startswith("/topk"):
            parts = question.split()
            if len(parts) != 2:
                print("用法: /topk 5")
                continue
            try:
                top_k = max(1, int(parts[1]))
                print(f"✅ top-k 已调整为 {top_k}")
            except ValueError:
                print("用法: /topk 5")
            continue
        if question.lower() == "/sources":
            print("\n".join(f"  - {s}" for s in store.sources()))
            continue

        ask(question, kb, model=model, top_k=top_k)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
