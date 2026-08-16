#!/usr/bin/env python3
"""阶段 2 产出：能自主检索、多步查证的 Agent。

对比 scripts/chat.py：
- chat.py：固定“检索一次 -> 回答”；
- agent.py：把检索变成工具，LLM 自主决定查什么、查几次、何时回答。

用法：
    python scripts/agent.py                        # 交互式
    python scripts/agent.py -q "对比A和B中关于X的观点"
    python scripts/agent.py --manual               # 强制手写循环（教学用）
命令：
    /sources  查看知识库文档
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
from kb_agent.agent import run_agent, run_agent_demo, run_agent_manual
from kb_agent.embeddings import get_embedder
from kb_agent.kb_tools import KnowledgeBase
from kb_agent.llm import get_chat_model
from kb_agent.vector_store import get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kb-agent 知识库 Agent")
    parser.add_argument("--question", "-q", type=str, default=None, help="直接提问一次后退出")
    parser.add_argument("--max-turns", type=int, default=6, help="最多工具调用轮数（默认 6）")
    parser.add_argument("--manual", action="store_true", help="强制使用手写循环（教学观察用）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ensure_dirs()

    store = get_vector_store()
    if store.count() == 0:
        print("❌ 知识库为空。请先运行: python scripts/ingest.py")
        return 1

    kb = KnowledgeBase(store, get_embedder())

    provider = config.resolve_llm_provider()
    print("=" * 64)
    print("🕵️  kb-agent 知识库 Agent")
    print(f"   片段数: {store.count()} | 最多工具轮数: {args.max_turns}")
    if provider:
        if provider == "anthropic":
            mode = "Claude + Tool Use（SDK ToolRunner）" if not args.manual else "Claude + 手写循环（教学）"
        else:
            mode = f"{provider.upper()} + Tool Use（手写循环）"
        print(f"   模式: {mode}")
    else:
        print("   模式: 演示 Agent（未检测到 LLM API Key）")
    print("=" * 64)

    def ask_once(question: str) -> None:
        if config.has_llm():
            if args.manual:
                run_agent_manual(question, kb, model=get_chat_model(), max_turns=args.max_turns)
            else:
                run_agent(question, kb, max_turns=args.max_turns)
        else:
            run_agent_demo(question, kb, max_turns=args.max_turns)
        print()

    if args.question:
        ask_once(args.question)
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
        if question.lower() == "/sources":
            print(kb.list_documents())
            continue
        ask_once(question)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
