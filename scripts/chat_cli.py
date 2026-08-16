#!/usr/bin/env python3
"""阶段 1 · 第 1 周产出：命令行多轮聊天（流式输出）。

要点：
- API 无状态：程序自己维护 messages 列表，每轮把完整历史发回去；
- system 提示词只发一次，角色/规则放在里面；
- stream 模式逐 token 打印，像打字机一样。

用法：
    python scripts/chat_cli.py
命令：
    /clear  清空历史
    /exit   退出
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent.llm import get_chat_model

SYSTEM_PROMPT = "你是一位耐心、简洁的 AI 学习助手。用简体中文回答。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="命令行多轮聊天（流式输出）")
    return parser.parse_args()


def main() -> int:
    parse_args()
    model = get_chat_model()
    messages: list[dict] = []

    print("=" * 64)
    print("💬 命令行多轮聊天（流式输出）")
    print(f"   模型: {model.name}  |  输入 /clear 清空历史，/exit 退出")
    print("=" * 64)

    while True:
        try:
            question = input("\n🧑 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            return 0

        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit"):
            print("👋 再见！")
            return 0
        if question.lower() == "/clear":
            messages.clear()
            print("🗑️  对话历史已清空。")
            continue

        messages.append({"role": "user", "content": question})
        print("🤖 助手：", end="", flush=True)
        answer = model.stream(
            messages,
            system=SYSTEM_PROMPT,
            on_text=lambda token: print(token, end="", flush=True),
        )
        print()
        messages.append({"role": "assistant", "content": answer})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
