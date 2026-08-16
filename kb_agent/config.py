"""集中管理路径与配置。

本项目最重要的工程习惯之一：
- API Key 只放在环境变量 / `.env` 文件中；
- `.env` 被 `.gitignore` 排除，绝不能提交到 git。

因为你在国内，本项目同时支持三套模型服务：
1. DeepSeek（对话 LLM，OpenAI 兼容接口）
2. 智谱 GLM（对话 LLM + Embedding，OpenAI 兼容接口）
3. Anthropic Claude / Voyage（保留可选，未来网络允许时可切换）

配置方式：复制 .env.example 为 .env，按需填写。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：kb_agent/config.py 的上一级
BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# 程序启动时自动读取项目根目录下的 .env（如果存在）
load_dotenv(BASE_DIR / ".env")


def ensure_dirs() -> None:
    """确保 data 等目录存在，避免第一次运行时报错。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


_PLACEHOLDER_MARKERS = ("在这里填入", "your_", "your-", "changeme")


def get_api_key(*names: str) -> str | None:
    """依次检查多个环境变量名，返回第一个非空、非占位符的值。"""
    for name in names:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        if any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
            print(f"⚠️  检测到 {name} 仍是 .env.example 中的占位符，请先填入真实 Key。")
            return None
        return value
    return None


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


# ---------------------------------------------------------------------------
# LLM（对话模型）Provider
# ---------------------------------------------------------------------------

def anthropic_key() -> str | None:
    return get_api_key("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")


def deepseek_key() -> str | None:
    return get_api_key("DEEPSEEK_API_KEY")


def glm_key() -> str | None:
    return get_api_key("GLM_API_KEY", "ZHIPU_API_KEY")


def openai_compat_key() -> str | None:
    return get_api_key("OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY")


def _provider_keys() -> dict[str, str | None]:
    return {
        "anthropic": anthropic_key(),
        "deepseek": deepseek_key(),
        "glm": glm_key(),
        "openai": openai_compat_key(),
    }


def resolve_llm_provider() -> str | None:
    """按 LLM_PROVIDER 环境变量选择服务商。

    - auto：自动探测第一个有 Key 的服务商；
    - deepseek / glm / anthropic / openai：强制使用指定服务商。
    """
    want = env_str("LLM_PROVIDER", "auto").lower()
    keys = _provider_keys()
    if want == "auto":
        # 国内场景优先 DeepSeek / GLM；若配置了 Claude 也会识别
        for name in ("deepseek", "glm", "anthropic", "openai"):
            if keys.get(name):
                return name
        return None
    if want in keys and keys.get(want):
        return want
    if want in keys:
        print(f"⚠️  LLM_PROVIDER={want}，但未检测到对应的 API Key。")
    else:
        print(f"⚠️  未知的 LLM_PROVIDER={want}（可选: auto/deepseek/glm/anthropic/openai）。")
    return None


def deepseek_model() -> str:
    return env_str("DEEPSEEK_MODEL", "deepseek-chat")


def glm_model() -> str:
    return env_str("GLM_MODEL", "glm-4-flash")


def anthropic_model() -> str:
    return env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def openai_compat_model() -> str:
    return env_str("OPENAI_COMPAT_MODEL", "gpt-4o-mini")


def llm_settings(provider: str | None = None) -> dict:
    """返回指定 LLM Provider 的连接参数。"""
    provider = provider or resolve_llm_provider()
    if provider == "deepseek":
        return {
            "provider": "deepseek",
            "api_key": deepseek_key() or "",
            "base_url": env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": deepseek_model(),
        }
    if provider == "glm":
        return {
            "provider": "glm",
            "api_key": glm_key() or "",
            "base_url": env_str("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            "model": glm_model(),
        }
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": openai_compat_key() or "",
            "base_url": env_str("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"),
            "model": openai_compat_model(),
        }
    if provider == "anthropic":
        return {
            "provider": "anthropic",
            "api_key": anthropic_key() or "",
            "model": anthropic_model(),
        }
    return {}


def has_llm() -> bool:
    return resolve_llm_provider() is not None


# 兼容旧代码
def has_claude() -> bool:
    return anthropic_key() is not None


# ---------------------------------------------------------------------------
# Embedding Provider
# ---------------------------------------------------------------------------

def voyage_key() -> str | None:
    return get_api_key("VOYAGE_API_KEY", "VOYAGE_API")


def voyage_model() -> str:
    return env_str("VOYAGE_MODEL", "voyage-3-lite")


def glm_embedding_model() -> str:
    return env_str("GLM_EMBEDDING_MODEL", "embedding-3")


def openai_embedding_model() -> str:
    return env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def resolve_embedding_provider() -> str | None:
    """选择 Embedding 服务商。

    DeepSeek 目前不提供 Embedding API，因此国内组合通常是：
    LLM=DeepSeek + Embedding=GLM。
    """
    want = env_str("EMBEDDING_PROVIDER", "auto").lower()
    available = {
        "voyage": voyage_key() is not None,
        "glm": glm_key() is not None,
        "openai": openai_compat_key() is not None,
    }
    if want == "auto":
        # 有 GLM Key 时优先 GLM（国内可用）；其次 Voyage；最后通用 OpenAI 兼容
        for name in ("glm", "voyage", "openai"):
            if available.get(name):
                return name
        return None
    if want in available and available.get(want):
        return want
    if want in available:
        print(f"⚠️  EMBEDDING_PROVIDER={want}，但未检测到对应的 API Key。")
    else:
        print(f"⚠️  未知的 EMBEDDING_PROVIDER={want}（可选: auto/glm/voyage/openai）。")
    return None


def embedding_settings(provider: str | None = None) -> dict:
    provider = provider or resolve_embedding_provider()
    if provider == "glm":
        return {
            "provider": "glm",
            "api_key": glm_key() or "",
            "base_url": env_str("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            "model": glm_embedding_model(),
        }
    if provider == "voyage":
        return {"provider": "voyage", "api_key": voyage_key() or "", "model": voyage_model()}
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": openai_compat_key() or "",
            "base_url": env_str("OPENAI_EMBEDDING_BASE_URL", openai_compat_base_url()),
            "model": openai_embedding_model(),
        }
    return {}


def openai_compat_base_url() -> str:
    return env_str("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1")


def has_embedding() -> bool:
    return resolve_embedding_provider() is not None


# 兼容旧代码
def has_voyage() -> bool:
    return voyage_key() is not None
