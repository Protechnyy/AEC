"""OpenAI-compatible LLM utilities for zero-shot relation extraction."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

DEFAULT_MODEL = "gpt-4o"
DEFAULT_LOCAL_API_KEY = "EMPTY"
DEBUG_DIR_ENV = "AEC_LLM_DEBUG_DIR"

MODEL_ALIASES: dict[str, dict[str, str | None]] = {
    "llama3-8b": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "base_url": None},
    "llama3-70b": {"model": "meta-llama/Meta-Llama-3-70B-Instruct", "base_url": None},
    "gpt3.5-turbo": {"model": "gpt-3.5-turbo", "base_url": "https://api.openai.com/v1"},
    "gpt4o": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
}


def resolve_llm_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None, str | None]:
    raw_model = model or os.getenv("AEC_LLM_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    alias_config = MODEL_ALIASES.get(raw_model.lower())
    resolved_model = str(alias_config.get("model")) if alias_config else raw_model
    resolved_base_url = base_url or os.getenv("AEC_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not resolved_base_url and alias_config:
        alias_base_url = alias_config.get("base_url")
        resolved_base_url = str(alias_base_url) if alias_base_url else None
    resolved_api_key = api_key or os.getenv("AEC_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if resolved_base_url and not resolved_api_key:
        resolved_api_key = DEFAULT_LOCAL_API_KEY
    return resolved_model, resolved_base_url, resolved_api_key


def _debug_log_llm_reply(tag: str, reply: str) -> None:
    debug_dir = os.getenv(DEBUG_DIR_ENV)
    if not debug_dir:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob(f"{tag}_*.txt"))
    (path / f"{tag}_{len(existing) + 1:04d}.txt").write_text(reply, encoding="utf-8")


def _extract_json_fragment(reply: str) -> str:
    reply = reply.strip()
    if not reply:
        return reply
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", reply)
    if fenced_match:
        return fenced_match.group(1).strip()
    candidates: list[str] = []
    for opener, closer in (("[", "]"), ("{", "}")):
        start = reply.find(opener)
        end = reply.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(reply[start : end + 1].strip())
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return reply


def _load_json_reply(reply: str) -> Any | None:
    for candidate in (reply.strip(), _extract_json_fragment(reply)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def call_llm(
    messages: list[dict[str, str]],
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    request_tag: str = "llm",
    max_tokens: int | None = None,
) -> str:
    """Call a chat model and return the assistant text reply."""

    if OpenAI is None:
        raise RuntimeError("openai package is not installed; install it to use LLM features.")

    resolved_model, resolved_base_url, resolved_api_key = resolve_llm_config(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if not resolved_api_key:
        raise RuntimeError(
            "No API key configured. Set OPENAI_API_KEY/AEC_LLM_API_KEY, or provide "
            "an OpenAI-compatible base URL so a local server can be used."
        )

    timeout_seconds = float(os.getenv("AEC_LLM_TIMEOUT", "180"))
    retries = max(0, int(os.getenv("AEC_LLM_RETRIES", "1")))
    resolved_max_tokens = max_tokens if max_tokens is not None else int(os.getenv("AEC_LLM_MAX_TOKENS", "768"))
    resolved_top_p = float(os.getenv("AEC_LLM_TOP_P", "1.0"))
    seed_env = os.getenv("AEC_LLM_SEED")
    resolved_seed = int(seed_env) if seed_env not in {None, ""} else None
    verbose = os.getenv("AEC_LLM_VERBOSE", "1").lower() not in {"0", "false", "no"}
    fail_soft = os.getenv("AEC_LLM_FAIL_SOFT", "1").lower() not in {"0", "false", "no"}

    client_kwargs: dict[str, str] = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        start = time.monotonic()
        if verbose:
            print(
                f"[LLM:{request_tag}] start attempt={attempt + 1}/{retries + 1} "
                f"model={resolved_model} timeout={timeout_seconds:.0f}s max_tokens={resolved_max_tokens}",
                flush=True,
            )
        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": resolved_top_p,
            "max_tokens": resolved_max_tokens,
            "timeout": timeout_seconds,
        }
        if resolved_seed is not None:
            request_kwargs["seed"] = resolved_seed
        try:
            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content
            reply = content.strip() if isinstance(content, str) else str(content or "").strip()
            _debug_log_llm_reply(request_tag, reply)
            if verbose:
                print(f"[LLM:{request_tag}] done elapsed={time.monotonic() - start:.1f}s", flush=True)
            return reply
        except Exception as exc:
            last_error = exc
            if verbose:
                print(
                    f"[LLM:{request_tag}] failed attempt={attempt + 1}/{retries + 1} "
                    f"elapsed={time.monotonic() - start:.1f}s error={exc.__class__.__name__}: {exc}",
                    flush=True,
                )
            if attempt >= retries:
                if fail_soft:
                    return "[]" if request_tag == "re_planning" else "{}"
                raise
    if last_error is not None:
        raise last_error
    return ""
