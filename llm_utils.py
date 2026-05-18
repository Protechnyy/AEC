"""
Utility function for OpenAI-compatible chat models.
"""

from __future__ import annotations

import os
from typing import Dict, List

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


def call_llm(messages: List[Dict[str, str]], model: str = "gpt-4o") -> str:
    """Call an OpenAI-compatible chat model and return assistant content."""

    if openai is None:
        raise RuntimeError(
            "openai package is not installed; install requirements.txt first."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set; cannot call the model."
        )

    base_url = os.getenv("OPENAI_BASE_URL") or None
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return (content or "").strip()
