"""Unified API client for OpenAI and OpenRouter models."""
import os
import time
import json
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client(provider: str) -> OpenAI:
    """Get an OpenAI-compatible client for the specified provider."""
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "openrouter":
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def chat_completion(
    messages: list[dict],
    model_id: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 5,
) -> str:
    """Send a chat completion request with retry logic."""
    client = get_client(provider)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            wait_time = min(2 ** attempt * 2, 60)
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(f"API call failed after {max_retries} retries")


def run_multi_turn_conversation(
    system_prompt: str,
    turns: list[dict],
    model_id: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[str]:
    """Run a multi-turn conversation, collecting all assistant responses.

    Args:
        system_prompt: System message.
        turns: List of dicts with 'user' key for user messages.
            If a turn has 'assistant_override', that response is used instead of API call.
        model_id: Model identifier.
        provider: API provider.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.

    Returns:
        List of assistant response strings.
    """
    messages = [{"role": "system", "content": system_prompt}]
    responses = []

    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})

        if "assistant_override" in turn:
            resp = turn["assistant_override"]
        else:
            resp = chat_completion(
                messages, model_id, provider, temperature, max_tokens
            )

        messages.append({"role": "assistant", "content": resp})
        responses.append(resp)

    return responses
