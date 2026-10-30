"""ANSWERING part 3: the ONLY file that talks to NVIDIA.
NVIDIA's API follows the OpenAI format, so we use the openai package."""
from openai import OpenAI

from rag import config


def chat(messages: list[dict]) -> str:
    if not config.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing. Copy .env.example to .env and set it.")
    client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)
    response = client.chat.completions.create(
        model=config.NVIDIA_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        extra_body=config.LLM_EXTRA_BODY or None,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("The model returned an empty answer. Try raising LLM_MAX_TOKENS.")
    return text.strip()