"""Small interchangeable LLM adapters; tokens stay in local server memory."""
import json
import re
from urllib.parse import quote

from .network import post_json


def complete(config, messages, system, max_tokens=6000):
    provider = config.get("provider", "compatible")
    model = str(config.get("model", "")).strip()
    token = str(config.get("token", ""))
    if not model:
        raise ValueError("Enter a model identifier offered by your provider.")
    if provider == "anthropic":
        data = post_json("https://api.anthropic.com/v1/messages",
                         {"x-api-key": token, "anthropic-version": "2023-06-01"},
                         {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages})
        text = "\n".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
    elif provider == "gemini":
        data = post_json(f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent",
                         {"x-goog-api-key": token},
                         {"systemInstruction": {"parts": [{"text": system}]},
                          "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                                        "parts": [{"text": m["content"]}]} for m in messages],
                          "generationConfig": {"maxOutputTokens": max_tokens}})
        candidates = data.get("candidates", [])
        text = "\n".join(p.get("text", "") for p in (candidates[0].get("content", {}).get("parts", []) if candidates else []))
    elif provider == "compatible":
        # Includes OpenRouter, Groq and local Ollama's OpenAI-compatible endpoint.
        base = str(config.get("endpoint", "")).rstrip("/")
        if not base:
            raise ValueError("Enter the provider's base URL, usually ending in /v1.")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages]}
        # Some reasoning providers reject max_tokens; omission is an explicit compatibility choice.
        if config.get("send_max_tokens", True):
            body["max_tokens"] = max_tokens
        data = post_json(base + "/chat/completions", headers, body, allow_local=True)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        raise ValueError("Unknown LLM provider protocol.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The LLM returned no text. Try a model that supports ordinary text responses.")
    return text.strip()


def json_reply(config, system, payload):
    """Recover wrappers locally, then retry at most twice from original evidence.

    Never execute returned code, repair numbers heuristically, or accept a
    truncated object. Provider/network failures are not repeatedly retried here.
    """
    messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    for attempt in range(3):
        instruction = system + ('\nReturn one complete JSON object only. No commentary or fences. Keep issues concise.' if attempt else '')
        try:
            text = complete(config, messages, instruction, max_tokens=12000)
            return parse_object(text)
        except ValueError as error:
            if not isinstance(error, (json.JSONDecodeError, InvalidObject)) and 'returned no text' not in str(error):
                raise
    raise ValueError('The provider returned invalid JSON after three attempts. Other sources can still be processed; no manual JSON editing is required.')


class InvalidObject(ValueError):
    pass


def parse_object(text):
    """Accept exactly one complete object, optionally fenced or with prose."""
    def invalid_constant(value):
        raise InvalidObject('Non-finite JSON number')
    decoder = json.JSONDecoder(parse_constant=invalid_constant)
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    if clean.startswith('['):
        raise InvalidObject('Expected an object, not an array')
    # Start at the first opening brace only: searching later braces could
    # accidentally accept an inner object from a truncated outer response.
    start = clean.find('{')
    if start < 0:
        raise InvalidObject('No JSON object')
    value, end = decoder.raw_decode(clean, start)
    if not isinstance(value, dict) or any(c in clean[end:] for c in '{}[]'):
        raise InvalidObject('Multiple or incomplete JSON objects')
    return value
