from __future__ import annotations

import json
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    def __init__(self, api_url: str, model: str, api_key: str, timeout_s: int = 60) -> None:
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_json(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> dict:
        return self.chat_json_messages(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat_json_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.1,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError("LLM request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response body is not valid JSON.") from exc

        try:
            content = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response is missing choices[0].message.content.") from exc

        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 2:
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM message content is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM message content must decode to a JSON object.")
        return parsed
