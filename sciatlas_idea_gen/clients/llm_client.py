"""OpenAI-compatible LLM client used across all pipeline steps.

Points at the dmxapi gateway by default (see ``docs/APi信息.txt``) but works
with any OpenAI-compatible ``base_url``. Provides plain chat and a
``chat_json`` helper that parses strict-JSON prompt outputs. When ``trace_path``
is provided, every call is appended to a per-run JSONL trace so each step's
prompt/response pair is inspectable after the run.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..config import LLMConfig
from ..utils import extract_json, get_logger, retry

log = get_logger("sciatlas.llm")

THINKING_SYSTEM_PROMPT = (
    "Think carefully and use hidden step-by-step reasoning before answering. "
    "Do not reveal your chain-of-thought. "
    "Return only the final requested answer."
)


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        trace_path: str | Path | None = None,
    ) -> None:
        self.cfg = config or LLMConfig()
        self._client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.request_timeout,
        )
        self._thinking_mode_supported: bool | None = None
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_context: str | None = None
        self._trace_lock = threading.Lock()

    def set_trace_context(self, context: str | None) -> None:
        self._trace_context = context

    def _append_trace(
        self,
        *,
        mode: str,
        prompt: str,
        system: str | None,
        model: str,
        temperature: float,
        response: str | None,
        parsed_json: Any = None,
        error: str | None = None,
    ) -> None:
        if self._trace_path is None:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": self._trace_context,
            "mode": mode,
            "model": model,
            "temperature": temperature,
            "system": system,
            "prompt": prompt,
            "response": response,
            "parsed_json": parsed_json,
            "error": error,
        }
        with self._trace_lock:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _chat_once(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float | None,
        model: str | None,
    ) -> tuple[str, str, float]:
        system_message = THINKING_SYSTEM_PROMPT
        if system:
            system_message = f"{THINKING_SYSTEM_PROMPT}\n\n{system}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        resolved_model = model or self.cfg.model
        resolved_temperature = self.cfg.temperature if temperature is None else temperature
        request_kwargs = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temperature,
        }
        if self._thinking_mode_supported is not False:
            try:
                resp = self._client.chat.completions.create(
                    **request_kwargs,
                    extra_body={"thinking": {"type": "enabled"}},
                )
                self._thinking_mode_supported = True
                return resp.choices[0].message.content or "", system_message, resolved_temperature
            except Exception as exc:
                self._thinking_mode_supported = False
                log.warning("LLM thinking mode unsupported, falling back: %s", exc)
        resp = self._client.chat.completions.create(**request_kwargs)
        return resp.choices[0].message.content or "", system_message, resolved_temperature

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        def _do() -> str:
            raw, system_message, resolved_temperature = self._chat_once(
                prompt,
                system=system,
                temperature=temperature,
                model=model,
            )
            self._append_trace(
                mode="chat",
                prompt=prompt,
                system=system_message,
                model=model or self.cfg.model,
                temperature=resolved_temperature,
                response=raw,
            )
            return raw

        try:
            return retry(
                _do,
                max_retries=self.cfg.max_retries,
                description=f"LLM chat ({model or self.cfg.model})",
            )
        except Exception as exc:
            self._append_trace(
                mode="chat",
                prompt=prompt,
                system=system,
                model=model or self.cfg.model,
                temperature=self.cfg.temperature if temperature is None else temperature,
                response=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> Any:
        """Call the model and parse its response as JSON.

        Retries the *whole* call (not just parsing) so a malformed response is
        re-generated rather than silently dropped.
        """

        def _do() -> Any:
            raw, system_message, resolved_temperature = self._chat_once(
                prompt,
                system=system,
                temperature=temperature,
                model=model,
            )
            try:
                parsed = extract_json(raw)
            except Exception as exc:
                self._append_trace(
                    mode="chat_json",
                    prompt=prompt,
                    system=system_message,
                    model=model or self.cfg.model,
                    temperature=resolved_temperature,
                    response=raw,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._append_trace(
                mode="chat_json",
                prompt=prompt,
                system=system_message,
                model=model or self.cfg.model,
                temperature=resolved_temperature,
                response=raw,
                parsed_json=parsed,
            )
            return parsed

        try:
            return retry(
                _do,
                max_retries=self.cfg.max_retries,
                retry_on=(ValueError,),
                description="LLM chat_json parse",
            )
        except Exception as exc:
            self._append_trace(
                mode="chat_json",
                prompt=prompt,
                system=system,
                model=model or self.cfg.model,
                temperature=self.cfg.temperature if temperature is None else temperature,
                response=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
