#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_review_search import DmxJsonClient, DEFAULT_ENV_PATH
from generate_literature_review_report import load_codex_gpt_override


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Codex GPT override config and a minimal JSON response.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env, kept for compatibility.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens for the probe call.")
    args = parser.parse_args()

    override = load_codex_gpt_override()
    print("override_config=" + json.dumps(override, ensure_ascii=False, indent=2))

    base_url = override.get("base_url")
    model = override.get("model")
    api_key = override.get("api_key")
    wire_api = override.get("wire_api") or "responses"
    if not base_url or not model or not api_key:
        raise SystemExit("Incomplete override config: expected base_url, model, and api_key.")

    client = DmxJsonClient(
        env_path=Path(args.env).expanduser().resolve(),
        api_url=base_url,
        model=model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=0.0,
        use_env_proxy=False,
        api_key_override=api_key,
        wire_api=wire_api,
    )

    system_prompt = "Return strict JSON only."
    user_prompt = 'Return exactly this JSON object and nothing else: {"ok": true, "source": "gpt_override_test"}.'
    text = client.chat_json_text(system_prompt=system_prompt, user_prompt=user_prompt, label="gpt_override_test")
    print("raw_response=" + text)
    payload = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, label="gpt_override_test_parse")
    print("parsed_response=" + json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
