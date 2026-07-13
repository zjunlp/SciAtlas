from __future__ import annotations

import os
import re
from typing import Iterable, Mapping


DEFAULT_CUDA_DEVICE_ENV_VARS = (
    "INNOEVAL_CUDA_DEVICES",
    "CUDA_DEVICES",
)


def normalize_torch_device(value: str | None) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"cpu", "mps"}:
        return lowered
    if lowered == "cuda":
        return default_torch_device()
    if re.fullmatch(r"\d+", lowered):
        return f"cuda:{lowered}"
    if re.fullmatch(r"cuda:\d+", lowered):
        return lowered
    return text


def parse_cuda_devices(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    devices: list[str] = []
    for token in str(raw_value).split(","):
        device = normalize_torch_device(token)
        if device:
            devices.append(device)
    return devices


def unique_cuda_devices(values: Iterable[str | None]) -> list[str]:
    devices: list[str] = []
    seen: set[str] = set()
    for value in values:
        for device in parse_cuda_devices(value):
            if device not in seen:
                seen.add(device)
                devices.append(device)
    return devices


def format_cuda_devices(devices: Iterable[str]) -> str:
    return ",".join(device for device in devices if device)


def first_configured_cuda_devices(
    *,
    env: Mapping[str, str] | None = None,
    env_vars: Iterable[str] = DEFAULT_CUDA_DEVICE_ENV_VARS,
) -> list[str]:
    values = env or os.environ
    for env_var in env_vars:
        devices = parse_cuda_devices(values.get(env_var))
        if devices:
            return devices
    return []


def default_torch_device(
    *,
    index: int = 0,
    env: Mapping[str, str] | None = None,
    env_vars: Iterable[str] = DEFAULT_CUDA_DEVICE_ENV_VARS,
) -> str:
    devices = first_configured_cuda_devices(env=env, env_vars=env_vars)
    if devices:
        return devices[index % len(devices)]
    try:
        import torch

        if torch.cuda.is_available():
            device_count = max(int(torch.cuda.device_count()), 1)
            return f"cuda:{index % device_count}"
        return "cpu"
    except Exception:
        return "cpu"


def device_at(devices: list[str], index: int) -> str | None:
    if not devices:
        return None
    return devices[index % len(devices)]
