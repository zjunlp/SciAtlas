from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path


def post_pdf(
    url: str,
    pdf_path: Path,
    fields: dict[str, str | None],
    retries: int = 8,
    backoff: float = 2.0,
    timeout_s: int = 600,
) -> str:
    boundary = f"----GROBIDBoundary{int(time.time() * 1000)}"
    pdf_bytes = pdf_path.read_bytes()
    body = bytearray()

    for name, value in fields.items():
        if value is None:
            continue
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
    )
    body.extend(pdf_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 503 and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

    raise RuntimeError(f"Exhausted retries calling {url}")
