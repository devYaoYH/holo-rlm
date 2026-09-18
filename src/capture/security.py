"""Fail-closed secret and remote-URL checks for local trajectory artifacts."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

SECRET_PATTERNS = {
    "authorization header": re.compile(r"(?i)\bauthorization\s*[:=]\s*[^\s,}\]]+"),
    "bearer token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    "cookie": re.compile(r"(?i)\b(?:set-cookie|cookie)\s*[:=]"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    "generic API key": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{12,}"),
}
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class UnsafeCapture(ValueError):
    pass


def scan_text(text: str, *, source: str) -> None:
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            raise UnsafeCapture(f"{source}: rejected possible {label}")
    for raw_url in URL_PATTERN.findall(text):
        url = raw_url.rstrip(".,);]")
        host = urlparse(url).hostname
        if host not in LOCAL_HOSTS:
            raise UnsafeCapture(f"{source}: non-local URL is forbidden: {url}")


def scan_payload(payload: Any, *, source: str) -> None:
    scan_text(json.dumps(payload, sort_keys=True, default=str), source=source)
