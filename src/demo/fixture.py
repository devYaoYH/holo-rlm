"""In-process fixture lifecycle and its deterministic action API."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from apps.booking_fixture.server import FixtureServer, FixtureState


class FixtureClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=5, trust_env=False)

    def close(self) -> None:
        self._http.close()

    def health(self) -> dict:
        response = self._http.get("/api/health")
        response.raise_for_status()
        return response.json()

    def config(self) -> dict:
        response = self._http.get("/api/config")
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        response = self._http.get("/api/state")
        response.raise_for_status()
        return response.json()

    def reset(self, seed: int, variant: int = 0) -> dict:
        response = self._http.post("/api/reset", json={"seed": seed, "variant": variant})
        response.raise_for_status()
        return response.json()

    def apply(self, action: dict) -> dict:
        if action["action"] == "scroll":
            before = self.state()
            response = self._http.post("/api/scroll", json={"scroll_y": before["scroll_y"] + action["delta_y"]})
        elif action["action"] in {"click", "wait"}:
            response = self._http.post("/api/action", json=action)
        elif action["action"] == "finish":
            return self.state()
        else:
            raise ValueError(f"unsupported action: {action['action']}")
        response.raise_for_status()
        return response.json()


@contextmanager
def running_fixture(
    host: str = "127.0.0.1",
    port: int = 0,
    seed: int = 0,
    variant: int = 0,
    scenario_config: dict | None = None,
) -> Iterator[FixtureClient]:
    server = FixtureServer((host, port), FixtureState(), scenario_config=scenario_config)
    server.state.reset(seed, variant)
    thread = threading.Thread(target=server.serve_forever, name="booking-fixture", daemon=True)
    thread.start()
    actual_port = server.server_address[1]
    client = FixtureClient(f"http://{host}:{actual_port}")
    try:
        client.health()
        yield client
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
