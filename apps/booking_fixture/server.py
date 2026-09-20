"""Local-only deterministic fixture server with an assertion state endpoint."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .generator import default_layout

ROOT = Path(__file__).resolve().parent
TARGET_ID = "harbor-lantern"
VIEWPORT = {"width": 1280, "height": 800}

_BASE_HOTELS = [
    ("harbor-light", "Harbor Light Inn", 184, "A quiet quay-side inn with blue rooms.", ["#5679a6", "#8db6c9"]),
    (TARGET_ID, "Harbor Lantern Hotel", 219, "Warm rooms beside the old lantern pier.", ["#b85f3c", "#e0a65d"]),
    ("lantern-harbor", "Lantern Harbor Suites", 241, "Suite-style rooms near the ferry terminal.", ["#5c6f91", "#8aa3ba"]),
    ("harbor-lantern-lodge", "Harbor Lantern Lodge", 198, "A compact lodge in the warehouse district.", ["#6b7450", "#abb182"]),
    ("mariner-house", "The Mariner House", 205, "A brick townhouse overlooking the basin.", ["#775a69", "#b6909f"]),
    ("signal-quay", "Signal Quay Rooms", 172, "Simple rooms close to the signal tower.", ["#3f7482", "#8cb5b9"]),
    ("anchor-yard", "Anchor Yard Hotel", 232, "Modern rooms around a sheltered courtyard.", ["#54616f", "#9ca7ae"]),
]


def fixture_config(seed: int, variant: int = 0) -> dict[str, Any]:
    """Return deterministic content; four seeds vary target position and density."""

    seed = seed % 4
    variant = variant % 5
    target_positions = [1, 2, 0, 3]
    items = list(_BASE_HOTELS)
    target = next(item for item in items if item[0] == TARGET_ID)
    items.remove(target)
    items.insert(target_positions[seed], target)
    if seed in {1, 3}:
        items.extend(
            [
                ("lantern-bay", "Lantern Bay Hotel", 187, "A seeded distractor near the market.", ["#76638a", "#bfa8ca"]),
                ("harbor-glow", "Harbor Glow Inn", 194, "A seeded distractor above the marina.", ["#826b43", "#c3ad7b"]),
            ]
        )
    target_names = [
        "Harbor Lantern Hotel",
        "Copper Lantern Hotel",
        "Harbor Beacon Hotel",
        "Lantern Quay Hotel",
        "Harbor Lantern House",
    ]
    target_name = target_names[variant]
    hotels = []
    for index, (hotel_id, name, price, description, colors) in enumerate(items):
        if hotel_id == TARGET_ID:
            name = target_name
        hotels.append(
            {
                "id": hotel_id,
                "name": name,
                "price": price + seed * 3,
                "description": description,
                "colors": colors,
                "rating": f"{8.1 + ((index + seed) % 8) / 10:.1f}",
                "initials": "".join(word[0] for word in name.split()[:2]),
                "index": index,
            }
        )
    viewport = dict(VIEWPORT)
    layout = default_layout(viewport)
    stride = layout["card_height"] + layout["card_gap"]
    content_bottom = layout["first_card_y"] + (len(hotels) - 1) * stride + layout["card_height"]
    return {
        "fixture_version": "booking-fixture-v1",
        "seed": seed,
        "variant": variant,
        "viewport": viewport,
        "layout": layout,
        "max_scroll": max(0, content_bottom - viewport["height"] + layout["header_height"] + 24),
        "target_id": TARGET_ID,
        "target_name": target_name,
        "initial_scroll_delta": [900, 1000, 1100, 1200, 800][variant],
        "expected_destination": f"/details/{TARGET_ID}",
        "hotels": hotels,
    }


@dataclass
class FixtureState:
    seed: int = 0
    variant: int = 0
    scroll_y: int = 0
    details_open: bool = False
    destination: str = "/"
    selected_hotel_id: str | None = None
    action_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, seed: int, variant: int = 0) -> dict[str, Any]:
        with self.lock:
            self.seed = seed % 4
            self.variant = variant % 5
            self.scroll_y = 0
            self.details_open = False
            self.destination = "/"
            self.selected_hotel_id = None
            self.action_count = 0
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "variant": self.variant,
            "scroll_y": self.scroll_y,
            "details_open": self.details_open,
            "destination": self.destination,
            "selected_hotel_id": self.selected_hotel_id,
            "action_count": self.action_count,
        }


class FixtureServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        state: FixtureState | None = None,
        scenario_config: dict[str, Any] | None = None,
    ) -> None:
        host = address[0]
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The fixture may bind only to localhost.")
        self.state = state or FixtureState()
        self.scenario_config = scenario_config
        super().__init__(address, FixtureHandler)

    def config(self) -> dict[str, Any]:
        return self.scenario_config or fixture_config(self.state.seed, self.state.variant)


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, sort_keys=True).encode(), "application/json", status)

    def _read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            query = parse_qs(parsed.query)
            if "seed" in query:
                self.server.state.reset(int(query["seed"][0]), int(query.get("variant", [0])[0]))
            self._send((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._json(self.server.config())
        elif path == "/api/state":
            self._json(self.server.state.snapshot())
        elif path == "/api/health":
            self._json({"status": "ok", "fixture_version": "booking-fixture-v1"})
        elif path.startswith("/details/"):
            hotel_id = path.removeprefix("/details/")
            allowed = {
                hotel["id"]
                for hotel in self.server.config()["hotels"]
            }
            if hotel_id not in allowed:
                self._json({"error": "unknown local hotel"}, HTTPStatus.NOT_FOUND)
                return
            with self.server.state.lock:
                self.server.state.details_open = True
                self.server.state.destination = path
                self.server.state.selected_hotel_id = hotel_id
                self.server.state.action_count += 1
            body = (ROOT / "details.html").read_text().replace(
                "Harbor Lantern Hotel",
                next(
                    h["name"]
                    for h in self.server.config()["hotels"]
                    if h["id"] == hotel_id
                ),
            )
            self._send(body.encode(), "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            if relative not in {"style.css", "app.js"}:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            file_path = ROOT / "static" / relative
            content_type = "text/css" if relative.endswith(".css") else "application/javascript"
            self._send(file_path.read_bytes(), content_type)
        else:
            self._json({"error": "fixture has no external or unknown route"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/api/reset":
            self._json(self.server.state.reset(int(payload.get("seed", 0)), int(payload.get("variant", 0))))
        elif path == "/api/scroll":
            with self.server.state.lock:
                self.server.state.scroll_y = max(
                    0,
                    min(int(payload.get("scroll_y", 0)), int(self.server.config().get("max_scroll", 2400))),
                )
                self.server.state.action_count += 1
                state = self.server.state.snapshot()
            self._json(state)
        elif path == "/api/action":
            action = payload.get("action")
            if action == "click":
                x, y = int(payload.get("x", -1)), int(payload.get("y", -1))
                config = self.server.config()
                layout = config["layout"]
                clicked_hotel = next(
                    (
                        hotel
                        for hotel in config["hotels"]
                        if layout["content_left"] + layout["button_x_offset"]
                        <= x
                        <= layout["content_left"] + layout["button_x_offset"] + layout["button_width"]
                        and layout["first_card_y"]
                        + hotel["index"] * (layout["card_height"] + layout["card_gap"])
                        - self.server.state.scroll_y
                        + layout["button_y_offset"]
                        <= y
                        <= layout["first_card_y"]
                        + hotel["index"] * (layout["card_height"] + layout["card_gap"])
                        - self.server.state.scroll_y
                        + layout["button_y_offset"]
                        + layout["button_height"]
                    ),
                    None,
                )
                hit = clicked_hotel is not None
                with self.server.state.lock:
                    self.server.state.action_count += 1
                if hit:
                    with self.server.state.lock:
                        self.server.state.details_open = True
                        self.server.state.selected_hotel_id = clicked_hotel["id"]
                        self.server.state.destination = f"/details/{clicked_hotel['id']}"
                self._json({**self.server.state.snapshot(), "click_hit": hit})
            elif action == "wait":
                with self.server.state.lock:
                    self.server.state.action_count += 1
                    state = self.server.state.snapshot()
                self._json(state)
            else:
                self._json({"error": "unsupported fixture action"}, HTTPStatus.BAD_REQUEST)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def serve(host: str = "127.0.0.1", port: int = 8765, seed: int = 0, variant: int = 0) -> None:
    server = FixtureServer((host, port))
    server.state.reset(seed, variant)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
