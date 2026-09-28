"""Serve trace tensors lazily to a local, read-only browser viewer.

Run: python -m apps.trace_viewer.server --root data
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def image_spans(path: Path, positions: dict, model: dict) -> list[dict]:
    try:
        ids = np.load(path / "input_ids.npy", allow_pickle=False).reshape(-1)
    except (OSError, ValueError):
        return []
    image_id = positions.get("image_token_id")
    if image_id is None:
        return []
    indices = np.flatnonzero(ids == image_id)
    runs = []
    if len(indices):
        cuts = np.flatnonzero(np.diff(indices) != 1) + 1
        for run in np.split(indices, cuts):
            runs.append((int(run[0]), int(run[-1]) + 1))
    grid_info = load_json(path / "vision_inputs.json").get("image_grid_thw", {})
    grids = grid_info.get("values", []) if isinstance(grid_info, dict) else []
    merge = model.get("vision_config", {}).get("spatial_merge_size", 2)
    images = sorted(item.name for item in path.iterdir() if item.name.startswith("model-input-") and item.suffix.lower() in {".png", ".jpg", ".jpeg"})
    result = []
    for index, (start, end) in enumerate(runs):
        rows = cols = None
        if index < len(grids) and isinstance(grids[index], list) and len(grids[index]) == 3 and merge:
            t, h, w = grids[index]
            if t == 1 and h % merge == 0 and w % merge == 0 and (h // merge) * (w // merge) == end - start:
                rows, cols = h // merge, w // merge
        result.append({"index": index, "start": start, "end": end, "rows": rows, "cols": cols,
                       "file": images[index] if index < len(images) else None})
    return result


def trace_metadata(path: Path) -> dict:
    model = load_json(path / "model.json")
    positions = load_json(path / "positions.json")
    pieces = load_json(path / "generated_tokens.json")
    logprobs = load_json(path / "token_logprobs.json")
    token_ids = pieces.get("token_ids", [])
    if not token_ids:
        try:
            token_ids = np.load(path / "generated_ids.npy", allow_pickle=False).reshape(-1).tolist()
        except (OSError, ValueError):
            token_ids = []
    logrows = logprobs.get("tokens", [])
    tokens = [{"index": i, "id": int(token_id),
               "piece": pieces.get("tokens", [])[i] if i < len(pieces.get("tokens", [])) else str(token_id),
               "text": logrows[i].get("text") if i < len(logrows) else None,
               "logprob": logrows[i].get("logprob_nats") if i < len(logrows) else None}
              for i, token_id in enumerate(token_ids)]
    full = model.get("attention_layer_indices", [])
    if not full and model.get("model_type") == "qwen3_5":
        # Early traces omitted this metadata. The Qwen3.5 32-block config in
        # this repository uses full attention every fourth block.
        full = list(range(3, 32, 4))
    blocks = max(32, max(full, default=31) + 1)
    checkpoint = Path(model.get("model_path", "")).name if model.get("model_path") else None
    return {"id": path.name, "checkpoint": checkpoint,
            "model_id": model.get("model_id") or load_json(path / "request.json").get("model"),
            "model_revision": model.get("model_revision"), "model_type": model.get("model_type"),
            "device": model.get("device"), "prompt_tokens": positions.get("prompt_token_count"),
            "generated_tokens": positions.get("generated_token_count"), "tokens": tokens,
            "completion": load_json(path / "completion.json").get("text", ""),
            "full_attention_layers": full, "blocks": blocks, "images": image_spans(path, positions, model),
            "has_hidden": (path / "hidden_state_last_query_rows.npz").is_file(),
            "has_attention": (path / "attention_last_query_rows.npz").is_file(),
            "has_logprobs": bool(logrows)}


def attention_key(archive: np.lib.npyio.NpzFile, step: int, block: int, full_layers: list[int]) -> str | None:
    # Older capture versions stored the tuple ordinal; newer ones may use physical block IDs.
    direct = f"step_{step:03d}_layer_{block:03d}"
    ordinal = f"step_{step:03d}_layer_{full_layers.index(block):03d}" if block in full_layers else None
    names = set(archive.files)
    if ordinal and ordinal in names and all(f"step_{step:03d}_layer_{x:03d}" in names for x in range(len(full_layers))):
        return ordinal
    return direct if direct in names else None


def clean_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def step_data(path: Path, step: int, block: int, head: int) -> dict:
    meta = trace_metadata(path)
    if step < 0 or step >= len(meta["tokens"]) or block < 0 or block >= meta["blocks"]:
        raise ValueError("step or block out of range")
    hidden = []
    vector = None
    input_vector = None
    delta_vector = None
    delta_norm = None
    hidden_path = path / "hidden_state_last_query_rows.npz"
    if hidden_path.exists():
        with np.load(hidden_path, allow_pickle=False) as archive:
            for layer in range(meta["blocks"] + 1):
                key = f"step_{step:03d}_layer_{layer:03d}"
                if key not in archive.files:
                    hidden.append(None)
                    continue
                values = archive[key].astype(np.float32, copy=False).reshape(-1)
                norm = float(np.linalg.norm(values))
                hidden.append({"layer": layer, "norm": clean_float(norm),
                               "mean_abs": clean_float(np.mean(np.abs(values))),
                               "max_abs": clean_float(np.max(np.abs(values)))})
                if layer == block:
                    input_vector = values
                if layer == block + 1:
                    vector = values
    if vector is not None and input_vector is not None and vector.shape == input_vector.shape:
        delta = vector - input_vector
        delta_vector = [clean_float(value) for value in delta]
        delta_norm = clean_float(np.linalg.norm(delta))
    if vector is not None:
        vector = [clean_float(value) for value in vector]
    attention = None
    attention_path = path / "attention_last_query_rows.npz"
    if block in meta["full_attention_layers"] and attention_path.exists():
        with np.load(attention_path, allow_pickle=False) as archive:
            key = attention_key(archive, step, block, meta["full_attention_layers"])
            if key:
                matrix = archive[key].astype(np.float32, copy=False)
                if matrix.ndim == 2 and (-1 <= head < matrix.shape[0]):
                    row = matrix.mean(axis=0) if head == -1 else matrix[head]
                    images = []
                    for image in meta["images"]:
                        fragment = row[image["start"]:image["end"]]
                        images.append({**image, "mass": clean_float(np.sum(fragment)),
                                       "values": [clean_float(value) for value in fragment]})
                    prompt_count = meta["prompt_tokens"] or 0
                    indices = np.argsort(row)[-24:][::-1]
                    top = []
                    for index in indices:
                        position = int(index)
                        image_index = next((image["index"] for image in meta["images"] if image["start"] <= position < image["end"]), None)
                        generated_index = position - prompt_count if position >= prompt_count else None
                        top.append({"position": position, "weight": clean_float(row[index]),
                                    "image": image_index, "generated_index": generated_index})
                    attention = {"heads": int(matrix.shape[0]), "selected_head": head,
                                 "key_count": len(row), "row_sum": clean_float(np.sum(row)),
                                 "prompt_mass": clean_float(np.sum(row[:prompt_count])),
                                 "generation_mass": clean_float(np.sum(row[prompt_count:])),
                                 "images": images, "top": top}
    return {"step": step, "block": block, "head": head, "query_position": (meta["prompt_tokens"] or 0) + step - 1,
            "token": meta["tokens"][step], "hidden": hidden, "vector": vector,
            "delta_vector": delta_vector, "delta_norm": delta_norm, "attention": attention}


def make_handler(paths: dict[str, Path]):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            route = urlparse(self.path)
            parts = [part for part in route.path.split("/") if part]
            try:
                if route.path in ("/", "/index.html"):
                    return self.reply(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if route.path in ("/app.js", "/style.css"):
                    name = route.path[1:]
                    return self.reply(200, (STATIC / name).read_bytes(), "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8")
                if parts == ["api", "traces"]:
                    payload = [{"id": key, "source": str(value.parent), "hidden": (value / "hidden_state_last_query_rows.npz").exists(),
                                "logprobs": (value / "token_logprobs.json").exists()} for key, value in paths.items()]
                    return self.reply(200, json.dumps(payload).encode(), "application/json")
                if len(parts) >= 3 and parts[:2] == ["api", "trace"] and parts[2] in paths:
                    path = paths[parts[2]]
                    if len(parts) == 3:
                        value = trace_metadata(path)
                    elif len(parts) == 5 and parts[3] == "step":
                        query = parse_qs(route.query)
                        value = step_data(path, int(parts[4]), int(query.get("block", [3])[0]), int(query.get("head", [-1])[0]))
                    elif len(parts) == 5 and parts[3] == "image" and parts[4] in {image["file"] for image in trace_metadata(path)["images"]}:
                        name = parts[4]
                        mime = "image/png" if name.endswith(".png") else "image/jpeg"
                        return self.reply(200, (path / name).read_bytes(), mime)
                    else:
                        raise ValueError("unknown route")
                    return self.reply(200, json.dumps(value, allow_nan=False).encode(), "application/json")
            except (ValueError, OSError, KeyError, AttributeError) as exc:
                return self.reply(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            self.reply(404, b"Not found", "text/plain; charset=utf-8")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data", help="Trace directory or parent directory")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = [root / "manifest.json"] if (root / "manifest.json").is_file() else root.rglob("manifest.json")
    trace_dirs = [manifest.parent for manifest in manifests
                  if manifest.parent.name.startswith("trace-") and (root != ROOT / "data" or "hf-stage" not in manifest.relative_to(root).parts)]
    trace_dirs.sort(key=lambda item: (
        not ((item / "hidden_state_last_query_rows.npz").is_file()
             and load_json(item / "model.json").get("model_revision")
             and isinstance(load_json(item / "vision_inputs.json").get("image_grid_thw"), dict)),
        not ((item / "hidden_state_last_query_rows.npz").is_file() and load_json(item / "model.json").get("model_revision")),
        not (item / "hidden_state_last_query_rows.npz").is_file(), item.name,
    ))
    paths = {path.name: path for path in trace_dirs}
    if not paths:
        parser.error(f"no trace manifests under {root}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(paths))
    print(f"Trace viewer: http://127.0.0.1:{args.port} ({len(paths)} traces)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
