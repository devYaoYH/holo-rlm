"""Static, self-contained HTML replay report generator."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .normalizer import read_jsonl


def render_report(bundle: Path) -> bytes:
    manifest = json.loads((bundle / "manifest.json").read_text()) if (bundle / "manifest.json").exists() else {}
    events = list(read_jsonl(bundle / "events.jsonl"))
    by_step: dict[int, dict[str, dict]] = {}
    for event in events:
        step = event.get("parent_step_ref")
        if isinstance(step, int):
            by_step.setdefault(step, {})[event["event_type"]] = event
    sections = []
    for step, grouped in sorted(by_step.items()):
        observation = grouped.get("observation", {})
        result = grouped.get("action_result", {})
        action = grouped.get("proposed_action", {}).get("data", {}).get("normalized")
        pre = next((a["path"] for a in observation.get("artifacts", []) if a.get("role") == "model_input"), None)
        post = next((a["path"] for a in result.get("artifacts", []) if a.get("role") == "post_action"), None)
        sections.append(
            f"<section><h2>Step {step}</h2><pre>{html.escape(json.dumps(action, indent=2))}</pre>"
            f"<div class='frames'>{_image(pre, 'Pre-action model input')}{_image(post, 'Post-action')}</div>"
            f"<pre>{html.escape(json.dumps(result.get('data', {}), indent=2))}</pre></section>"
        )
    task = html.escape(str(manifest.get("task_id", "trajectory")))
    body = "".join(sections)
    return f"""<!doctype html><meta charset='utf-8'><title>{task}</title>
<style>body{{font:15px system-ui;margin:32px;background:#f5f7fa;color:#172033}}section{{background:white;padding:22px;margin:20px 0;border-radius:10px}}.frames{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{max-width:100%;border:1px solid #cad3de}}pre{{overflow:auto;background:#eef2f6;padding:12px}}figcaption{{font-weight:700;margin-bottom:6px}}</style>
<h1>{task}</h1><p>Generated locally from exact hashed trajectory artifacts.</p>{body}""".encode()


def _image(path: str | None, label: str) -> str:
    if path is None:
        return f"<figure><figcaption>{html.escape(label)}</figcaption><em>not captured</em></figure>"
    return f"<figure><figcaption>{html.escape(label)}</figcaption><img src='{html.escape(path)}' alt='{html.escape(label)}'></figure>"
