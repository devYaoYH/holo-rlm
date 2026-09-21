"""Self-contained viewer for raw, causal-token, and prompt-ensemble attribution."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contrast import PromptContrast, encode_signed_map, layer_head_statistics, spatial_metrics
from .pipeline import AttributionError


def write_prompt_contrast_viewer(
    contrast: PromptContrast,
    output_dir: Path,
    *,
    sample_id: str,
    target_instruction: str,
    bbox: tuple[float, float, float, float],
    predicted_click: tuple[float, float],
    correct: bool,
    format_valid: bool = True,
    source_label: str = "ScreenSpot-Pro",
    frame_bboxes: tuple[tuple[float, float, float, float] | None, ...] | None = None,
    excluded_controls: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    """Write one portable live-demo viewer and its machine-readable analysis."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir == contrast.target.trace_path or contrast.target.trace_path in output_dir.parents:
        raise AttributionError("contrast viewer output must stay outside immutable trace directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame_bboxes is None:
        frame_bboxes = tuple(
            bbox if index == contrast.target.frame_count - 1 else None
            for index in range(contrast.target.frame_count)
        )
    if len(frame_bboxes) != contrast.target.frame_count:
        raise AttributionError("frame bbox count must match the target trace frame count")
    frame_payloads = []
    mode_metrics: dict[str, list[dict[str, Any] | None]] = {
        "raw": [],
        "causal": [],
        "prompt_baseline": [],
        "prompt_difference": [],
    }
    for frame, frame_bbox in zip(contrast.target.frames, frame_bboxes, strict=True):
        maps = {
            "raw": contrast.raw_maps[frame.index],
            "causal": contrast.causal_maps[frame.index],
            "prompt_baseline": contrast.prompt_baseline_maps[frame.index],
            "prompt_difference": contrast.prompt_difference_maps[frame.index],
        }
        for name, values in maps.items():
            mode_metrics[name].append(
                spatial_metrics(values, frame_bbox, frame.image_size) if frame_bbox is not None else None
            )
        suffix = frame.image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        frame_payloads.append(
            {
                "index": frame.index,
                "label": f"Frame {frame.index}",
                "imageDataUrl": f"data:{mime};base64,{base64.b64encode(frame.image_path.read_bytes()).decode('ascii')}",
                "width": frame.image_size[0],
                "height": frame.image_size[1],
                "rows": frame.layout.rows,
                "columns": frame.layout.columns,
                "bbox": list(frame_bbox) if frame_bbox is not None else None,
                "isCurrent": frame.index == contrast.target.frame_count - 1,
                "maps": {name: encode_signed_map(values) for name, values in maps.items()},
            }
        )
    head_stats = layer_head_statistics(
        contrast,
        bbox,
        frame_index=contrast.target.frame_count - 1,
    )
    summary = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_label": source_label,
        "target_instruction": target_instruction,
        "correct": correct,
        "format_valid": format_valid,
        "bbox": list(bbox),
        "frame_bboxes": [list(value) if value is not None else None for value in frame_bboxes],
        "predicted_click": list(predicted_click),
        "method": contrast.method,
        "parameters": list(contrast.parameters),
        "target_steps": list(contrast.target_steps),
        "controls": list(contrast.control_instructions),
        "excluded_controls": list(excluded_controls),
        "trace_ids": {
            "target": contrast.target.trace_path.name,
            "controls": [control.trace_path.name for control in contrast.controls],
        },
        "stability": contrast.stability,
        "metrics": mode_metrics,
        "layer_head_statistics": head_stats,
        "interpretation": {
            "raw": "L1-normalized value-norm cross-layer rollout for x and y value tokens.",
            "causal": "Target coordinate map minus the mean of generated-token maps strictly before x begins.",
            "prompt_baseline": "Mean of separately L1-normalized coordinate maps from matched click instructions over the exact same image history.",
            "prompt_difference": "Target normalized coordinate map minus the same-image-history diverse-instruction mean. Red is above baseline; blue is below baseline.",
        },
        "caveat": "Attention rollout is a routing diagnostic, not proof that changing a highlighted patch would change the click.",
    }
    (output_dir / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    preview_paths: dict[str, str] = {}
    preview_maps = {
        "raw": contrast.raw_maps,
        "causal": contrast.causal_maps,
        "prompt-baseline": contrast.prompt_baseline_maps,
        "target-minus-prompt-baseline": contrast.prompt_difference_maps,
    }
    last_index = contrast.target.frame_count - 1
    for frame, frame_bbox in zip(contrast.target.frames, frame_bboxes, strict=True):
        frame_click = predicted_click if frame.index == last_index else None
        for name, maps in preview_maps.items():
            preview = output_dir / f"preview-frame-{frame.index:03d}-{name}.png"
            _write_preview(
                frame.image_path,
                maps[frame.index],
                bbox=frame_bbox,
                predicted_click=frame_click,
                output_path=preview,
            )
            preview_paths[f"frame-{frame.index:03d}-{name}"] = str(preview)
            if frame.index == last_index:
                legacy_preview = output_dir / f"preview-{name}.png"
                _write_preview(
                    frame.image_path,
                    maps[frame.index],
                    bbox=frame_bbox,
                    predicted_click=frame_click,
                    output_path=legacy_preview,
                )
                preview_paths[name] = str(legacy_preview)
    payload = {
        "sampleId": sample_id,
        "sourceLabel": source_label,
        "instruction": target_instruction,
        "correct": correct,
        "formatValid": format_valid,
        "bbox": list(bbox),
        "predictedClick": list(predicted_click),
        "method": contrast.method,
        "parameters": list(contrast.parameters),
        "frames": frame_payloads,
        "controls": list(contrast.control_instructions),
        "excludedControls": list(excluded_controls),
        "stability": contrast.stability,
        "metrics": mode_metrics,
        "layerStats": head_stats["layers"],
        "headStats": head_stats["heads"],
    }
    serialized = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    viewer = output_dir / "viewer.html"
    viewer.write_text(_HTML.replace("__PROMPT_CONTRAST_PAYLOAD__", serialized))
    return {
        "viewer": str(viewer),
        "analysis": str(output_dir / "analysis.json"),
        "previews": preview_paths,
        "sample_id": sample_id,
        "correct": correct,
        "controls": len(contrast.controls),
        "layers": len(contrast.target.layer_indices),
        "heads": contrast.target.head_count,
        "stability": contrast.stability,
    }


def _write_preview(
    image_path: Path,
    values: np.ndarray,
    *,
    bbox: tuple[float, float, float, float] | None,
    predicted_click: tuple[float, float] | None,
    output_path: Path,
) -> None:
    """Render a deterministic static counterpart of the live canvas overlay."""

    with Image.open(image_path) as source:
        base = source.convert("RGBA")
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    scaled = values / maximum if maximum > 0 else np.zeros_like(values)
    rows, columns = values.shape
    patch = Image.new("RGBA", (columns, rows))
    pixels = []
    for value in scaled.reshape(-1):
        magnitude = min(1.0, abs(float(value))) ** 0.62
        if value >= 0:
            color = (255, round(80 + 145 * (1 - magnitude)), 38)
            alpha = round(235 * magnitude * 0.42)
        else:
            color = (35, 145, 255)
            alpha = round(220 * magnitude * 0.42)
        pixels.append((*color, alpha))
    patch.putdata(pixels)
    overlay = patch.resize(base.size, resample=Image.Resampling.NEAREST)
    result = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(result)
    line_width = max(3, round(result.width / 900))
    if bbox is not None:
        draw.rectangle(bbox, outline=(50, 230, 161, 255), width=line_width)
    if predicted_click is not None:
        x, y = predicted_click
        arm = max(24, round(result.width / 85))
        radius = max(15, round(result.width / 120))
        halo_width = max(8, line_width * 3)
        marker_width = max(4, line_width + 2)
        marker_lines = ((x - arm, y, x + arm, y), (x, y - arm, x, y + arm))
        marker_ring = (x - radius, y - radius, x + radius, y + radius)
        for line in marker_lines:
            draw.line(line, fill=(8, 12, 18, 230), width=halo_width)
        draw.ellipse(marker_ring, outline=(8, 12, 18, 230), width=halo_width)
        for line in marker_lines:
            draw.line(line, fill=(255, 255, 255, 255), width=marker_width)
        draw.ellipse(marker_ring, outline=(255, 255, 255, 255), width=marker_width)
    result.convert("RGB").save(output_path, optimize=True)


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Instruction-specific attribution comparison</title>
<style>
:root{color-scheme:light dark;--bg:#f4f5f7;--panel:#fff;--fg:#17181b;--muted:#666b75;--line:#d9dce2;--accent:#6d43dc;--good:#16845b;--bad:#c2463b;--soft:#eee9ff}
@media(prefers-color-scheme:dark){:root{--bg:#101216;--panel:#1a1d22;--fg:#f3f4f6;--muted:#a7abb3;--line:#343943;--accent:#b8a2ff;--good:#62d5a5;--bad:#ff8d84;--soft:#30294b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:24px}h1{font-size:25px;margin:4px 0 6px}.eyebrow{color:var(--accent);font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.subtitle{max-width:950px;color:var(--muted)}.outcome{font-weight:700;color:var(--good)}.outcome.miss{color:var(--bad)}.stage{margin-top:18px;background:#07080a;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 18px 55px rgba(16,18,27,.14)}.frame-wrap{position:relative;line-height:0}.frame-wrap img{display:block;width:100%;height:auto}.frame-wrap canvas{position:absolute;inset:0;width:100%;height:100%}.stage-footer{display:flex;justify-content:space-between;gap:14px;background:#111318;color:#eceef1;padding:10px 13px;line-height:1.35}.stage-footer span:last-child{color:#aeb2bb;text-align:right}.controls{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}.control-grid{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:14px}.control label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}.control select,.control input[type=range]{width:100%}.control select{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:8px}.checks{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px}.explainer{margin:10px 0 0;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.metric{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px}.metric-label{color:var(--muted);font-size:10px;letter-spacing:.07em;text-transform:uppercase}.metric-value{font-size:20px;margin-top:2px;font-variant-numeric:tabular-nums}.analysis{display:grid;grid-template-columns:1.3fr 1fr;gap:14px;margin-top:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px;min-width:0}.panel h2{font-size:14px;margin:0 0 10px}.table-wrap{overflow:auto;max-height:390px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0;background:var(--panel)}.controls-list{margin:0;padding-left:20px}.controls-list li{margin:6px 0}.stability{margin-top:12px;color:var(--muted)}@media(max-width:900px){main{padding:14px}.analysis{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.control-grid{grid-template-columns:1fr}.stage-footer{align-items:flex-start;flex-direction:column}.stage-footer span:last-child{text-align:left}}@media(max-width:520px){.metrics{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<header><div class="eyebrow"><span id="source-label"></span> · local Holo 3.1 4B</div><h1>Instruction-specific visual attribution</h1><div class="subtitle"><span id="sample"></span> · <span id="instruction"></span> · <span id="outcome" class="outcome"></span></div></header>
<section class="stage"><div class="frame-wrap"><img id="frame" alt="GUI screenshot"><canvas id="overlay" role="img"></canvas></div><div class="stage-footer"><span id="mode-label"></span><span id="coordinates"></span></div></section>
<section class="controls"><div class="control-grid"><div class="control"><label for="frame-select">Input frame</label><select id="frame-select"></select></div><div class="control"><label for="mode">Attribution comparison</label><select id="mode"><option value="raw">Raw value-norm rollout</option><option value="causal">Causal previous-token difference</option><option value="prompt_baseline">Diverse-instruction baseline</option><option value="prompt_difference">Target minus diverse-instruction baseline</option></select></div><div class="control"><label for="opacity">Overlay opacity · <output id="opacity-output">48%</output></label><input id="opacity" type="range" min="0" max="100" value="48"></div><div class="control"><label for="head-sort">Per-head ranking</label><select id="head-sort"><option value="prompt_difference">Prompt difference</option><option value="causal">Causal difference</option><option value="raw">Raw attention</option></select></div></div><div class="checks"><label><input id="grid" type="checkbox"> Show patch boundaries</label><label><input id="bbox" type="checkbox" checked> Show ground-truth box</label><label><input id="click" type="checkbox" checked> Show predicted click</label></div><p id="explainer" class="explainer"></p></section>
<section class="metrics"><div class="metric"><div class="metric-label">Target lift</div><div id="lift" class="metric-value"></div></div><div class="metric"><div class="metric-label">Positive mass in target</div><div id="target-mass" class="metric-value"></div></div><div class="metric"><div class="metric-label">Peak distance</div><div id="distance" class="metric-value"></div></div><div class="metric"><div class="metric-label">Map entropy</div><div id="entropy" class="metric-value"></div></div></section>
<section class="analysis"><div class="panel"><h2>Layer and head target alignment</h2><div class="table-wrap"><table><thead><tr><th>Layer</th><th>Head</th><th>Target lift</th><th>Target mass</th><th>Peak distance</th></tr></thead><tbody id="head-table"></tbody></table></div></div><div class="panel"><h2>Same-image control instructions</h2><ol id="controls" class="controls-list"></ol><div id="excluded-controls" class="stability"></div><div id="stability" class="stability"></div><h2 style="margin-top:18px">Layer summary</h2><div class="table-wrap"><table><thead><tr><th>Layer</th><th>Mean lift</th><th>Best head</th><th>Best lift</th></tr></thead><tbody id="layer-table"></tbody></table></div></div></section>
</main><script>
const data=__PROMPT_CONTRAST_PAYLOAD__,$=id=>document.getElementById(id),image=$('frame'),canvas=$('overlay'),ctx=canvas.getContext('2d');
const state={frame:0,mode:'raw',opacity:.48,grid:false,bbox:true,click:true,headSort:'prompt_difference'};
const descriptions={raw:'Coordinate-token attribution after value-norm correction and full cross-layer rollout, normalized jointly across every retained image patch.',causal:'The raw coordinate map minus only generated-token maps that occurred before the x value began. Future output tokens never enter the baseline.',prompt_baseline:'The mean coordinate map from matched click instructions over this exact image and action history. Each control run is normalized before averaging.',prompt_difference:'Instruction-specific differential: target map minus the same-image-history control mean. Red is above baseline; blue is below baseline.'};
function decode(encoded){const raw=atob(encoded),bytes=new Uint8Array(raw.length),result=new Int8Array(raw.length);for(let i=0;i<raw.length;i++)result[i]=bytes[i];return result}
function colorSigned(value){const magnitude=Math.pow(Math.min(1,Math.abs(value)),.62);if(value>=0)return[255,80+Math.round(145*(1-magnitude)),38,Math.round(235*magnitude)];return[35,145,255,Math.round(220*magnitude)]}
function drawClickMarker(x,y){const arm=Math.max(24,canvas.width/85),radius=Math.max(15,canvas.width/120),halo=Math.max(8,canvas.width/320),inner=Math.max(4,canvas.width/650);const stroke=(color,width)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(x-arm,y);ctx.lineTo(x+arm,y);ctx.moveTo(x,y-arm);ctx.lineTo(x,y+arm);ctx.moveTo(x+radius,y);ctx.arc(x,y,radius,0,Math.PI*2);ctx.stroke()};stroke('rgba(8,12,18,.90)',halo);stroke('#fff',inner)}
function render(){const frame=data.frames[state.frame],encoded=frame.maps[state.mode],codes=decode(encoded.valuesI8),small=document.createElement('canvas');small.width=frame.columns;small.height=frame.rows;const sctx=small.getContext('2d'),pixels=sctx.createImageData(frame.columns,frame.rows);for(let i=0;i<codes.length;i++){const rgba=colorSigned(codes[i]/127),o=i*4;pixels.data[o]=rgba[0];pixels.data[o+1]=rgba[1];pixels.data[o+2]=rgba[2];pixels.data[o+3]=rgba[3]}sctx.putImageData(pixels,0,0);canvas.width=frame.width;canvas.height=frame.height;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.globalAlpha=state.opacity;ctx.drawImage(small,0,0,canvas.width,canvas.height);ctx.globalAlpha=1;if(state.grid){ctx.strokeStyle='rgba(255,255,255,.25)';ctx.lineWidth=Math.max(1,canvas.width/1600);for(let c=1;c<frame.columns;c++){const x=c*canvas.width/frame.columns;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}for(let r=1;r<frame.rows;r++){const y=r*canvas.height/frame.rows;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}}const annotated=Array.isArray(frame.bbox);if(state.bbox&&annotated){const [x1,y1,x2,y2]=frame.bbox;ctx.strokeStyle='#32e6a1';ctx.lineWidth=Math.max(3,canvas.width/900);ctx.strokeRect(x1,y1,x2-x1,y2-y1)}if(state.click&&frame.isCurrent){drawClickMarker(...data.predictedClick)}const m=data.metrics[state.mode][state.frame];$('lift').textContent=m?`${m.target_lift.toFixed(1)}x`:'—';$('target-mass').textContent=m?`${(m.target_mass*100).toFixed(2)}%`:'—';$('distance').textContent=m?`${(m.peak_distance_diagonal*100).toFixed(1)}% diag`:'—';$('entropy').textContent=m?m.normalized_entropy.toFixed(2):'—';$('mode-label').textContent=`${$('mode').selectedOptions[0].textContent} · ${frame.label}`;const patchText=`${frame.columns} x ${frame.rows} patches`;$('coordinates').textContent=frame.isCurrent?`white ring/cross: predicted click (${Math.round(data.predictedClick[0])}, ${Math.round(data.predictedClick[1])})${annotated?' · green box: target':' '} · ${patchText}`:annotated?`historical frame with the target visible · green box: target · ${patchText}`:`historical input frame before the target was visible · ${patchText}`;$('explainer').textContent=descriptions[state.mode];renderTables()}
function renderTables(){const key=state.headSort,rows=[...data.headStats].sort((a,b)=>b[key].target_lift-a[key].target_lift).slice(0,24);$('head-table').innerHTML=rows.map(row=>`<tr><td>${row.transformer_layer}</td><td>${row.head}</td><td>${row[key].target_lift.toFixed(1)}x</td><td>${(row[key].target_mass*100).toFixed(2)}%</td><td>${(row[key].peak_distance_diagonal*100).toFixed(1)}%</td></tr>`).join('');const metric=key==='prompt_difference'?'prompt':key;$('layer-table').innerHTML=data.layerStats.map(row=>`<tr><td>${row.transformer_layer}</td><td>${row[metric+'_mean_target_lift'].toFixed(1)}x</td><td>${key==='prompt_difference'?row.best_prompt_head:'—'}</td><td>${key==='prompt_difference'?row.best_prompt_target_lift.toFixed(1)+'x':'—'}</td></tr>`).join('')}
$('source-label').textContent=data.sourceLabel;$('sample').textContent=data.sampleId;$('instruction').textContent=`“${data.instruction}”`;$('outcome').textContent=data.correct?(data.formatValid?'OFFICIAL HIT':'VISUAL HIT · MALFORMED TOOL SYNTAX'):'GROUNDING MISS';if(!data.correct)$('outcome').classList.add('miss');data.controls.forEach(value=>{const li=document.createElement('li');li.textContent=value;$('controls').append(li)});if(data.excludedControls.length)$('excluded-controls').textContent=`Excluded ${data.excludedControls.length} control(s) missing a complete x/y token pair; see analysis.json.`;const mean=data.stability.mean_leave_one_out_cosine,min=data.stability.minimum_leave_one_out_cosine;$('stability').textContent=mean==null?'One control only; leave-one-out stability is unavailable.':`Leave-one-control-out cosine: mean ${mean.toFixed(3)}, minimum ${min.toFixed(3)}.`;data.frames.forEach((frame,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=frame.label;$('frame-select').append(option)});$('frame-select').disabled=data.frames.length===1;
$('frame-select').addEventListener('change',event=>{state.frame=Number(event.target.value);image.src=data.frames[state.frame].imageDataUrl});$('mode').addEventListener('change',event=>{state.mode=event.target.value;render()});$('opacity').addEventListener('input',event=>{state.opacity=Number(event.target.value)/100;$('opacity-output').textContent=`${event.target.value}%`;render()});$('head-sort').addEventListener('change',event=>{state.headSort=event.target.value;renderTables()});$('grid').addEventListener('change',event=>{state.grid=event.target.checked;render()});$('bbox').addEventListener('change',event=>{state.bbox=event.target.checked;render()});$('click').addEventListener('change',event=>{state.click=event.target.checked;render()});image.addEventListener('load',render);image.src=data.frames[0].imageDataUrl;
</script></body></html>'''
