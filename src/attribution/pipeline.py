"""Project focused attention rows back onto multimodal image-token patches."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_ATTENTION_KEY = re.compile(r"^step_(\d+)_layer_(\d+)$")
_VALUE_NORM_KEY = re.compile(r"^layer_(\d+)$")
_PARAMETER_SPAN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_FIELD_SPAN = re.compile(
    r'"(?P<parameter>(?:\\.|[^"\\])+)"\s*:\s*'
    r'(?P<value>"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
)


class AttributionError(ValueError):
    """Raised when a trace cannot be mapped to its input-frame patch grids."""


@dataclass(frozen=True)
class PatchLayout:
    """The merged visual-token grid represented by a contiguous prompt span."""

    rows: int
    columns: int
    temporal: int
    merge_size: int
    image_positions: np.ndarray
    source: str
    raw_grid_thw: tuple[int, int, int] | None

    @property
    def cells(self) -> int:
        return self.rows * self.columns


@dataclass(frozen=True)
class GeneratedSpan:
    """A semantic generated-token span that can be averaged into one map."""

    id: str
    label: str
    parameter: str
    value: str
    steps: tuple[int, ...]


@dataclass(frozen=True)
class InputFrameAttribution:
    """Attribution tensors for one image-token span in a multimodal prompt."""

    index: int
    image_path: Path
    image_size: tuple[int, int]
    layout: PatchLayout
    maps: np.ndarray
    value_weighted_maps: np.ndarray | None
    rollout_maps: np.ndarray | None
    value_weighted_rollout_maps: np.ndarray | None


@dataclass(frozen=True)
class AttentionAttribution:
    """Generated-token attribution projected independently onto every input frame."""

    trace_path: Path
    frames: tuple[InputFrameAttribution, ...]
    steps: tuple[int, ...]
    layer_ordinals: tuple[int, ...]
    layer_indices: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_token_pieces: tuple[str | None, ...]
    generated_spans: tuple[GeneratedSpan, ...]
    warnings: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def image_path(self) -> Path:
        return self.frames[-1].image_path

    @property
    def image_size(self) -> tuple[int, int]:
        return self.frames[-1].image_size

    @property
    def layout(self) -> PatchLayout:
        return self.frames[-1].layout

    @property
    def maps(self) -> np.ndarray:
        return self.frames[-1].maps

    @property
    def value_weighted_maps(self) -> np.ndarray | None:
        return self.frames[-1].value_weighted_maps

    @property
    def rollout_maps(self) -> np.ndarray | None:
        return self.frames[-1].rollout_maps

    @property
    def value_weighted_rollout_maps(self) -> np.ndarray | None:
        return self.frames[-1].value_weighted_rollout_maps

    @property
    def head_count(self) -> int:
        return int(self.maps.shape[2])

    def aggregate(
        self,
        step: int,
        *,
        layer_ordinal: int | None = None,
        head: int | None = None,
        method: str = "attention",
        frame_index: int = -1,
    ) -> np.ndarray:
        """Average a selected direct or value-norm-corrected attention slice."""

        try:
            frame = self.frames[frame_index]
        except IndexError as exc:
            raise AttributionError(f"trace has no input frame {frame_index}") from exc
        try:
            step_axis = self.steps.index(step)
        except ValueError as exc:
            raise AttributionError(f"trace has no captured attention for generation step {step}") from exc
        if method == "attention":
            maps = frame.maps
        elif method == "value_norm":
            if frame.value_weighted_maps is None:
                raise AttributionError("this trace did not capture value-vector norms")
            maps = frame.value_weighted_maps
        elif method == "rollout":
            if frame.rollout_maps is None:
                raise AttributionError("this trace did not capture prompt matrices for cross-layer rollout")
            if layer_ordinal is not None or head is not None:
                raise AttributionError("cross-layer rollout is already aggregated across layers and heads")
            return frame.rollout_maps[step_axis]
        elif method == "value_norm_rollout":
            if frame.value_weighted_rollout_maps is None:
                raise AttributionError("this trace cannot compute value-norm cross-layer rollout")
            if layer_ordinal is not None or head is not None:
                raise AttributionError("cross-layer rollout is already aggregated across layers and heads")
            return frame.value_weighted_rollout_maps[step_axis]
        else:
            raise AttributionError(f"unknown attribution method: {method}")
        values = maps[step_axis]
        if layer_ordinal is not None:
            try:
                layer_axis = self.layer_ordinals.index(layer_ordinal)
            except ValueError as exc:
                raise AttributionError(f"trace has no captured attention layer {layer_ordinal}") from exc
            values = values[layer_axis : layer_axis + 1]
        if head is not None:
            if not 0 <= head < int(frame.maps.shape[2]):
                raise AttributionError(f"head {head} is outside 0..{self.head_count - 1}")
            values = values[:, head : head + 1]
        return np.nanmean(values, axis=(0, 1))

    def aggregate_steps(
        self,
        steps: tuple[int, ...],
        *,
        layer_ordinal: int | None = None,
        head: int | None = None,
        method: str = "attention",
        frame_index: int = -1,
    ) -> np.ndarray:
        """Average maps for a non-empty generated-token target span."""

        if not steps:
            raise AttributionError("cannot aggregate an empty generated-token span")
        if len(set(steps)) != len(steps):
            raise AttributionError("generated-token span contains duplicate steps")
        maps = [
            self.aggregate(
                step,
                layer_ordinal=layer_ordinal,
                head=head,
                method=method,
                frame_index=frame_index,
            )
            for step in steps
        ]
        return np.mean(np.stack(maps), axis=0, dtype=np.float32)

    def causal_difference(
        self,
        target_steps: tuple[int, ...],
        *,
        layer_ordinal: int | None = None,
        head: int | None = None,
        method: str = "attention",
        frame_index: int = -1,
    ) -> np.ndarray:
        """Subtract only maps generated strictly before a target span begins.

        The returned map is signed. Positive values are target-specific excess
        attribution; negative values indicate below-baseline attribution. A
        target beginning at the first captured step has no causal reference and
        therefore raises instead of silently using future tokens.
        """

        if not target_steps:
            raise AttributionError("cannot baseline an empty generated-token span")
        try:
            axes = tuple(self.steps.index(step) for step in target_steps)
        except ValueError as exc:
            raise AttributionError("target span contains an uncaptured generation step") from exc
        if tuple(sorted(axes)) != axes or len(set(axes)) != len(axes):
            raise AttributionError("target steps must be unique and ordered by generation")
        start_axis = axes[0]
        if start_axis == 0:
            raise AttributionError("target span has no previous generated tokens for a causal baseline")
        target = self.aggregate_steps(
            target_steps,
            layer_ordinal=layer_ordinal,
            head=head,
            method=method,
            frame_index=frame_index,
        )
        baseline = self.aggregate_steps(
            self.steps[:start_axis],
            layer_ordinal=layer_ordinal,
            head=head,
            method=method,
            frame_index=frame_index,
        )
        return target - baseline


def resolve_trace_path(path: Path, *, trace_root: Path | None = None, trace_index: int = 0) -> Path:
    """Resolve either a trace directory or a trajectory bundle to one trace directory."""

    candidate = path.expanduser().resolve()
    if (candidate / "attention_last_query_rows.npz").is_file():
        return candidate
    annotations_path = candidate / "annotations.json"
    if not annotations_path.is_file():
        raise AttributionError(f"not an instrumented trace or trajectory bundle: {candidate}")
    annotations = json.loads(annotations_path.read_text())
    trace_ids = annotations.get("instrumented_trace_ids")
    if not isinstance(trace_ids, list) or not trace_ids:
        raise AttributionError(f"trajectory has no instrumented trace IDs: {candidate}")
    if not 0 <= trace_index < len(trace_ids):
        raise AttributionError(f"trace index {trace_index} is outside 0..{len(trace_ids) - 1}")
    root = trace_root or candidate.parents[2] / "traces"
    root = root.expanduser().resolve()
    trace_path = (root / str(trace_ids[trace_index])).resolve()
    if trace_path != root and root not in trace_path.parents:
        raise AttributionError("trajectory trace ID resolves outside the configured trace root")
    if not (trace_path / "attention_last_query_rows.npz").is_file():
        raise AttributionError(f"attention trace not found: {trace_path}")
    return trace_path


def load_attribution(trace_path: Path) -> AttentionAttribution:
    """Load captured attention rows and project their image-key span onto a 2-D grid."""

    trace_path = trace_path.expanduser().resolve()
    required = ("input_ids.npy", "positions.json", "attention_last_query_rows.npz")
    missing = [name for name in required if not (trace_path / name).is_file()]
    if missing:
        raise AttributionError(f"trace is missing required artifacts: {', '.join(missing)}")

    positions = json.loads((trace_path / "positions.json").read_text())
    image_token_id = positions.get("image_token_id")
    if not isinstance(image_token_id, int):
        raise AttributionError("positions.json does not contain an integer image_token_id")
    input_ids = np.load(trace_path / "input_ids.npy", allow_pickle=False)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise AttributionError(f"only batch-size-one traces are supported; got input_ids shape {input_ids.shape}")
    all_image_positions = np.flatnonzero(input_ids[0] == image_token_id)
    if all_image_positions.size == 0:
        raise AttributionError(f"prompt contains no image token ID {image_token_id}")
    image_position_spans = _contiguous_spans(all_image_positions)

    image_paths = sorted(trace_path.glob("model-input-*.*"))
    if len(image_paths) != len(image_position_spans):
        raise AttributionError(
            f"captured {len(image_paths)} input images but found {len(image_position_spans)} image-token spans"
        )

    warnings: list[str] = []
    frames: list[InputFrameAttribution] = []
    steps: tuple[int, ...] | None = None
    layer_ordinals: tuple[int, ...] | None = None
    layer_indices: tuple[int, ...] | None = None
    for frame_index, (image_path, image_positions) in enumerate(
        zip(image_paths, image_position_spans, strict=True)
    ):
        with Image.open(image_path) as image:
            image_size = image.size
        layout = _resolve_patch_layout(
            trace_path,
            image_positions,
            image_size,
            warnings,
            frame_index=frame_index,
            frame_count=len(image_paths),
        )
        maps, frame_steps, frame_layer_ordinals = _load_attention_maps(trace_path, layout)
        if steps is None:
            steps = frame_steps
            layer_ordinals = frame_layer_ordinals
            layer_indices = _resolve_layer_indices(trace_path, layer_ordinals, warnings)
        elif frame_steps != steps or frame_layer_ordinals != layer_ordinals:
            raise AttributionError("captured attention step/layer axes differ between image spans")
        assert layer_ordinals is not None and layer_indices is not None
        value_weighted_maps = _load_value_weighted_maps(
            trace_path,
            layout,
            steps,
            layer_ordinals,
            layer_indices,
            maps.shape[2],
            warnings,
        )
        rollout_maps, value_weighted_rollout_maps = _load_rollout_maps(
            trace_path,
            layout,
            steps,
            layer_ordinals,
            layer_indices,
            maps.shape[2],
            warnings,
        )
        frames.append(
            InputFrameAttribution(
                index=frame_index,
                image_path=image_path,
                image_size=image_size,
                layout=layout,
                maps=maps,
                value_weighted_maps=value_weighted_maps,
                rollout_maps=rollout_maps,
                value_weighted_rollout_maps=value_weighted_rollout_maps,
            )
        )
    assert steps is not None and layer_ordinals is not None and layer_indices is not None
    generated_ids = _load_generated_ids(trace_path, steps)
    generated_pieces = _load_generated_pieces(trace_path, generated_ids)
    generated_spans = _generated_parameter_spans(steps, generated_pieces)
    return AttentionAttribution(
        trace_path=trace_path,
        frames=tuple(frames),
        steps=steps,
        layer_ordinals=layer_ordinals,
        layer_indices=layer_indices,
        generated_token_ids=generated_ids,
        generated_token_pieces=generated_pieces,
        generated_spans=generated_spans,
        warnings=tuple(warnings),
    )


def _contiguous_spans(positions: np.ndarray) -> tuple[np.ndarray, ...]:
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    return tuple(np.asarray(span, dtype=positions.dtype) for span in np.split(positions, boundaries))


def _resolve_patch_layout(
    trace_path: Path,
    image_positions: np.ndarray,
    image_size: tuple[int, int],
    warnings: list[str],
    *,
    frame_index: int,
    frame_count: int,
) -> PatchLayout:
    processor = _read_json(trace_path / "processor.json", {})
    preprocessing = processor.get("files", {}).get("preprocessor_config.json", {})
    merge_size = int(preprocessing.get("merge_size", 2))
    if merge_size < 1:
        raise AttributionError(f"invalid vision merge size: {merge_size}")

    vision_inputs = _read_json(trace_path / "vision_inputs.json", {})
    grid_value = vision_inputs.get("image_grid_thw")
    values = grid_value.get("values") if isinstance(grid_value, dict) else None
    if isinstance(values, list) and len(values) == frame_count and len(values[frame_index]) == 3:
        raw_grid = tuple(int(value) for value in values[frame_index])
        temporal, raw_rows, raw_columns = raw_grid
        if raw_rows % merge_size or raw_columns % merge_size:
            raise AttributionError(f"raw vision grid {raw_grid} is not divisible by merge size {merge_size}")
        rows, columns = raw_rows // merge_size, raw_columns // merge_size
        expected = temporal * rows * columns
        if expected != image_positions.size:
            raise AttributionError(
                f"vision grid {raw_grid} maps to {expected} tokens, but prompt contains {image_positions.size}"
            )
        if temporal != 1:
            raise AttributionError("temporal image grids are not supported by the frame viewer yet")
        return PatchLayout(
            rows=rows,
            columns=columns,
            temporal=temporal,
            merge_size=merge_size,
            image_positions=image_positions,
            source="captured_image_grid_thw",
            raw_grid_thw=raw_grid,
        )

    rows, columns = _factor_grid(int(image_positions.size), image_size)
    pixel_value = vision_inputs.get("pixel_values")
    pixel_shape = pixel_value.get("shape") if isinstance(pixel_value, dict) else pixel_value
    if (
        frame_count == 1
        and isinstance(pixel_shape, list)
        and pixel_shape
        and int(pixel_shape[0]) != image_positions.size * merge_size**2
    ):
        raise AttributionError(
            "cannot safely infer the patch grid: pixel_values and image-token counts disagree"
        )
    warnings.append(
        "image_grid_thw values were not retained by this older trace; the grid was inferred from image aspect ratio "
        "and validated against the merged patch count"
    )
    return PatchLayout(
        rows=rows,
        columns=columns,
        temporal=1,
        merge_size=merge_size,
        image_positions=image_positions,
        source="inferred_from_image_aspect_ratio",
        raw_grid_thw=(1, rows * merge_size, columns * merge_size),
    )


def _factor_grid(token_count: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    target_ratio = width / height
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, math.isqrt(token_count) + 1):
        if token_count % rows:
            continue
        columns = token_count // rows
        for candidate_rows, candidate_columns in ((rows, columns), (columns, rows)):
            ratio_error = abs(math.log((candidate_columns / candidate_rows) / target_ratio))
            candidates.append((ratio_error, candidate_rows, candidate_columns))
    if not candidates:
        raise AttributionError(f"cannot factor {token_count} image tokens into a patch grid")
    _, rows, columns = min(candidates)
    return rows, columns


def _load_attention_maps(
    trace_path: Path,
    layout: PatchLayout,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    archive = np.load(trace_path / "attention_last_query_rows.npz", allow_pickle=False)
    indexed: dict[tuple[int, int], np.ndarray] = {}
    for key in archive.files:
        match = _ATTENTION_KEY.fullmatch(key)
        if not match:
            continue
        step, layer = (int(value) for value in match.groups())
        row = np.asarray(archive[key], dtype=np.float32)
        if row.ndim != 2:
            raise AttributionError(f"{key} must have [heads, keys] shape; got {row.shape}")
        if int(layout.image_positions[-1]) >= row.shape[1]:
            raise AttributionError(f"{key} has only {row.shape[1]} keys, before the image span ends")
        indexed[(step, layer)] = row[:, layout.image_positions]
    if not indexed:
        raise AttributionError("attention archive contains no recognized step/layer arrays")

    steps = tuple(sorted({key[0] for key in indexed}))
    layers = tuple(sorted({key[1] for key in indexed}))
    head_counts = {array.shape[0] for array in indexed.values()}
    if len(head_counts) != 1:
        raise AttributionError(f"attention head count changes within the trace: {sorted(head_counts)}")
    head_count = head_counts.pop()
    result = np.full(
        (len(steps), len(layers), head_count, layout.rows, layout.columns),
        np.nan,
        dtype=np.float32,
    )
    for step_axis, step in enumerate(steps):
        for layer_axis, layer in enumerate(layers):
            values = indexed.get((step, layer))
            if values is not None:
                result[step_axis, layer_axis] = values.reshape(head_count, layout.rows, layout.columns)
    return result, steps, layers


def _load_value_weighted_maps(
    trace_path: Path,
    layout: PatchLayout,
    steps: tuple[int, ...],
    layer_ordinals: tuple[int, ...],
    layer_indices: tuple[int, ...],
    attention_heads: int,
    warnings: list[str],
) -> np.ndarray | None:
    """Apply A' = A*||V|| / sum(A*||V||) before selecting image keys.

    Qwen's grouped-query attention stores fewer KV heads than query heads, so
    each stored value-norm row is repeated across its associated query heads.
    The resulting rows remain probability distributions and can later be used
    as the per-layer matrices in a full rollout capture.
    """

    norm_path = trace_path / "value_norms.npz"
    if not norm_path.is_file():
        warnings.append(
            "value-vector norms were not captured; value-norm-corrected attribution is unavailable for this trace"
        )
        return None

    norm_archive = np.load(norm_path, allow_pickle=False)
    norms_by_layer: dict[int, np.ndarray] = {}
    for key in norm_archive.files:
        match = _VALUE_NORM_KEY.fullmatch(key)
        if not match:
            continue
        values = np.asarray(norm_archive[key], dtype=np.float32)
        if values.ndim != 2:
            warnings.append(f"{key} in value_norms.npz is not [kv_heads, keys]; correction is unavailable")
            return None
        norms_by_layer[int(match.group(1))] = values

    missing = [index for index in layer_indices if index not in norms_by_layer]
    if missing:
        warnings.append(
            f"value-vector norms are missing for transformer layers {missing}; correction is unavailable"
        )
        return None

    result = np.full(
        (len(steps), len(layer_ordinals), attention_heads, layout.rows, layout.columns),
        np.nan,
        dtype=np.float32,
    )
    attention_archive = np.load(trace_path / "attention_last_query_rows.npz", allow_pickle=False)
    for step_axis, step in enumerate(steps):
        for layer_axis, (ordinal, layer_index) in enumerate(zip(layer_ordinals, layer_indices, strict=True)):
            key = f"step_{step:03d}_layer_{ordinal:03d}"
            if key not in attention_archive:
                continue
            attention = np.asarray(attention_archive[key], dtype=np.float32)
            norms = norms_by_layer[layer_index]
            kv_heads, norm_keys = norms.shape
            if attention.shape[0] != attention_heads:
                warnings.append(f"{key} has an inconsistent attention-head count; correction is unavailable")
                return None
            if kv_heads < 1 or attention_heads % kv_heads:
                warnings.append(
                    f"layer {layer_index} has {kv_heads} KV heads for {attention_heads} attention heads; "
                    "correction is unavailable"
                )
                return None
            key_count = attention.shape[1]
            if norm_keys < key_count:
                warnings.append(
                    f"layer {layer_index} has value norms for {norm_keys} keys but {key} attends to {key_count}; "
                    "correction is unavailable"
                )
                return None
            expanded_norms = np.repeat(norms[:, :key_count], attention_heads // kv_heads, axis=0)
            weighted = attention * expanded_norms
            denominators = weighted.sum(axis=1, keepdims=True)
            corrected = np.divide(
                weighted,
                denominators,
                out=np.zeros_like(weighted),
                where=denominators > 0,
            )
            image_values = corrected[:, layout.image_positions]
            result[step_axis, layer_axis] = image_values.reshape(
                attention_heads,
                layout.rows,
                layout.columns,
            )
    return result


def _load_rollout_maps(
    trace_path: Path,
    layout: PatchLayout,
    steps: tuple[int, ...],
    layer_ordinals: tuple[int, ...],
    layer_indices: tuple[int, ...],
    attention_heads: int,
    warnings: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Roll attention through every captured full-attention block.

    The hybrid model's linear-attention blocks do not expose square softmax
    matrices, so rollout composes all conventional attention blocks in model
    order. Prompt rows come from the captured square matrices; generated rows
    come from the focused per-step archive.
    """

    prompt_path = trace_path / "attention_prompt_mean.npz"
    if not prompt_path.is_file():
        warnings.append("prompt attention matrices were not captured; cross-layer rollout is unavailable")
        return None, None
    if not steps or steps != tuple(range(steps[-1] + 1)):
        warnings.append("captured generation steps are not contiguous from zero; cross-layer rollout is unavailable")
        return None, None

    positions = _read_json(trace_path / "positions.json", {})
    prompt_count = positions.get("prompt_token_count") if isinstance(positions, dict) else None
    if not isinstance(prompt_count, int) or prompt_count < 1:
        warnings.append("prompt token count is unavailable; cross-layer rollout is unavailable")
        return None, None

    direct = _compute_rollout(
        trace_path,
        layout,
        steps,
        layer_ordinals,
        layer_indices,
        attention_heads,
        prompt_count,
        prompt_path,
        value_weighted=False,
    )

    weighted_prompt_path = trace_path / "attention_prompt_value_weighted_mean.npz"
    norm_path = trace_path / "value_norms.npz"
    weighted = None
    if weighted_prompt_path.is_file() and norm_path.is_file():
        weighted = _compute_rollout(
            trace_path,
            layout,
            steps,
            layer_ordinals,
            layer_indices,
            attention_heads,
            prompt_count,
            weighted_prompt_path,
            value_weighted=True,
        )
    else:
        warnings.append("value-weighted prompt matrices are unavailable; value-norm rollout is unavailable")
    return direct, weighted


def _compute_rollout(
    trace_path: Path,
    layout: PatchLayout,
    steps: tuple[int, ...],
    layer_ordinals: tuple[int, ...],
    layer_indices: tuple[int, ...],
    attention_heads: int,
    prompt_count: int,
    prompt_path: Path,
    *,
    value_weighted: bool,
) -> np.ndarray:
    prompt_archive = np.load(prompt_path, allow_pickle=False)
    attention_archive = np.load(trace_path / "attention_last_query_rows.npz", allow_pickle=False)
    norm_archive = np.load(trace_path / "value_norms.npz", allow_pickle=False) if value_weighted else None
    prompt_transitions: list[np.ndarray] = []
    generated_transitions: list[dict[int, np.ndarray]] = []

    for ordinal, layer_index in zip(layer_ordinals, layer_indices, strict=True):
        prompt_key = f"layer_{layer_index:03d}"
        if prompt_key not in prompt_archive:
            raise AttributionError(f"rollout archive is missing transformer layer {layer_index}")
        prompt_matrix = np.asarray(prompt_archive[prompt_key], dtype=np.float32)
        if prompt_matrix.shape != (prompt_count, prompt_count):
            raise AttributionError(
                f"{prompt_path.name}:{prompt_key} has shape {prompt_matrix.shape}, expected "
                f"({prompt_count}, {prompt_count})"
            )
        prompt_transitions.append(_row_normalize(prompt_matrix))

        layer_rows: dict[int, np.ndarray] = {}
        norms = None
        if value_weighted:
            assert norm_archive is not None
            if prompt_key not in norm_archive:
                raise AttributionError(f"value-norm archive is missing transformer layer {layer_index}")
            norms = np.asarray(norm_archive[prompt_key], dtype=np.float32)
        for step in steps[1:]:
            attention_key = f"step_{step:03d}_layer_{ordinal:03d}"
            if attention_key not in attention_archive:
                raise AttributionError(f"rollout requires missing attention row {attention_key}")
            attention = np.asarray(attention_archive[attention_key], dtype=np.float32)
            if attention.shape[0] != attention_heads:
                raise AttributionError(f"{attention_key} has an inconsistent attention-head count")
            if norms is None:
                row = attention.mean(axis=0)
            else:
                kv_heads, norm_keys = norms.shape
                if kv_heads < 1 or attention_heads % kv_heads or norm_keys < attention.shape[1]:
                    raise AttributionError(f"value norms cannot be aligned to {attention_key}")
                expanded = np.repeat(norms[:, : attention.shape[1]], attention_heads // kv_heads, axis=0)
                row = (attention * expanded).mean(axis=0)
            layer_rows[step] = _row_normalize(row[None, :])[0]
        generated_transitions.append(layer_rows)

    result = np.zeros((len(steps), layout.rows, layout.columns), dtype=np.float32)
    for step_axis, step in enumerate(steps):
        sequence_length = prompt_count + step
        query_position = sequence_length - 1
        influence = np.zeros(sequence_length, dtype=np.float32)
        influence[query_position] = 1.0
        for layer_axis in range(len(layer_indices) - 1, -1, -1):
            output = np.zeros_like(influence)
            prompt_values = influence[:prompt_count]
            output[:prompt_count] += 0.5 * (prompt_values @ prompt_transitions[layer_axis])
            output[:prompt_count] += 0.5 * prompt_values
            for generated_step in range(1, step + 1):
                position = prompt_count + generated_step - 1
                coefficient = influence[position]
                if coefficient == 0:
                    continue
                row = generated_transitions[layer_axis][generated_step]
                output[: row.shape[0]] += 0.5 * coefficient * row
                output[position] += 0.5 * coefficient
            influence = output
        image_values = influence[layout.image_positions]
        result[step_axis] = image_values.reshape(layout.rows, layout.columns)
    return result


def _row_normalize(values: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denominators = nonnegative.sum(axis=-1, keepdims=True)
    return np.divide(nonnegative, denominators, out=np.zeros_like(nonnegative), where=denominators > 0)


def _resolve_layer_indices(
    trace_path: Path,
    layer_ordinals: tuple[int, ...],
    warnings: list[str],
) -> tuple[int, ...]:
    model = _read_json(trace_path / "model.json", {})
    captured = model.get("attention_layer_indices")
    if isinstance(captured, list) and len(captured) == len(layer_ordinals):
        return tuple(int(value) for value in captured)

    model_path = model.get("model_path")
    config_path = Path(model_path) / "config.json" if isinstance(model_path, str) else None
    if config_path is not None and config_path.is_file():
        config = _read_json(config_path, {})
        layer_types = config.get("text_config", {}).get("layer_types", [])
        full_attention = tuple(index for index, value in enumerate(layer_types) if value == "full_attention")
        if len(full_attention) == len(layer_ordinals):
            warnings.append("transformer layer identities were recovered from the local checkpoint config")
            return full_attention

    warnings.append("transformer layer identities are unavailable; viewer labels use capture ordinals")
    return layer_ordinals


def _load_generated_ids(trace_path: Path, steps: tuple[int, ...]) -> tuple[int, ...]:
    path = trace_path / "generated_ids.npy"
    if not path.is_file():
        return tuple(-1 for _ in steps)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != 1:
        return tuple(-1 for _ in steps)
    return tuple(int(values[0, step]) if step < values.shape[1] else -1 for step in steps)


def _load_generated_pieces(trace_path: Path, token_ids: tuple[int, ...]) -> tuple[str | None, ...]:
    captured = _read_json(trace_path / "generated_tokens.json", None)
    if isinstance(captured, dict) and isinstance(captured.get("tokens"), list):
        tokens = captured["tokens"]
        return tuple(str(tokens[index]) if index < len(tokens) else None for index in range(len(token_ids)))

    model = _read_json(trace_path / "model.json", {})
    model_path = model.get("model_path")
    tokenizer_path = Path(model_path) / "tokenizer.json" if isinstance(model_path, str) else None
    if tokenizer_path is None or not tokenizer_path.is_file():
        return tuple(None for _ in token_ids)
    tokenizer = _read_json(tokenizer_path, {})
    vocab = tokenizer.get("model", {}).get("vocab", {})
    wanted = set(token_ids)
    by_id = {int(token_id): token for token, token_id in vocab.items() if int(token_id) in wanted}
    added = tokenizer.get("added_tokens", [])
    for token in added:
        token_id = token.get("id") if isinstance(token, dict) else None
        content = token.get("content") if isinstance(token, dict) else None
        if isinstance(token_id, int) and token_id in wanted and isinstance(content, str):
            by_id[token_id] = content
    return tuple(by_id.get(token_id) for token_id in token_ids)


def _generated_parameter_spans(
    steps: tuple[int, ...],
    pieces: tuple[str | None, ...],
) -> tuple[GeneratedSpan, ...]:
    """Recover XML-tool or JSON-field value spans and contributing token steps."""

    normalized_pieces = [
        (piece or "").replace("Ġ", " ").replace("Ċ", "\n")
        for piece in pieces
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in normalized_pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    text = "".join(normalized_pieces)
    captured_steps = set(steps)
    matches: list[tuple[int, str, str, tuple[int, int]]] = []
    matches.extend(
        (match.start(), match.group(1).strip(), match.group(2).strip(), match.span(2))
        for match in _PARAMETER_SPAN.finditer(text)
    )
    matches.extend(
        (
            match.start(),
            json.loads(f'"{match.group("parameter")}"'),
            _decode_json_scalar(match.group("value")),
            match.span("value"),
        )
        for match in _JSON_FIELD_SPAN.finditer(text)
    )

    spans: list[GeneratedSpan] = []
    for index, (_, parameter, value, (start, end)) in enumerate(sorted(matches)):
        token_steps = tuple(
            step
            for step, (piece_start, piece_end) in zip(steps, offsets, strict=True)
            if step in captured_steps and piece_end > start and piece_start < end
        )
        if not value or not token_steps:
            continue
        spans.append(
            GeneratedSpan(
                id=f"parameter-{index}",
                label=f"{parameter} = {value}",
                parameter=parameter,
                value=value,
                steps=token_steps,
            )
        )
    return tuple(spans)


def _decode_json_scalar(value: str) -> str:
    decoded = json.loads(value)
    if decoded is None:
        return "null"
    if decoded is True:
        return "true"
    if decoded is False:
        return "false"
    return str(decoded)


def _read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text())
