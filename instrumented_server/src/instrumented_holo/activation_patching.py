"""Reusable teacher-forced activation patching for matched multimodal requests.

The library separates three concerns that were previously fused in the
PowerPoint case-study script:

* exact construction of scored candidate suffixes;
* preparation and scoring of an arbitrary multimodal message history; and
* localized activation replacement or ablation hooks.

It deliberately does not decide which clean/corrupted pair is scientifically
valid.  Callers provide the two requests, the candidate completions, and any
image regions to map onto visual-token positions.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from PIL import Image, ImageChops

from .model import InstrumentedHolo, normalise_messages

ScoreReduction = Literal["sum", "mean"]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateSequence:
    """One teacher-forced completion and the character spans to score."""

    label: str
    text: str
    scored_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EncodedCandidate:
    """Tokenized candidate with offsets for the scored subset."""

    label: str
    token_ids: tuple[int, ...]
    scored_token_offsets: tuple[int, ...]


@dataclass(frozen=True)
class ImageRegion:
    """A pixel-space region on one image in message order."""

    image_index: int
    box: Box
    limit: int | None = None


@dataclass(frozen=True)
class PreparedCondition:
    """Language-model inputs after image encoding, ready for interventions."""

    name: str
    inputs_embeds: Any
    attention_mask: Any
    position_ids: Any
    input_ids: Any
    prompt_length: int
    candidates: tuple[EncodedCandidate, ...]
    image_positions: tuple[tuple[int, ...], ...]
    image_grids: tuple[tuple[int, int], ...]
    region_positions: Mapping[str, tuple[int, ...]]
    prediction_positions: tuple[tuple[int, ...], ...]
    scored_token_ids: tuple[tuple[int, ...], ...]

    @property
    def all_image_positions(self) -> tuple[int, ...]:
        return tuple(position for span in self.image_positions for position in span)


@dataclass(frozen=True)
class MarginScore:
    """Candidate log likelihoods and candidate-0 minus candidate-1 margin."""

    margin_nats: float
    candidate_log_prob_nats: tuple[float, float]
    reduction: ScoreReduction


def require_patch_alignment(clean: PreparedCondition, corrupted: PreparedCondition) -> None:
    """Reject clean/corrupted pairs whose token positions cannot be patched directly."""

    if tuple(clean.input_ids.shape) != tuple(corrupted.input_ids.shape):
        raise ValueError("clean and corrupted token sequences must have identical shapes")
    if clean.prediction_positions != corrupted.prediction_positions:
        raise ValueError("clean and corrupted scored-token positions must align")
    if tuple(map(len, clean.image_positions)) != tuple(map(len, corrupted.image_positions)):
        raise ValueError("clean and corrupted image-token spans must have identical lengths")
    if clean.image_grids != corrupted.image_grids:
        raise ValueError("clean and corrupted image grids must match")


def parse_box(value: str) -> Box:
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("boxes must have four comma-separated integers")
    try:
        validate_box(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return values


def validate_box(box: Box, image_size: tuple[int, int] | None = None) -> None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box maximums must be greater than minimums")
    if image_size is not None:
        width, height = image_size
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"box {box} lies outside the {width}x{height} image")


def coordinate_center(box: Box, image_size: tuple[int, int]) -> tuple[int, int]:
    """Return the rounded 0..1000 center used for teacher-forced clicks."""

    validate_box(box, image_size)
    x1, y1, x2, y2 = box
    width, height = image_size
    return round((x1 + x2) * 500 / width), round((y1 + y2) * 500 / height)


def swap_equal_tiles(image: Image.Image, first: Box, second: Box) -> Image.Image:
    """Swap disjoint, equal-sized rectangles with no resize or resampling."""

    validate_box(first, image.size)
    validate_box(second, image.size)
    if (first[2] - first[0], first[3] - first[1]) != (second[2] - second[0], second[3] - second[1]):
        raise ValueError("the target and distractor swap boxes must have identical dimensions")
    if max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(
        first[3], second[3]
    ):
        raise ValueError("swap boxes must not overlap")
    first_pixels = image.crop(first)
    second_pixels = image.crop(second)
    result = image.copy()
    result.paste(second_pixels, first[:2])
    result.paste(first_pixels, second[:2])
    return result


def changed_pixel_fraction(first: Image.Image, second: Image.Image) -> float:
    """Return the fraction of pixels with any changed channel."""

    if first.size != second.size:
        raise ValueError("images must have identical dimensions")
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    red, green, blue = difference.split()
    any_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed = sum(any_channel.point(lambda value: 1 if value else 0).histogram()[1:])
    return changed / (first.width * first.height)


def overlapping_patch_indices(
    box: Box,
    image_size: tuple[int, int],
    grid: tuple[int, int],
    *,
    limit: int | None = None,
) -> tuple[int, ...]:
    """Select grid cells with nonzero overlap, preferring largest overlap."""

    validate_box(box, image_size)
    x1, y1, x2, y2 = box
    width, height = image_size
    rows, columns = grid
    if rows <= 0 or columns <= 0:
        raise ValueError("grid dimensions must be positive")
    selected: list[tuple[int, float]] = []
    for row in range(rows):
        py1, py2 = row * height / rows, (row + 1) * height / rows
        for column in range(columns):
            px1, px2 = column * width / columns, (column + 1) * width / columns
            overlap = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                0.0, min(y2, py2) - max(y1, py1)
            )
            if overlap > 0:
                selected.append((row * columns + column, overlap))
    if limit is not None:
        if limit <= 0:
            raise ValueError("patch limit must be positive")
        selected = sorted(selected, key=lambda item: (-item[1], item[0]))[:limit]
    return tuple(sorted(index for index, _overlap in selected))


def png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def coordinate_tool_candidate(label: str, coordinate: tuple[int, int]) -> CandidateSequence:
    """Build a native desktop_action completion scored only on x/y values."""

    x, y = coordinate
    prefix = "<tool_call>\n<function=desktop_action>\n<parameter=action>\nclick\n</parameter>\n<parameter=x>\n"
    middle = "\n</parameter>\n<parameter=y>\n"
    suffix = "\n</parameter>\n</function>\n</tool_call>"
    x_text, y_text = str(x), str(y)
    x_span = (len(prefix), len(prefix) + len(x_text))
    y_start = len(prefix) + len(x_text) + len(middle)
    y_span = (y_start, y_start + len(y_text))
    return CandidateSequence(
        label=label,
        text=prefix + x_text + middle + y_text + suffix,
        scored_spans=(x_span, y_span),
    )


def json_coordinate_candidate(label: str, coordinate: tuple[int, int]) -> CandidateSequence:
    """Build the canonical compact ScreenSpot JSON response, scoring only x/y values."""

    x, y = coordinate
    text = json.dumps({"x": x, "y": y}, separators=(",", ":"))
    x_text, y_text = str(x), str(y)
    x_start = text.index(x_text, text.index('"x"'))
    y_start = text.index(y_text, text.index('"y"'))
    return CandidateSequence(
        label=label,
        text=text,
        scored_spans=((x_start, x_start + len(x_text)), (y_start, y_start + len(y_text))),
    )


def native_tool_candidate(
    label: str,
    action: Mapping[str, Any],
    *,
    scored_fields: Sequence[str] | None = None,
) -> CandidateSequence:
    """Build one native Holo tool call and score the declared parameter values."""

    if not action or "action" not in action:
        raise ValueError("tool action must contain an action field")
    fields = tuple(scored_fields or action.keys())
    unknown = [field for field in fields if field not in action]
    if unknown:
        raise ValueError(f"scored tool fields are absent from the action: {unknown}")
    chunks = ["<tool_call>\n<function=desktop_action>\n"]
    spans: dict[str, tuple[int, int]] = {}
    length = len(chunks[0])
    for field, raw_value in action.items():
        value = json.dumps(raw_value, separators=(",", ":"))
        if isinstance(raw_value, str):
            value = raw_value
        prefix = f"<parameter={field}>\n"
        suffix = "\n</parameter>\n"
        chunks.extend((prefix, value, suffix))
        start = length + len(prefix)
        spans[str(field)] = (start, start + len(value))
        length += len(prefix) + len(value) + len(suffix)
    chunks.append("</function>\n</tool_call>")
    return CandidateSequence(
        label=label,
        text="".join(chunks),
        scored_spans=tuple(spans[field] for field in fields),
    )


def encode_candidate(tokenizer: Any, candidate: CandidateSequence) -> EncodedCandidate:
    encoded = tokenizer(candidate.text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tuple(int(value) for value in encoded["input_ids"])
    selected = tuple(
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if any(start < span_end and end > span_start for span_start, span_end in candidate.scored_spans)
    )
    if not selected:
        raise RuntimeError(f"tokenizer produced no scored tokens for {candidate.label!r}")
    return EncodedCandidate(candidate.label, token_ids, selected)


def _image_sizes(messages: Sequence[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    sizes: list[tuple[int, int]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image":
                sizes.append(part["image"].size)
    return tuple(sizes)


def _split_image_positions(
    input_ids: Any,
    image_token_id: int,
    grids: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    values = input_ids[0].detach().cpu().tolist()
    positions = [index for index, value in enumerate(values) if int(value) == image_token_id]
    expected = sum(math.prod(grid) for grid in grids)
    if len(positions) != expected:
        raise RuntimeError(f"prompt contains {len(positions)} image tokens but grids require {expected}")
    spans: list[tuple[int, ...]] = []
    offset = 0
    for grid in grids:
        count = math.prod(grid)
        span = tuple(positions[offset : offset + count])
        if span != tuple(range(span[0], span[-1] + 1)):
            raise RuntimeError("each image must occupy one contiguous image-token span")
        spans.append(span)
        offset += count
    return tuple(spans)


class ActivationPatchingRunner:
    """Prepare, score, patch, and ablate two teacher-forced candidates."""

    def __init__(self, engine: InstrumentedHolo, *, reduction: ScoreReduction = "sum") -> None:
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.engine = engine
        self.reduction = reduction

    def prepare(
        self,
        *,
        name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        candidates: tuple[CandidateSequence, CandidateSequence],
        regions: Mapping[str, ImageRegion] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> PreparedCondition:
        torch = self.engine.torch
        model = self.engine.model
        processor = self.engine.processor
        assert torch is not None and model is not None and processor is not None

        normalised = normalise_messages(messages)
        image_sizes = _image_sizes(normalised)
        template_kwargs = {"enable_thinking": False, **(chat_template_kwargs or {})}
        prompt = processor.apply_chat_template(
            normalised,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.engine.device)
        prompt_length = int(prompt["input_ids"].shape[-1])
        encodings = tuple(encode_candidate(processor.tokenizer, candidate) for candidate in candidates)
        if self.reduction == "sum" and len({len(value.scored_token_offsets) for value in encodings}) != 1:
            raise ValueError(
                "sum margins require equal scored-token counts; use reduction='mean' for unequal candidates"
            )

        count = len(encodings)
        max_suffix = max(len(value.token_ids) for value in encodings)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id
        suffixes = torch.full((count, max_suffix), int(pad_id), device=self.engine.device, dtype=torch.long)
        suffix_mask = torch.zeros(
            (count, max_suffix), device=self.engine.device, dtype=prompt["attention_mask"].dtype
        )
        for index, encoding in enumerate(encodings):
            length = len(encoding.token_ids)
            suffixes[index, :length] = torch.tensor(encoding.token_ids, device=self.engine.device)
            suffix_mask[index, :length] = 1
        input_ids = torch.cat([prompt["input_ids"].repeat(count, 1), suffixes], dim=1)
        attention_mask = torch.cat([prompt["attention_mask"].repeat(count, 1), suffix_mask], dim=1)
        mm_ids = torch.cat(
            [
                prompt["mm_token_type_ids"].repeat(count, 1),
                torch.zeros_like(suffix_mask, dtype=prompt["mm_token_type_ids"].dtype),
            ],
            dim=1,
        )
        image_grid = prompt["image_grid_thw"]
        batch_grid = image_grid.repeat(count, 1)

        with torch.inference_mode():
            inputs_embeds = model.get_input_embeddings()(input_ids)
            image_outputs = model.model.get_image_features(
                prompt["pixel_values"], image_grid, return_dict=True
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            repeated_image_embeds = image_embeds.repeat(count, 1)
            image_mask, _ = model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=repeated_image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, repeated_image_embeds)
            position_ids = model.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=batch_grid,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=mm_ids,
            )

        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        grids: list[tuple[int, int]] = []
        for values in image_grid.detach().cpu().tolist():
            _temporal, raw_rows, raw_columns = [int(value) for value in values]
            if raw_rows % merge_size or raw_columns % merge_size:
                raise RuntimeError("vision grid is not divisible by the processor merge size")
            grids.append((raw_rows // merge_size, raw_columns // merge_size))
        if len(grids) != len(image_sizes):
            raise RuntimeError("processor image grids do not match images in the message history")
        image_positions = _split_image_positions(
            prompt["input_ids"], int(model.config.image_token_id), grids
        )

        region_positions: dict[str, tuple[int, ...]] = {}
        for region_name, region in (regions or {}).items():
            if not 0 <= region.image_index < len(image_positions):
                raise ValueError(f"region {region_name!r} refers to missing image {region.image_index}")
            indices = overlapping_patch_indices(
                region.box,
                image_sizes[region.image_index],
                grids[region.image_index],
                limit=region.limit,
            )
            region_positions[region_name] = tuple(image_positions[region.image_index][index] for index in indices)

        prediction_positions: list[tuple[int, ...]] = []
        scored_token_ids: list[tuple[int, ...]] = []
        for encoding in encodings:
            token_positions = tuple(prompt_length + offset for offset in encoding.scored_token_offsets)
            prediction_positions.append(tuple(position - 1 for position in token_positions))
            scored_token_ids.append(tuple(encoding.token_ids[offset] for offset in encoding.scored_token_offsets))

        return PreparedCondition(
            name=name,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_ids=input_ids,
            prompt_length=prompt_length,
            candidates=encodings,
            image_positions=image_positions,
            image_grids=tuple(grids),
            region_positions=region_positions,
            prediction_positions=tuple(prediction_positions),
            scored_token_ids=tuple(scored_token_ids),
        )

    def score(
        self,
        condition: PreparedCondition,
        hooks: tuple[tuple[Any, Literal["pre", "forward"], Callable[..., Any]], ...] = (),
    ) -> MarginScore:
        torch = self.engine.torch
        model = self.engine.model
        assert torch is not None and model is not None
        with ExitStack() as stack:
            for module, kind, hook in hooks:
                handle = (
                    module.register_forward_pre_hook(hook)
                    if kind == "pre"
                    else module.register_forward_hook(hook)
                )
                stack.callback(handle.remove)
            with torch.inference_mode():
                outputs = model.model.language_model(
                    input_ids=None,
                    inputs_embeds=condition.inputs_embeds,
                    attention_mask=condition.attention_mask,
                    position_ids=condition.position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                scores: list[float] = []
                for batch_index, (positions, token_ids) in enumerate(
                    zip(condition.prediction_positions, condition.scored_token_ids, strict=True)
                ):
                    selected = outputs.last_hidden_state[batch_index, list(positions), :]
                    logits = model.lm_head(selected).float()
                    targets = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
                    token_log_probs = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])
                    aggregate = token_log_probs.sum() if self.reduction == "sum" else token_log_probs.mean()
                    scores.append(float(aggregate.item()))
                del outputs
        return MarginScore(scores[0] - scores[1], (scores[0], scores[1]), self.reduction)


def replace_positions(
    output: Any,
    positions: tuple[int, ...],
    source: Any | None,
    ablation_pool: tuple[int, ...] | None = None,
) -> Any:
    result = output.clone()
    if source is None:
        pool = ablation_pool or positions
        retained = tuple(index for index in pool if index not in set(positions)) or pool
        replacement = output[:, retained, :].mean(dim=1, keepdim=True)
        result[:, positions, :] = replacement
    else:
        result[:, positions, :] = source[:, positions, :].to(result)
    return result


def replace_prediction_positions(output: Any, condition: PreparedCondition, source: Any | None) -> Any:
    result = output.clone()
    for batch_index, positions in enumerate(condition.prediction_positions):
        if source is None:
            result[batch_index, positions, :] = 0
        else:
            result[batch_index, positions, :] = source[batch_index, positions, :].to(result)
    return result


def replace_attention_head(
    output: Any,
    condition: PreparedCondition,
    source: Any | None,
    *,
    head: int,
    head_dim: int,
) -> Any:
    result = output.clone()
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    for batch_index, positions in enumerate(condition.prediction_positions):
        if source is None:
            result[batch_index, positions, head_slice] = 0
        else:
            result[batch_index, positions, head_slice] = source[
                batch_index, positions, head_slice
            ].to(result)
    return result


def capture_hook(store: dict[str, Any], name: str) -> Callable[[Any, Any, Any], None]:
    def hook(_module: Any, _args: Any, output: Any) -> None:
        store[name] = output.detach().clone()

    return hook


def capture_pre_hook(store: dict[str, Any], name: str) -> Callable[[Any, Any], None]:
    def hook(_module: Any, args: Any) -> None:
        store[name] = args[0].detach().clone()

    return hook


def residual_hook(
    positions: tuple[int, ...],
    source: Any | None,
    ablation_pool: tuple[int, ...] | None = None,
) -> Callable[[Any, Any, Any], Any]:
    return lambda _module, _args, output: replace_positions(output, positions, source, ablation_pool)


def prediction_hook(
    condition: PreparedCondition, source: Any | None
) -> Callable[[Any, Any, Any], Any]:
    return lambda _module, _args, output: replace_prediction_positions(output, condition, source)


def attention_hook(
    condition: PreparedCondition,
    source: Any | None,
    *,
    head: int,
    head_dim: int,
) -> Callable[[Any, Any], Any]:
    def hook(_module: Any, args: Any) -> Any:
        return (
            replace_attention_head(args[0], condition, source, head=head, head_dim=head_dim),
            *args[1:],
        )

    return hook
