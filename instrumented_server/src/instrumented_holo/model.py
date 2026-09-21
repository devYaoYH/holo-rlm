"""Native Transformers model loading and focused activation capture."""

from __future__ import annotations

import base64
import gc
import json
import logging
import math
import re
import time
from importlib.metadata import version
from io import BytesIO
from typing import Any

from PIL import Image

from .settings import Settings
from .traces import TraceOptions, TraceWriter

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_COORDINATE_RE = re.compile(r'"(?P<parameter>x|y)"\s*:\s*(?P<value>-?\d+)')
logger = logging.getLogger(__name__)


def unexpected_missing_checkpoint_keys(
    model_keys: set[str],
    loaded_keys: set[str],
    *,
    tie_word_embeddings: bool,
) -> set[str]:
    """Return missing tensors after accounting for the standard tied LM-head alias."""

    allowed = {"lm_head.weight"} if tie_word_embeddings else set()
    return model_keys - loaded_keys - allowed


def generated_parameter_token_spans(
    text: str,
    token_offsets: tuple[tuple[int, int], ...],
) -> tuple[tuple[str | None, ...], tuple[tuple[str, str, tuple[int, ...]], ...]]:
    """Map generated-token character offsets onto native tool parameter values."""

    labels: list[str | None] = [None] * len(token_offsets)
    spans: list[tuple[str, str, tuple[int, ...]]] = []
    value_matches: list[tuple[str, str, tuple[int, int]]] = []
    for match in _PARAMETER_RE.finditer(text):
        parameter = match.group(1).strip()
        raw_value = match.group(2)
        leading = len(raw_value) - len(raw_value.lstrip())
        trailing = len(raw_value) - len(raw_value.rstrip())
        start, raw_end = match.span(2)
        start += leading
        end = raw_end - trailing
        value = text[start:end]
        value_matches.append((parameter, value, (start, end)))
    if not value_matches:
        value_matches.extend(
            (match.group("parameter"), match.group("value"), match.span("value"))
            for match in _JSON_COORDINATE_RE.finditer(text)
        )

    for parameter, value, (start, end) in value_matches:
        indices = tuple(
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_end > start and token_start < end
        )
        if not value or not indices:
            continue
        for index in indices:
            labels[index] = parameter
        spans.append((parameter, value, indices))
    return tuple(labels), tuple(spans)


def generation_stop_strings(
    tools: list[dict[str, Any]] | None,
    raw_request: dict[str, Any],
) -> tuple[str, ...]:
    """Return the native delimiter that closes the requested output protocol."""

    if tools:
        return ("</tool_call>",)
    if isinstance(raw_request.get("structured_outputs"), dict):
        return ("<|im_end|>",)
    return ()


class UnsupportedImageSource(ValueError):
    """Raised when an image would require an unexpected network fetch."""


def decode_data_image(url: str) -> Image.Image:
    """Decode only data URLs so a local model server never fetches remote images."""

    if not url.startswith("data:image/") or "," not in url:
        raise UnsupportedImageSource(
            "Only data:image/...;base64,... inputs are accepted by the instrumented server. "
            "Capture the screenshot bytes at the caller and send them as a data URL."
        )
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        raise UnsupportedImageSource("Image data URLs must use base64 encoding.")
    image = Image.open(BytesIO(base64.b64decode(payload, validate=True)))
    return image.convert("RGB")


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI image_url parts to the local processor's image content parts."""

    normalised: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalised.append({"role": message["role"], "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("Each message content must be a string or a list of content parts.")
        parts: list[dict[str, Any]] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                value = part.get("image_url", {})
                url = value if isinstance(value, str) else value.get("url")
                if not isinstance(url, str):
                    raise ValueError("image_url parts require an image_url.url string.")
                parts.append({"type": "image", "image": decode_data_image(url)})
            else:
                raise ValueError(f"Unsupported content-part type: {part_type!r}")
        normalised.append({"role": message["role"], "content": parts})
    return normalised


def parse_assistant_output(text: str, *, thinking_enabled: bool = True) -> tuple[str, list[dict[str, Any]]]:
    """Strip hidden thinking and project Holo/Qwen native XML calls to OpenAI calls."""

    if thinking_enabled and "</think>" in text:
        without_thinking = text.split("</think>", 1)[1].strip()
    elif thinking_enabled and text.lstrip().startswith("<think>"):
        without_thinking = _THINK_RE.sub("", text).strip()
    elif thinking_enabled:
        # The generation prompt pre-fills `<think>\n`, so decoded new tokens
        # contain only the body. With no closing tag the generation ended inside
        # hidden reasoning and must not be returned as assistant content.
        without_thinking = ""
    else:
        without_thinking = text.strip()
    tool_calls: list[dict[str, Any]] = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(without_thinking)):
        name, body = match.groups()
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER_RE.finditer(body):
            key, raw_value = parameter.groups()
            value = raw_value.strip()
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
        tool_calls.append(
            {
                "id": f"call_{int(time.time_ns())}_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, separators=(",", ":"))},
            }
        )
    visible = _TOOL_CALL_RE.sub("", without_thinking).strip()
    return visible, tool_calls


class InstrumentedHolo:
    """Lazy native-model loader so health checks do not allocate model memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device: str | None = None
        self.dtype: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.settings.model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.settings.model_path}")

        import torch
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        if self.settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = self.settings.device

        if self.settings.dtype == "auto":
            # The native checkpoint is BF16. Keeping that dtype on MPS avoids a
            # second, temporary MPS allocation just to cast every loaded tensor
            # to FP16. CPU retains FP32 as its conservative default.
            self.dtype = torch.bfloat16 if self.device in {"cuda", "mps"} else torch.float32
        else:
            self.dtype = getattr(torch, self.settings.dtype)

        processor_path = self.settings.processor_path or self.settings.model_path
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        if self.settings.image_max_pixels < self.settings.image_min_pixels:
            raise ValueError("HOLO_IMAGE_MAX_PIXELS must be greater than or equal to HOLO_IMAGE_MIN_PIXELS")
        # Bound each historical frame before visual tokenization. Full prompt
        # attention is quadratic in the combined text+image sequence length;
        # the checkpoint default admits up to 16M pixels per image and a
        # four-frame traced request exhausted a 24 GiB unified-memory machine.
        self.processor.image_processor.size = {
            "shortest_edge": self.settings.image_min_pixels,
            "longest_edge": self.settings.image_max_pixels,
        }
        if self.settings.load_strategy != "stream":
            raise ValueError(
                f"Unsupported HOLO_LOAD_STRATEGY={self.settings.load_strategy!r}; "
                "only 'stream' is supported."
            )

        # Build a parameter-free (meta-device) skeleton, then load one tensor at
        # a time from each safetensor shard directly onto the target device.
        # This deliberately avoids `from_pretrained(...).to('mps')`, whose
        # copy-then-free semantics can hold two whole models at once. It is more
        # granular than Accelerate's one-device fast path, which reads an entire
        # safetensor shard into a dict before assigning it to the model.
        config = AutoConfig.from_pretrained(self.settings.model_path, local_files_only=True)
        checkpoint_dtype_name, checkpoint_dtype = self._checkpoint_float_dtype(safe_open, torch)
        if (
            self.device == "mps"
            and checkpoint_dtype is not None
            and self.dtype != checkpoint_dtype
            and not self.settings.allow_dtype_conversion
        ):
            raise ValueError(
                f"Refusing Metal checkpoint conversion from {checkpoint_dtype_name} to "
                f"{str(self.dtype).replace('torch.', '')}: MPS may retain both allocations and approach 2x weight "
                "memory. Use HOLO_DTYPE=auto (native checkpoint dtype) or explicitly set "
                "HOLO_ALLOW_DTYPE_CONVERSION=1 if the extra memory is intentional."
            )
        model_kwargs: dict[str, Any] = {}
        if self.settings.eager_attention:
            model_kwargs["attn_implementation"] = "eager"
        # Parameters are meta tensors, but small runtime buffers (for example
        # rotary inv_freq) must be real CPU tensors. Putting buffers on meta can
        # leave non-persistent Qwen buffers as unallocated MPS placeholders.
        with init_empty_weights(include_buffers=False):
            model = AutoModelForMultimodalLM.from_config(config, **model_kwargs)
        model_keys = set(model.state_dict().keys())
        index_path = self.settings.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing safetensor shard index: {index_path}")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        loaded_keys: set[str] = set()
        for shard_name in shard_names:
            logger.warning("Streaming checkpoint shard %s directly to %s (%s)", shard_name, self.device, self.dtype)
            shard_path = self.settings.model_path / shard_name
            with safe_open(shard_path, framework="pt", device=self.device) as shard:
                # safetensors.safe_open exposes keys() but is not itself iterable.
                for key in shard.keys():  # noqa: SIM118
                    if key not in model_keys:
                        continue
                    value = shard.get_tensor(key)
                    # Match the requested compute dtype per tensor, keeping the
                    # conversion window bounded to this one tensor.
                    if self.torch.is_floating_point(value) and value.dtype != self.dtype:
                        value = value.to(dtype=self.dtype)
                    # The meta skeleton defaults to FP32. Without an explicit
                    # dtype Accelerate casts every BF16 checkpoint tensor to
                    # the skeleton dtype, doubling model residency on Metal.
                    set_module_tensor_to_device(
                        model,
                        key,
                        self.device,
                        value=value,
                        dtype=self.dtype,
                        clear_cache=False,
                    )
                    loaded_keys.add(key)
                    del value
            gc.collect()
            if self.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
                logger.warning(
                    "After %s: MPS allocated %.2f GiB; driver allocated %.2f GiB",
                    shard_name,
                    torch.mps.current_allocated_memory() / 2**30,
                    torch.mps.driver_allocated_memory() / 2**30,
                )
        tie_word_embeddings = bool(
            getattr(config, "tie_word_embeddings", False)
            or getattr(getattr(config, "text_config", None), "tie_word_embeddings", False)
        )
        missing_keys = unexpected_missing_checkpoint_keys(
            model_keys,
            loaded_keys,
            tie_word_embeddings=tie_word_embeddings,
        )
        if missing_keys:
            examples = ", ".join(sorted(missing_keys)[:5])
            raise RuntimeError(f"Checkpoint did not contain {len(missing_keys)} model tensors (for example: {examples})")
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        for name, buffer in model.named_buffers():
            if str(buffer.device) != self.device:
                set_module_tensor_to_device(model, name, self.device, value=buffer)
        meta_tensors = [
            name
            for name, value in [*model.named_parameters(), *model.named_buffers()]
            if getattr(value, "is_meta", False)
        ]
        if meta_tensors:
            examples = ", ".join(meta_tensors[:5])
            raise RuntimeError(f"Model still has {len(meta_tensors)} meta tensors after streaming load: {examples}")
        wrong_dtype = [
            f"{name}={value.dtype}"
            for name, value in model.named_parameters()
            if value.is_floating_point() and value.dtype != self.dtype
        ]
        if wrong_dtype:
            examples = ", ".join(wrong_dtype[:5])
            raise RuntimeError(f"Model has {len(wrong_dtype)} parameters outside target dtype {self.dtype}: {examples}")
        self.model = model
        self.model.eval()

    def _checkpoint_float_dtype(self, safe_open: Any, torch: Any) -> tuple[str | None, Any | None]:
        """Inspect safetensor slice metadata without materializing a tensor."""

        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
        }
        for shard_path in sorted(self.settings.model_path.glob("*.safetensors")):
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for key in shard.keys():  # noqa: SIM118
                    dtype_name = shard.get_slice(key).get_dtype()
                    if dtype_name in dtype_map:
                        return dtype_name, dtype_map[dtype_name]
        return None, None

    def model_metadata(self) -> dict[str, Any]:
        config_path = self.settings.model_path / "config.json"
        config = json.loads(config_path.read_text())
        layer_types = config.get("text_config", {}).get("layer_types", [])
        attention_layer_indices = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
        ]
        return {
            "model_path": str(self.settings.model_path),
            "processor_path": str(self.settings.processor_path or self.settings.model_path),
            "model_revision": self.checkpoint_revision(),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "image_token_id": config.get("image_token_id"),
            "vision_config": config.get("vision_config"),
            "attention_layer_indices": attention_layer_indices,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype else None,
            "load_strategy": self.settings.load_strategy,
            "eager_attention": self.settings.eager_attention,
            "image_min_pixels": self.settings.image_min_pixels,
            "image_max_pixels": self.settings.image_max_pixels,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        }

    def checkpoint_revision(self) -> str | None:
        metadata_path = self.settings.model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
        if metadata_path.is_file():
            lines = metadata_path.read_text().splitlines()
            if len(lines) >= 2 and re.fullmatch(r"[0-9a-f]{40}", lines[1]):
                return lines[1]
        return None

    def processor_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {"revision": self.checkpoint_revision(), "files": {}}
        for name in ("preprocessor_config.json", "video_preprocessor_config.json", "tokenizer_config.json"):
            path = self.settings.model_path / name
            if path.is_file():
                result["files"][name] = json.loads(path.read_text())
        if self.processor is not None:
            size = getattr(self.processor.image_processor, "size", None)
            if isinstance(size, dict):
                result["runtime_image_processor_size"] = dict(size)
        return result

    def _prepare_inputs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        enable_thinking: bool,
    ) -> Any:
        assert self.processor is not None and self.torch is not None and self.device is not None
        model_messages = normalise_messages(messages)
        inputs = self.processor.apply_chat_template(
            model_messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        trace_options: TraceOptions,
        raw_request: dict[str, Any],
    ) -> tuple[str, int, int, str | None, bool]:
        self.load()
        assert self.model is not None and self.processor is not None and self.torch is not None
        template_options = raw_request.get("chat_template_kwargs")
        enable_thinking = (
            bool(template_options.get("enable_thinking", True))
            if isinstance(template_options, dict)
            else True
        )
        inputs = self._prepare_inputs(messages, tools, enable_thinking=enable_thinking)
        trace_id: str | None = None
        writer: TraceWriter | None = None
        if trace_options.enabled:
            trace_id, writer = TraceWriter.create(self.settings.trace_dir)
            writer.write_json("request.json", raw_request)
            self._write_input_images(writer, raw_request)
            writer.write_json("model.json", self.model_metadata())
            writer.write_json("processor.json", self.processor_metadata())
            writer.write_array("input_ids.npy", inputs["input_ids"].detach().cpu().numpy())
            vision_metadata = {}
            for key, value in inputs.items():
                if "grid" not in key and "pixel" not in key and "image" not in key:
                    continue
                if hasattr(value, "shape"):
                    metadata: dict[str, Any] = {"shape": list(value.shape)}
                    # Grid coordinates are tiny and essential for projecting
                    # merged image tokens back to patches. Pixel tensors stay
                    # out of JSON; their exact source image is already stored.
                    if "grid" in key and hasattr(value, "numel") and value.numel() <= 256:
                        metadata["values"] = value.detach().cpu().tolist()
                    vision_metadata[key] = metadata
                else:
                    vision_metadata[key] = {"value": str(value)}
            writer.write_json("vision_inputs.json", vision_metadata)

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": bool(temperature and temperature > 0),
            "return_dict_in_generate": trace_options.enabled,
        }
        if temperature and temperature > 0:
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
        if trace_options.enabled and trace_options.capture_attentions:
            generate_kwargs["output_attentions"] = True
        if trace_options.enabled and trace_options.capture_hidden_states:
            generate_kwargs["output_hidden_states"] = True
        if trace_options.enabled and trace_options.capture_logprobs:
            generate_kwargs["output_scores"] = True
        if trace_options.enabled and (trace_options.capture_kv or trace_options.capture_value_norms):
            generate_kwargs["use_cache"] = True
        stop_strings = generation_stop_strings(tools, raw_request)
        if stop_strings:
            # Qwen's chat template terminates a structured assistant turn with
            # im_end. Tool calls use their own closing tag. Without an explicit
            # stop, bare Transformers generation can invent the next user turn.
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = self.processor.tokenizer

        with self.torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)

        sequences = generated.sequences if trace_options.enabled else generated
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_tokens = sequences[:, prompt_tokens:]
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if writer is not None:
            writer.write_array("generated_ids.npy", generated_tokens.detach().cpu().numpy())
            writer.write_json(
                "generated_tokens.json",
                {
                    "token_ids": [int(value) for value in generated_tokens[0].detach().cpu().tolist()],
                    "tokens": self.processor.tokenizer.convert_ids_to_tokens(
                        generated_tokens[0].detach().cpu().tolist()
                    ),
                },
            )
            writer.write_json(
                "positions.json",
                {
                    "prompt_token_count": prompt_tokens,
                    "generated_token_count": int(generated_tokens.shape[-1]),
                    "image_token_id": self.model.config.image_token_id,
                    "note": (
                        "Attention keys use these sequence positions. Vision-grid shapes and exact grid coordinates "
                        "are in vision_inputs.json."
                    ),
                },
            )
            self._write_focused_attentions(writer, generated, trace_options)
            self._write_focused_hidden_states(writer, generated, trace_options)
            self._write_token_logprobs(writer, generated, generated_tokens, trace_options)
            self._write_kv_metadata(writer, generated, trace_options)
            writer.write_json("completion.json", {"text": text})
            writer.write_json("manifest.json", writer.manifest())

        return text, prompt_tokens, int(generated_tokens.shape[-1]), trace_id, enable_thinking

    def _write_token_logprobs(
        self,
        writer: TraceWriter,
        generated: Any,
        generated_tokens: Any,
        options: TraceOptions,
    ) -> None:
        """Persist chosen-token log probabilities and native tool-parameter spans."""

        scores = getattr(generated, "scores", None)
        if not options.capture_logprobs or scores is None:
            return
        assert self.processor is not None and self.torch is not None
        token_ids = [int(value) for value in generated_tokens[0].detach().cpu().tolist()]
        if len(scores) != len(token_ids):
            raise RuntimeError(
                f"generation returned {len(scores)} score rows for {len(token_ids)} generated tokens"
            )
        tokenizer = self.processor.tokenizer
        prefixes = [
            tokenizer.decode(
                token_ids[:index],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index in range(len(token_ids) + 1)
        ]
        text = prefixes[-1]
        offsets = tuple((len(prefixes[index]), len(prefixes[index + 1])) for index in range(len(token_ids)))
        parameter_by_token, parameter_spans = generated_parameter_token_spans(text, offsets)
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        records: list[dict[str, Any]] = []
        cumulative = 0.0
        for index, (token_id, logits) in enumerate(zip(token_ids, scores, strict=True)):
            selected = self.torch.log_softmax(logits[0].float(), dim=-1)[token_id]
            logprob = float(selected.item())
            cumulative += logprob
            start, end = offsets[index]
            records.append(
                {
                    "index": index,
                    "token_id": token_id,
                    "token": pieces[index],
                    "text": text[start:end],
                    "char_span": [start, end],
                    "logprob_nats": logprob,
                    "probability": math.exp(logprob),
                    "cumulative_logprob_nats": cumulative,
                    "parameter": parameter_by_token[index],
                }
            )
        aggregate_parameters: dict[str, dict[str, Any]] = {}
        for parameter, value, indices in parameter_spans:
            aggregate_parameters[parameter] = {
                "value": value,
                "token_indices": list(indices),
                "token_count": len(indices),
                "logprob_nats": sum(records[index]["logprob_nats"] for index in indices),
            }
        writer.write_json(
            "token_logprobs.json",
            {
                "schema_version": 1,
                "units": "nats",
                "sequence_logprob_nats": cumulative,
                "generated_token_count": len(records),
                "all_generated_tokens_captured": True,
                "parameters": aggregate_parameters,
                "tokens": records,
            },
        )

    def _write_input_images(self, writer: TraceWriter, raw_request: dict[str, Any]) -> None:
        image_index = 0
        for message in raw_request.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                url = image_url if isinstance(image_url, str) else image_url.get("url")
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    continue
                header, encoded = url.split(",", 1)
                extension = "png" if "png" in header else "jpg" if "jpeg" in header or "jpg" in header else "bin"
                writer.write_bytes(f"model-input-{image_index:03d}.{extension}", base64.b64decode(encoded, validate=True))
                image_index += 1

    def _write_focused_hidden_states(self, writer: TraceWriter, generated: Any, options: TraceOptions) -> None:
        """Persist last-query residual-stream vectors per generated step and layer."""

        hidden_states = getattr(generated, "hidden_states", None)
        if not options.capture_hidden_states or not hidden_states:
            return
        arrays: dict[str, Any] = {}
        limit = min(options.max_traced_generation_steps, len(hidden_states))
        for step_index, per_layer in enumerate(hidden_states[:limit]):
            if per_layer is None:
                continue
            for layer_index, state in enumerate(per_layer):
                if state is not None and hasattr(state, "shape"):
                    arrays[f"step_{step_index:03d}_layer_{layer_index:03d}"] = (
                        state[0, -1, :].float().cpu().numpy()
                    )
        if arrays:
            writer.write_npz("hidden_state_last_query_rows.npz", arrays)

    def _write_focused_attentions(self, writer: TraceWriter, generated: Any, options: TraceOptions) -> None:
        """Persist focused rows plus head-mean prompt matrices for full rollout."""

        attentions = getattr(generated, "attentions", None)
        if not options.capture_attentions or not attentions:
            return
        arrays: dict[str, Any] = {}
        limit = min(options.max_traced_generation_steps, len(attentions))
        for step_index, per_layer in enumerate(attentions[:limit]):
            for layer_index, attention in enumerate(per_layer):
                if attention is None:
                    continue
                # [batch, heads, query_tokens, key_tokens] -> [heads, key_tokens]
                arrays[f"step_{step_index:03d}_layer_{layer_index:03d}"] = attention[0, :, -1, :].float().cpu().numpy()
        if arrays:
            writer.write_npz("attention_last_query_rows.npz", arrays)
        if not options.capture_rollout:
            return

        first_step = attentions[0]
        prompt_arrays: dict[str, Any] = {}
        weighted_prompt_arrays: dict[str, Any] = {}
        cache = getattr(generated, "past_key_values", None)
        cache_layers = getattr(cache, "layers", cache) if cache is not None else None
        layer_indices = self.model_metadata().get("attention_layer_indices", [])
        for ordinal, attention in enumerate(first_step):
            if attention is None or attention.ndim != 4:
                continue
            layer_index = int(layer_indices[ordinal]) if ordinal < len(layer_indices) else ordinal
            key = f"layer_{layer_index:03d}"
            prompt_attention = attention[0]
            prompt_arrays[key] = (
                prompt_attention.mean(dim=0).to(device="cpu", dtype=self.torch.float16).numpy()
            )
            if not options.capture_value_norms or cache_layers is None:
                continue
            value = getattr(cache_layers[layer_index], "values", None)
            if value is None:
                continue
            norms = self.torch.linalg.vector_norm(value.detach(), ord=2, dim=-1)[0]
            attention_heads = int(prompt_attention.shape[0])
            kv_heads = int(norms.shape[0])
            if kv_heads < 1 or attention_heads % kv_heads:
                continue
            key_count = int(prompt_attention.shape[-1])
            expanded_norms = norms[:, :key_count].repeat_interleave(attention_heads // kv_heads, dim=0)
            weighted = prompt_attention * expanded_norms[:, None, :]
            weighted_prompt_arrays[key] = weighted.mean(dim=0).to(
                device="cpu", dtype=self.torch.float16
            ).numpy()
        if prompt_arrays:
            writer.write_npz("attention_prompt_mean.npz", prompt_arrays)
        if weighted_prompt_arrays:
            writer.write_npz("attention_prompt_value_weighted_mean.npz", weighted_prompt_arrays)
        if prompt_arrays:
            writer.write_json(
                "attention_rollout.json",
                {
                    "schema_version": 1,
                    "prompt_matrix_shape": "[prompt_queries, prompt_keys]",
                    "head_reduction": "mean",
                    "residual_rule": "row_normalize(attention), then mix 0.5 attention + 0.5 identity",
                    "value_weighting": (
                        "mean over query heads of attention * corresponding KV-head value L2 norm"
                        if weighted_prompt_arrays
                        else None
                    ),
                    "layers": [int(key.removeprefix("layer_")) for key in prompt_arrays],
                },
            )

    def _write_kv_metadata(self, writer: TraceWriter, generated: Any, options: TraceOptions) -> None:
        cache = getattr(generated, "past_key_values", None)
        if cache is None:
            return
        try:
            summary = []
            arrays: dict[str, Any] = {}
            value_norm_arrays: dict[str, Any] = {}
            layers = getattr(cache, "layers", cache)
            for layer_index, layer in enumerate(layers):
                layer_summary: dict[str, Any] = {"layer": layer_index, "type": type(layer).__name__, "states": {}}
                for field_name in ("keys", "values"):
                    value = getattr(layer, field_name, None)
                    if value is not None and hasattr(value, "shape"):
                        layer_summary["states"][field_name] = list(value.shape)
                        if options.capture_kv:
                            arrays[f"layer_{layer_index:03d}_{field_name}"] = value.detach().float().cpu().numpy()
                        if field_name == "values" and options.capture_value_norms:
                            # Keep only one scalar per key and KV head. Computing the
                            # reduction on-device avoids retaining a second full-sized
                            # float32 copy of the cache or model weights.
                            norms = self.torch.linalg.vector_norm(value.detach(), ord=2, dim=-1)
                            if norms.ndim == 3 and norms.shape[0] == 1:
                                norms = norms[0]
                            value_norm_arrays[f"layer_{layer_index:03d}"] = norms.float().cpu().numpy()
                # Qwen3.5 interleaves conventional attention with linear-attention
                # layers whose cache is a compact convolution/recurrent state, not K/V.
                for field_name in ("conv_states", "recurrent_states"):
                    states = getattr(layer, field_name, None)
                    if isinstance(states, dict):
                        layer_summary["states"][field_name] = {
                            str(state_index): list(value.shape) if value is not None else None
                            for state_index, value in states.items()
                        }
                        if options.capture_kv:
                            for state_index, value in states.items():
                                if value is not None:
                                    arrays[f"layer_{layer_index:03d}_{field_name}_{state_index}"] = (
                                        value.detach().float().cpu().numpy()
                                    )
                summary.append(layer_summary)
            writer.write_json(
                "kv_layout.json",
                {
                    "cache_type": type(cache).__name__,
                    "layers": summary,
                    "values_saved": options.capture_kv,
                    "value_norms_saved": bool(value_norm_arrays),
                },
            )
            if arrays:
                writer.write_npz("kv_cache.npz", arrays)
            if value_norm_arrays:
                writer.write_npz("value_norms.npz", value_norm_arrays)
                writer.write_json(
                    "value_norms.json",
                    {
                        "schema_version": 1,
                        "formula": "L2 norm over the value-cache head dimension",
                        "array_shape": "[kv_heads, sequence_keys]",
                        "layers": [int(key.removeprefix("layer_")) for key in value_norm_arrays],
                    },
                )
        except Exception as exc:  # Cache implementations vary across Transformers versions.
            writer.write_json("kv_layout_error.json", {"error": repr(exc)})
