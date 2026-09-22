"""Serializable, pyvene-inspired intervention API for Holo multimodal runs.

The public configuration vocabulary follows pyvene's useful separation of a
representation (layer, component, unit) from the operation performed on it.
Execution remains Holo-specific because image regions must be projected onto
vision tokens and candidate coordinate tokens are scored under teacher forcing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .activation_patching import (
    ActivationPatchingRunner,
    ImageRegion,
    MarginScore,
    PreparedCondition,
    attention_hook,
    capture_hook,
    capture_pre_hook,
    prediction_hook,
    require_patch_alignment,
    residual_hook,
)

Component = Literal["residual_output", "mlp_output", "attention_head_output"]
Unit = Literal["all_image_tokens", "image", "image_region", "scored_token_predictions"]
Intervention = Literal["interchange"]
Ablation = Literal["mean", "zero"]
HookKind = Literal["pre", "forward"]


@dataclass(frozen=True)
class RepresentationConfig:
    """One model representation and the semantic units selected within it."""

    layer: int
    component: Component
    unit: Unit
    head: int | None = None
    image_index: int | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.component == "attention_head_output" and self.head is None:
            raise ValueError("attention_head_output requires a head")
        if self.head is not None and self.head < 0:
            raise ValueError("head must be non-negative")
        if self.component != "attention_head_output" and self.head is not None:
            raise ValueError("head is only valid for attention_head_output")
        if self.unit == "image" and self.image_index is None:
            raise ValueError("image unit requires image_index")
        if self.image_index is not None and self.image_index < 0:
            raise ValueError("image_index must be non-negative")
        if self.unit == "image_region" and self.region is None:
            raise ValueError("image_region unit requires region")
        if self.component in {"mlp_output", "attention_head_output"} and self.unit != "scored_token_predictions":
            raise ValueError(f"{self.component} currently supports scored_token_predictions only")

    @property
    def key(self) -> str:
        suffix = f":{self.head}" if self.head is not None else ""
        return f"{self.component}:{self.layer}{suffix}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepresentationConfig:
        """Parse the stable schema and the original kind/scope manifest schema."""

        if "representation" in value:
            value = value["representation"]
        component = value.get("component")
        unit = value.get("unit")
        if component is None and "kind" in value:
            component = {
                "residual": "residual_output",
                "mlp": "mlp_output",
                "attention_head": "attention_head_output",
            }.get(str(value["kind"]))
            scope = str(value.get("scope", "all_images"))
            unit = {
                "all_images": "all_image_tokens",
                "image": "image",
                "region": "image_region",
            }.get(scope) if value["kind"] == "residual" else "scored_token_predictions"
        if component not in {"residual_output", "mlp_output", "attention_head_output"}:
            raise ValueError(f"unsupported component: {component!r}")
        if unit not in {"all_image_tokens", "image", "image_region", "scored_token_predictions"}:
            raise ValueError(f"unsupported unit: {unit!r}")
        return cls(
            layer=int(value["layer"]),
            component=component,
            unit=unit,
            head=int(value["head"]) if value.get("head") is not None else None,
            image_index=int(value["image_index"]) if value.get("image_index") is not None else None,
            region=str(value["region"]) if value.get("region") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class InterventionConfig:
    """Operation to apply to one configured representation."""

    name: str
    representation: RepresentationConfig
    intervention: Intervention = "interchange"
    ablation: Ablation = "mean"

    def __post_init__(self) -> None:
        if self.intervention != "interchange":
            raise ValueError("only clean-to-corrupted interchange is currently supported")
        if self.ablation == "mean" and self.representation.component != "residual_output":
            raise ValueError("mean ablation is currently supported for residual outputs only")
        if self.representation.component == "residual_output":
            if self.representation.unit == "scored_token_predictions" and self.ablation != "zero":
                raise ValueError("coordinate-prediction residuals currently use zero ablation")
            if self.representation.unit != "scored_token_predictions" and self.ablation != "mean":
                raise ValueError("image-position residuals currently use mean ablation")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, index: int = 0) -> InterventionConfig:
        representation = RepresentationConfig.from_dict(value)
        default_ablation: Ablation = (
            "mean"
            if representation.component == "residual_output" and representation.unit != "scored_token_predictions"
            else "zero"
        )
        name = str(value.get("name") or f"{representation.component}-{representation.layer}-{index:03d}")
        return cls(
            name=name,
            representation=representation,
            intervention=str(value.get("intervention", "interchange")),  # type: ignore[arg-type]
            ablation=str(value.get("ablation", default_ablation)),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "representation": self.representation.to_dict(),
            "intervention": self.intervention,
            "ablation": self.ablation,
        }


@dataclass(frozen=True)
class InterventionEffect:
    """Paired patching and clean-run ablation result for one component."""

    config: InterventionConfig
    clean: MarginScore
    corrupted: MarginScore
    patched_corrupted: MarginScore
    ablated_clean: MarginScore
    restoration_nats: float
    recovery_fraction: float | None
    ablation_drop_nats: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "patched_corrupted_margin_nats": self.patched_corrupted.margin_nats,
            "restoration_nats": self.restoration_nats,
            "recovery_fraction": self.recovery_fraction,
            "ablated_clean_margin_nats": self.ablated_clean.margin_nats,
            "ablation_drop_nats": self.ablation_drop_nats,
            "patched_candidate_log_prob_nats": list(self.patched_corrupted.candidate_log_prob_nats),
            "ablated_candidate_log_prob_nats": list(self.ablated_clean.candidate_log_prob_nats),
            "patched_field_margins_nats": dict(self.patched_corrupted.field_margin_nats),
            "ablated_field_margins_nats": dict(self.ablated_clean.field_margin_nats),
            "field_restoration_nats": {
                field: self.patched_corrupted.field_margin_nats[field] - self.corrupted.field_margin_nats[field]
                for field in sorted(set(self.patched_corrupted.field_margin_nats).intersection(self.corrupted.field_margin_nats))
            },
            "field_ablation_drop_nats": {
                field: self.clean.field_margin_nats[field] - self.ablated_clean.field_margin_nats[field]
                for field in sorted(set(self.clean.field_margin_nats).intersection(self.ablated_clean.field_margin_nats))
            },
        }


@dataclass(frozen=True)
class InterventionOutput:
    """Original clean/corrupted outputs plus all requested intervention effects."""

    clean: MarginScore
    corrupted: MarginScore
    effects: tuple[InterventionEffect, ...]

    @property
    def clean_corrupted_gap_nats(self) -> float:
        return self.clean.margin_nats - self.corrupted.margin_nats


class IntervenableHolo:
    """Run aligned clean/corrupted patching from serializable configurations."""

    def __init__(
        self,
        runner: ActivationPatchingRunner,
        config: Sequence[InterventionConfig],
        *,
        regions: Mapping[str, ImageRegion] | None = None,
    ) -> None:
        self.runner = runner
        self.config = tuple(config)
        self.regions = dict(regions or {})
        self.source_activations: dict[str, Any] = {}

    def _module(self, representation: RepresentationConfig) -> tuple[Any, HookKind]:
        model = self.runner.engine.model
        assert model is not None
        layers = model.model.language_model.layers
        if not 0 <= representation.layer < len(layers):
            raise ValueError(f"intervention layer {representation.layer} is outside the model")
        layer = layers[representation.layer]
        if representation.component == "residual_output":
            return layer, "forward"
        if representation.component == "mlp_output":
            return layer.mlp, "forward"
        if not hasattr(layer, "self_attn"):
            raise ValueError(f"layer {representation.layer} is not a conventional attention layer")
        return layer.self_attn.o_proj, "pre"

    def _capture_hooks(self) -> tuple[tuple[Any, HookKind, Any], ...]:
        hooks = []
        seen: set[str] = set()
        for item in self.config:
            representation = item.representation
            if representation.key in seen:
                continue
            module, kind = self._module(representation)
            hook = (
                capture_pre_hook(self.source_activations, representation.key)
                if kind == "pre"
                else capture_hook(self.source_activations, representation.key)
            )
            hooks.append((module, kind, hook))
            seen.add(representation.key)
        return tuple(hooks)

    def _residual_positions(
        self,
        representation: RepresentationConfig,
        condition: PreparedCondition,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if representation.unit == "all_image_tokens":
            return condition.all_image_positions, condition.all_image_positions
        if representation.unit == "image":
            assert representation.image_index is not None
            positions = condition.image_positions[representation.image_index]
            return positions, positions
        if representation.unit == "image_region":
            assert representation.region is not None
            region = self.regions[representation.region]
            return condition.region_positions[representation.region], condition.image_positions[region.image_index]
        raise ValueError("residual outputs require an image-token unit")

    def _hooks(
        self,
        item: InterventionConfig,
        *,
        condition: PreparedCondition,
        patch: bool,
    ) -> tuple[tuple[Any, HookKind, Any], ...]:
        representation = item.representation
        module, kind = self._module(representation)
        source = self.source_activations[representation.key] if patch else None
        if representation.component == "residual_output":
            if representation.unit == "scored_token_predictions":
                hook = prediction_hook(condition, source)
            else:
                positions, pool = self._residual_positions(representation, condition)
                hook = residual_hook(positions, source, pool)
        elif representation.component == "mlp_output":
            hook = prediction_hook(condition, source)
        else:
            model = self.runner.engine.model
            assert model is not None and representation.head is not None
            attention = model.model.language_model.layers[representation.layer].self_attn
            if not 0 <= representation.head < int(model.config.text_config.num_attention_heads):
                raise ValueError(f"head {representation.head} is outside the configured query heads")
            hook = attention_hook(
                condition,
                source,
                head=representation.head,
                head_dim=int(attention.head_dim),
            )
        return ((module, kind, hook),)

    def compare(self, clean: PreparedCondition, corrupted: PreparedCondition) -> InterventionOutput:
        """Capture clean sources, then patch corrupted and ablate clean runs."""

        require_patch_alignment(clean, corrupted)
        self.source_activations.clear()
        clean_score = self.runner.score(clean, self._capture_hooks())
        corrupted_score = self.runner.score(corrupted)
        gap = clean_score.margin_nats - corrupted_score.margin_nats
        effects = []
        for item in self.config:
            patched = self.runner.score(corrupted, self._hooks(item, condition=corrupted, patch=True))
            ablated = self.runner.score(clean, self._hooks(item, condition=clean, patch=False))
            restoration = patched.margin_nats - corrupted_score.margin_nats
            effects.append(
                InterventionEffect(
                    config=item,
                    clean=clean_score,
                    corrupted=corrupted_score,
                    patched_corrupted=patched,
                    ablated_clean=ablated,
                    restoration_nats=restoration,
                    recovery_fraction=restoration / gap if abs(gap) > 1e-9 else None,
                    ablation_drop_nats=clean_score.margin_nats - ablated.margin_nats,
                )
            )
        return InterventionOutput(clean=clean_score, corrupted=corrupted_score, effects=tuple(effects))
