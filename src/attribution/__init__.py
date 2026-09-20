"""Attention-to-image attribution and static viewer generation."""

from .contrast import PromptContrast, build_prompt_contrast, layer_head_statistics, spatial_metrics
from .contrast_viewer import write_prompt_contrast_viewer
from .pipeline import AttentionAttribution, AttributionError, load_attribution, resolve_trace_path
from .viewer import write_attribution_viewer, write_trajectory_viewer

__all__ = [
    "AttentionAttribution",
    "AttributionError",
    "PromptContrast",
    "build_prompt_contrast",
    "layer_head_statistics",
    "load_attribution",
    "resolve_trace_path",
    "spatial_metrics",
    "write_attribution_viewer",
    "write_prompt_contrast_viewer",
    "write_trajectory_viewer",
]
