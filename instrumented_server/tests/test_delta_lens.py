from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from instrumented_holo.activation_patching import (
    json_coordinate_candidate,
    native_tool_candidate,
)
from instrumented_holo.attention_delta import _slice_position_ids, compare_attention_results
from instrumented_holo.delta_lens import (
    compare_model_results,
    load_delta_lens_plan,
    manifest_summary,
    override_plan_paths,
    select_plan_cases,
)
from instrumented_holo.model import InstrumentedHolo
from instrumented_holo.settings import Settings


def test_protocol_specific_candidates_score_only_declared_values() -> None:
    coordinate = json_coordinate_candidate("oracle", (482, 351))
    assert coordinate.text == '{"x":482,"y":351}'
    assert [coordinate.text[start:end] for start, end in coordinate.scored_spans] == ["482", "351"]

    scroll = native_tool_candidate(
        "scroll_down",
        {"action": "scroll", "delta_y": -500},
        scored_fields=["action", "delta_y"],
    )
    assert scroll.text == (
        "<tool_call>\n<function=desktop_action>\n"
        "<parameter=action>\nscroll\n</parameter>\n"
        "<parameter=delta_y>\n-500\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert [scroll.text[start:end] for start, end in scroll.scored_spans] == ["scroll", "-500"]


def test_manifest_validation_resolves_images_without_requiring_uncached_model(tmp_path: Path) -> None:
    processor = tmp_path / "processor"
    processor.mkdir()
    image = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), "white").save(image)
    manifest = {
        "schema_version": 1,
        "processor_path": "processor",
        "models": [
            {"label": "Qwen", "role": "base", "path": "missing-qwen"},
            {"label": "Holo", "role": "tuned", "path": "missing-holo"},
        ],
        "cases": [
            {
                "id": "screen",
                "protocol": "hcompany_element_localization_v1",
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_path", "path": "frame.png"},
                                {"type": "text", "text": "Localize the target"},
                            ],
                        }
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                "candidates": [
                    {"label": "oracle", "json_coordinate": [500, 500]},
                    {"label": "distractor", "json_coordinate": [250, 250]},
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    plan = load_delta_lens_plan(path)
    summary = manifest_summary(plan)

    assert summary["valid"] is True
    assert summary["models"][0]["present"] is False
    assert summary["cases"][0]["image_count"] == 1
    assert plan.cases[0].candidates[0].text == '{"x":500,"y":500}'

    remote_base = tmp_path / "remote-base"
    remote_tuned = tmp_path / "remote-tuned"
    remote_processor = tmp_path / "remote-processor"
    overridden = override_plan_paths(
        plan,
        base_model=remote_base,
        tuned_model=remote_tuned,
        processor_path=remote_processor,
    )
    assert {model.role: model.path for model in overridden.models} == {
        "base": remote_base,
        "tuned": remote_tuned,
    }
    assert overridden.processor_path == remote_processor

    selected = select_plan_cases(plan, ["screen"])
    assert [case.id for case in selected.cases] == ["screen"]


def test_case_selection_rejects_unknown_and_duplicate_ids(tmp_path: Path) -> None:
    processor = tmp_path / "processor"
    processor.mkdir()
    image = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), "white").save(image)
    payload = {
        "schema_version": 1,
        "processor_path": "processor",
        "models": [
            {"label": "Qwen", "role": "base", "path": "qwen"},
            {"label": "Holo", "role": "tuned", "path": "holo"},
        ],
        "cases": [
            {
                "id": "screen",
                "protocol": "test",
                "request": {
                    "messages": [{"role": "user", "content": [{"type": "image_path", "path": "frame.png"}]}]
                },
                "candidates": [
                    {"label": "oracle", "json_coordinate": [500, 500]},
                    {"label": "distractor", "json_coordinate": [250, 250]},
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    plan = load_delta_lens_plan(path)

    try:
        select_plan_cases(plan, ["missing"])
    except ValueError as exc:
        assert "unknown delta-lens case ids" in str(exc)
    else:
        raise AssertionError("unknown case id was accepted")

    try:
        select_plan_cases(plan, ["screen", "screen"])
    except ValueError as exc:
        assert "must not be repeated" in str(exc)
    else:
        raise AssertionError("duplicate case id was accepted")


def _model_payload(role: str, tuned_shift: float) -> dict:
    token = {
        "token_id": 17,
        "token": "500",
        "decoded": "500",
        "log_prob_nats": -2.0 + tuned_shift,
        "rank": 3 if role == "base" else 1,
    }
    candidate_0 = {
        "label": "oracle",
        "aggregate_log_prob_nats": -2.0 + tuned_shift,
        "tokens": [token],
    }
    candidate_1 = {
        "label": "distractor",
        "aggregate_log_prob_nats": -3.0,
        "tokens": [{**token, "log_prob_nats": -3.0}],
    }
    return {
        "role": role,
        "cases": [
            {
                "id": "case",
                "protocol": "test",
                "score_reduction": "sum",
                "alignment": {"input_ids_sha256": "same"},
                "layers": [
                    {
                        "layer": 0,
                        "label": "residual_after_layer_0",
                        "margin_nats": candidate_0["aggregate_log_prob_nats"]
                        - candidate_1["aggregate_log_prob_nats"],
                        "candidates": [candidate_0, candidate_1],
                    }
                ],
            }
        ],
    }


def test_comparison_is_tuned_minus_base_for_tokens_and_margin() -> None:
    comparison = compare_model_results(_model_payload("base", 0.0), _model_payload("tuned", 0.75))
    case = comparison["cases"][0]
    layer = case["layers"][0]

    assert layer["delta_margin_nats"] == 0.75
    assert layer["candidates"][0]["delta_log_prob_nats"] == 0.75
    assert layer["candidates"][0]["tokens"][0]["delta_log_prob_nats"] == 0.75
    assert case["summary"]["peak_absolute_delta_margin_layer"] == 0


def test_attention_comparison_preserves_signed_patch_delta_and_target_metrics() -> None:
    base = {
        "cases": [
            {
                "id": "case",
                "protocol": "test",
                "alignment": {"input_ids_sha256": "same"},
                "image_sizes": [[20, 10]],
                "image_grids": [[1, 2]],
                "maps": {"direct_attention": [[0.75, 0.25]]},
                "target": {
                    "direct_attention": {
                        "target_mass": 0.25,
                        "target_lift": 0.5,
                        "peak_inside_target": False,
                    }
                },
            }
        ]
    }
    tuned = {
        "cases": [
            {
                "id": "case",
                "protocol": "test",
                "alignment": {"input_ids_sha256": "same"},
                "image_sizes": [[20, 10]],
                "image_grids": [[1, 2]],
                "maps": {"direct_attention": [[0.4, 0.6]]},
                "target": {
                    "direct_attention": {
                        "target_mass": 0.6,
                        "target_lift": 1.2,
                        "peak_inside_target": True,
                    }
                },
            }
        ]
    }

    result = compare_attention_results(base, tuned)["cases"][0]

    assert result["methods"]["direct_attention"]["delta"] == [[-0.35, 0.35]]
    assert result["target_delta"]["direct_attention"]["delta_target_mass"] == 0.35
    assert result["target_delta"]["direct_attention"]["delta_target_lift"] == 0.7
    assert result["target_delta"]["direct_attention"]["tuned_peak_inside_target"] is True


def test_position_id_slice_keeps_rope_axis_and_selects_one_candidate() -> None:
    positions = torch.arange(3 * 2 * 5).reshape(3, 2, 5)

    selected = _slice_position_ids(positions)

    assert selected.shape == (3, 1, 5)
    assert torch.equal(selected[:, 0], positions[:, 0])


def test_checkpoint_revision_reads_commit_hash_instead_of_etag(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    metadata = model_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    commit = "1" * 40
    etag = "2" * 40
    (metadata / "config.json.metadata").write_text(f"{commit}\n{etag}\n123.0\n")

    engine = InstrumentedHolo(Settings(model_path=model_path))

    assert engine.checkpoint_revision() == commit
