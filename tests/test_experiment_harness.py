from __future__ import annotations

import json
import tarfile
from pathlib import Path

from demo.result_bundle import package_results, referenced_trace_ids
from demo.screenspot import ScreenSpotSample
from demo.screenspot_benchmark import select_samples


def _sample(index: int) -> ScreenSpotSample:
    return ScreenSpotSample(
        id=f"sample-{index}",
        instruction="click",
        image_path=Path(f"sample-{index}.png"),
        bbox=(0, 0, 1, 1),
        image_size=(1, 1),
        application="test",
        platform="test",
        ui_type="text",
        annotation_path=Path("test.json"),
    )


def test_screenspot_selection_is_stable_and_shardable() -> None:
    samples = tuple(_sample(index) for index in range(7))
    assert [sample.id for sample in select_samples(samples, shard_index=1, num_shards=3, offset=0, count=None)] == [
        "sample-1",
        "sample-4",
    ]
    assert [sample.id for sample in select_samples(samples, shard_index=0, num_shards=2, offset=1, count=2)] == [
        "sample-2",
        "sample-4",
    ]


def test_result_bundle_includes_only_referenced_traces(tmp_path: Path) -> None:
    run = tmp_path / "run"
    traces = tmp_path / "traces"
    run.mkdir()
    traces.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "items": [
                    {"trace_id": "trace-one"},
                    {"nested": {"instrumented_trace_id": "trace-two"}},
                ],
                "instrumented_trace_ids": ["trace-one", "trace-two"],
            }
        )
    )
    for trace_id in ("trace-one", "trace-two", "trace-unused"):
        path = traces / trace_id
        path.mkdir()
        (path / "manifest.json").write_text("{}")
    assert referenced_trace_ids(run) == ("trace-one", "trace-two")
    output = tmp_path / "results.tar.gz"
    result = package_results(run, traces, output)
    assert result["trace_count"] == 2
    assert output.is_file()
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
    assert "traces/trace-one/manifest.json" in names
    assert "traces/trace-two/manifest.json" in names
    assert "traces/trace-unused/manifest.json" not in names
