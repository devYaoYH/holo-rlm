from pathlib import Path

import pytest

from demo.backends import ScriptedBackend
from demo.runner import capture_run


@pytest.fixture(scope="session")
def valid_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bundles")
    bundle, result = capture_run(backend=ScriptedBackend(), output_root=root, seed=0, max_steps=6)
    assert result["annotations"]["task_success"] is True
    return bundle
