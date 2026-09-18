import base64
import json

from PIL import Image

from demo.backends import ScriptedBackend
from demo.runner import capture_run


def test_redaction_precedes_persistence_and_model_request(tmp_path) -> None:
    bundle, _ = capture_run(
        backend=ScriptedBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=6,
        redactions=((0, 0, 100, 100),),
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["redaction"]["applied"] is True
    with Image.open(bundle / "frames/0000-model-input.png") as image:
        assert image.getpixel((50, 50)) == (0, 0, 0)
    request = json.loads((bundle / "requests/0000.json").read_text())
    encoded = request["messages"][-1]["content"][-1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == (bundle / "frames/0000-model-input.png").read_bytes()
