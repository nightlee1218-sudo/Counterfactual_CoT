import base64
from pathlib import Path

from PIL import Image

from experiments.counterfactual_cot.data.generate_pair_images import target_edit_instruction
from experiments.counterfactual_cot.data.image_providers import extract_image
from experiments.counterfactual_cot.data.image_quality import inspect_pair_images
from experiments.counterfactual_cot.data.mask_utils import create_candidate_mask


def test_extracts_venus_data_url():
    payload = base64.b64encode(b"png-bytes").decode("ascii")
    response = {"choices": [{"message": {"content": [{
        "type": "venus_multimodal_url",
        "venus_multimodal_url": {"mimeType": "image/png", "url": f"data:image/png;base64,{payload}"},
    }]}}]}
    data, mime = extract_image(response, timeout_s=1)
    assert data == b"png-bytes"
    assert mime == "image/png"


def test_candidate_mask_and_local_check(tmp_path: Path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), "white").save(source)
    edited = Image.new("RGB", (64, 64), "white")
    for x in range(20, 30):
        for y in range(20, 30):
            edited.putpixel((x, y), (255, 0, 0))
    edited.save(target)
    metadata = create_candidate_mask(source, target, mask, threshold=10, dilation_px=3)
    check = inspect_pair_images(source, target, mask)
    assert metadata["candidate_only"] is True
    assert check["passed"] is True


def test_target_instruction_requires_local_edit_and_preservation():
    pair = {
        "intervention": {
            "intervention_type": "object_replace",
            "entity_id": "fruit_1",
            "field": "category",
            "old_value": "banana",
            "new_value": "apple",
        },
        "target": {"prompt": "two apples on a table", "cot_text": "fruit_1.object=apple"},
        "preservation": {"must_preserve": ["fruit_1.count", "background.surface"], "allowed_to_change": ["fruit_1.category"]},
    }
    text = target_edit_instruction(pair)
    assert "Edit the provided source image locally" in text
    assert "Do not independently regenerate the whole scene" in text
    assert "fruit_1.count" in text
