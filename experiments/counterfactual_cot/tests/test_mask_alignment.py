from PIL import Image


def test_mask_non_empty_non_full(tmp_path):
    path = tmp_path / "mask.png"
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    mask.save(path)
    with Image.open(path) as loaded:
        assert loaded.getextrema() == (0, 255)
