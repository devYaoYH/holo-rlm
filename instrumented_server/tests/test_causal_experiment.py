from types import SimpleNamespace

import pytest
from instrumented_holo.activation_patching import (
    changed_pixel_fraction,
    coordinate_center,
    coordinate_tool_candidate,
    overlapping_patch_indices,
    require_patch_alignment,
    swap_equal_tiles,
)
from PIL import Image


def test_coordinate_center_uses_normalized_screen_space() -> None:
    assert coordinate_center((1289, 563, 1489, 702), (2880, 1800)) == (482, 351)
    assert coordinate_center((1069, 563, 1269, 702), (2880, 1800)) == (406, 351)


def test_tile_swap_is_exact_and_preserves_canvas() -> None:
    image = Image.new("RGB", (6, 2), "black")
    for x in range(2):
        for y in range(2):
            image.putpixel((x, y), (255, 0, 0))
            image.putpixel((x + 4, y), (0, 0, 255))
    swapped = swap_equal_tiles(image, (0, 0, 2, 2), (4, 0, 6, 2))
    assert swapped.size == image.size
    assert swapped.getpixel((0, 0)) == (0, 0, 255)
    assert swapped.getpixel((4, 0)) == (255, 0, 0)
    assert swapped.getpixel((3, 0)) == (0, 0, 0)
    assert changed_pixel_fraction(image, swapped) == pytest.approx(8 / 12)


def test_tile_swap_rejects_ambiguous_geometry() -> None:
    image = Image.new("RGB", (8, 4), "black")
    with pytest.raises(ValueError, match="identical dimensions"):
        swap_equal_tiles(image, (0, 0, 2, 2), (4, 0, 7, 2))
    with pytest.raises(ValueError, match="must not overlap"):
        swap_equal_tiles(image, (0, 0, 3, 3), (2, 0, 5, 3))
    with pytest.raises(ValueError, match="outside"):
        swap_equal_tiles(image, (0, 0, 2, 2), (7, 0, 9, 2))


def test_patch_selection_uses_fractional_overlap() -> None:
    # 20x10 image split into a 2x4 grid. This box crosses six patch cells.
    assert overlapping_patch_indices((4, 4, 11, 6), (20, 10), (2, 4)) == (0, 1, 2, 4, 5, 6)
    assert overlapping_patch_indices((4, 4, 11, 6), (20, 10), (2, 4), limit=4) == (0, 1, 2, 5)


def test_coordinate_candidate_scores_only_coordinate_characters() -> None:
    candidate = coordinate_tool_candidate("target", (482, 351))
    assert tuple(candidate.text[start:end] for start, end in candidate.scored_spans) == ("482", "351")


def test_patch_alignment_requires_matching_sequence_and_image_geometry() -> None:
    clean = SimpleNamespace(
        input_ids=SimpleNamespace(shape=(2, 100)),
        prediction_positions=((90,), (90,)),
        image_positions=((10, 11),),
        image_grids=((1, 2),),
    )
    corrupted = SimpleNamespace(
        input_ids=SimpleNamespace(shape=(2, 100)),
        prediction_positions=((90,), (90,)),
        image_positions=((10, 11),),
        image_grids=((1, 2),),
    )
    require_patch_alignment(clean, corrupted)
    corrupted.image_grids = ((2, 1),)
    with pytest.raises(ValueError, match="image grids"):
        require_patch_alignment(clean, corrupted)
