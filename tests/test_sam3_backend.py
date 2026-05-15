"""Unit tests for sam3_backend — pure numpy/stdlib, no SAM 3 or LichtFeld needed."""
import numpy as np


def _convert_mask_logits(logit_array: np.ndarray) -> np.ndarray:
    """Mirrors the conversion in sam3_backend.run_video_segmentation."""
    return (logit_array > 0).astype(np.uint8) * 255


def test_mask_conversion_positive_becomes_255():
    logits = np.array([0.5, 1.2, 0.001])
    result = _convert_mask_logits(logits)
    assert list(result) == [255, 255, 255]
    assert result.dtype == np.uint8


def test_mask_conversion_negative_becomes_0():
    logits = np.array([-0.5, -1.2, -0.001])
    result = _convert_mask_logits(logits)
    assert list(result) == [0, 0, 0]


def test_mask_conversion_mixed():
    logits = np.array([0.5, -0.3, 1.2, -0.1])
    result = _convert_mask_logits(logits)
    assert list(result) == [255, 0, 255, 0]


def test_mask_conversion_zero_becomes_0():
    logits = np.array([0.0])
    result = _convert_mask_logits(logits)
    assert list(result) == [0]


def test_stem_stripping():
    """Mirrors Path(image_path).stem + '.png' logic used in generate_masks."""
    from pathlib import Path
    assert Path("images/00001.jpg").stem + ".png" == "00001.png"
    assert Path("/data/images/frame_042.jpeg").stem + ".png" == "frame_042.png"
    assert Path("00001.jpg").stem + ".png" == "00001.png"
