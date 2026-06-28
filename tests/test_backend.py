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


def test_predictor_cache_reuse():
    """load_predictor returns the cached bundle without re-running the loader."""
    from pathlib import Path
    from sam_segment.operators import grounded_sam2_backend

    sentinel = {"cached": True}
    previous = grounded_sam2_backend._predictor_cache
    grounded_sam2_backend._predictor_cache = sentinel
    try:
        result = grounded_sam2_backend.load_predictor(Path("/nonexistent"))
        assert result is sentinel
    finally:
        grounded_sam2_backend._predictor_cache = previous


def test_execute_rejects_empty_prompt():
    """Empty prompt is rejected before any state mutation."""
    from sam_segment.operators.generate_masks import (
        GenerateMasksOperator,
        mask_state,
    )
    op = GenerateMasksOperator()
    op.prompt = ""
    result = op.execute(None)
    assert result == {"CANCELLED"}
    assert mask_state["running"] is False


def test_execute_rejects_concurrent_run():
    """When a job is already running, a second execute() short-circuits."""
    from sam_segment.operators.generate_masks import (
        GenerateMasksOperator,
        mask_state,
    )
    previous_running = mask_state["running"]
    mask_state["running"] = True
    try:
        op = GenerateMasksOperator()
        op.prompt = "the car"
        result = op.execute(None)
        assert result == {"CANCELLED"}
    finally:
        mask_state["running"] = previous_running
