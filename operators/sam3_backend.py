import sys
import types
from pathlib import Path
from typing import Iterator

import numpy as np


def _stub_triton_if_missing():
    """Inject a no-op triton stub on Windows where triton has no wheels.

    SAM 3 imports triton at module level. The @triton.jit-decorated EDT kernels
    are only reached during video memory updates; the import itself must succeed.
    """
    try:
        import triton  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    _triton = types.ModuleType("triton")
    _tl = types.ModuleType("triton.language")

    # jit / autotune — just return the function unchanged
    _triton.jit = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)
    _triton.autotune = lambda configs, key, **kw: (lambda fn: fn)
    _triton.heuristics = lambda values: (lambda fn: fn)
    _triton.cdiv = lambda a, b: (a + b - 1) // b

    class _Config:
        def __init__(self, kwargs, num_warps=4, num_stages=2, **kw):
            pass

    _triton.Config = _Config

    # triton.language — attributes referenced at definition time
    for _name in [
        "float16", "float32", "float64", "int1", "int8", "int16",
        "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "constexpr", "TRITON_MAX_TENSOR_NUMEL",
        "load", "store", "arange", "zeros", "full", "broadcast_to",
        "sum", "max", "min", "dot", "where", "sqrt", "exp", "log",
        "abs", "sigmoid", "softmax", "trans", "cat", "program_id",
        "num_programs", "multiple_of", "max_contiguous",
    ]:
        setattr(_tl, _name, None)

    _triton.language = _tl
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl


def load_predictor(cache_dir: Path):
    """Load SAM 3 VideoPredictor. Downloads checkpoint on first run."""
    _stub_triton_if_missing()
    from sam3 import build_sam3_video_predictor

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / "sam3_hiera_large.pt"
    return build_sam3_video_predictor(str(checkpoint))


def run_video_segmentation(
    predictor,
    frames_dir: Path,
    prompt: str,
) -> Iterator[tuple[int, np.ndarray]]:
    """Run SAM 3 text-prompted segmentation on a directory of numerically-named frames.

    Yields (frame_idx, mask_HxW_uint8) for each frame.
    mask: 255 = keep object, 0 = discard.
    """
    state = predictor.init_state(video_path=str(frames_dir))
    predictor.add_new_prompts(state, frame_idx=0, text=prompt)
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
        yield frame_idx, mask
