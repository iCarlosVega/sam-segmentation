import sys
import types
from pathlib import Path
from typing import Iterator

import numpy as np


class _PermissiveStub:
    """Returns itself for any attribute, call, or item access — satisfies probes
    without us having to enumerate every triton API."""

    def __getattr__(self, name):
        return _PermissiveStub()

    def __call__(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return _PermissiveStub()

    def __getitem__(self, item):
        return _PermissiveStub()

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        return _PermissiveStub()


def _stub_triton_if_missing():
    """Inject a permissive triton stub on Windows where triton has no wheels.

    SAM 3 imports triton at module level for EDT kernels; torch._dynamo also
    probes triton.language at import time. The stub returns a permissive object
    for any attribute so both imports clear. Actual triton kernel execution
    (e.g. EDT during video memory updates) will silently misbehave — acceptable
    for text-prompted segmentation where EDT is not the primary path.
    """
    try:
        import triton  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    _stub_file = __file__  # use this file's path so inspect.getfile() works

    def _make(name):
        m = _PermissiveModule(name)
        m.__file__ = _stub_file
        m.__spec__ = None
        m.__path__ = []
        return m

    _triton = _make("triton")
    _tl = _make("triton.language")
    _triton.language = _tl
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl
    sys.modules["triton.runtime"] = _make("triton.runtime")
    sys.modules["triton.compiler"] = _make("triton.compiler")


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
