import sys
import types
from pathlib import Path
from typing import Iterator

import numpy as np


class _PermissiveStub:
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
    """SAM 2 + torch may still probe triton on Windows. Harmless when triton
    is genuinely absent — the stub satisfies attribute checks at import time."""
    try:
        import triton  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    _stub_file = __file__

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


def load_predictor(cache_dir: Path) -> dict:
    """Load GroundingDINO + SAM 2 VideoPredictor. Downloads weights on first run."""
    _stub_triton_if_missing()

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2_video_predictor

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gdino_id = "IDEA-Research/grounding-dino-tiny"
    gdino_processor = AutoProcessor.from_pretrained(gdino_id, cache_dir=str(cache_dir))
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_id, cache_dir=str(cache_dir)
    ).to(device)

    sam2_checkpoint = "facebook/sam2-hiera-small"
    sam2_predictor = build_sam2_video_predictor(sam2_checkpoint, device=device)

    return {
        "gdino_processor": gdino_processor,
        "gdino_model": gdino_model,
        "sam2": sam2_predictor,
        "device": device,
    }


def run_video_segmentation(
    bundle: dict,
    frames_dir: Path,
    prompt: str,
) -> Iterator[tuple[int, np.ndarray]]:
    """Run GroundingDINO on frame 0 to get a box, then SAM 2 propagates the mask.

    Yields (frame_idx, mask_HxW_uint8) for each frame.
    mask: 255 = keep object, 0 = discard.
    """
    import torch
    from PIL import Image

    device = bundle["device"]

    frame0 = sorted(frames_dir.iterdir())[0]
    image = Image.open(frame0).convert("RGB")
    text = prompt if prompt.endswith(".") else prompt + "."
    inputs = bundle["gdino_processor"](
        images=image, text=text, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = bundle["gdino_model"](**inputs)
    results = bundle["gdino_processor"].post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.35,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]
    if len(results["boxes"]) == 0:
        raise RuntimeError(f"GroundingDINO found no match for prompt: {prompt!r}")
    box = results["boxes"][0].cpu().numpy()

    sam2 = bundle["sam2"]
    state = sam2.init_state(video_path=str(frames_dir))
    sam2.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
    for frame_idx, _obj_ids, mask_logits in sam2.propagate_in_video(state):
        mask = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
        yield frame_idx, mask
