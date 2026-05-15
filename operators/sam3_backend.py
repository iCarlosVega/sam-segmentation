from pathlib import Path
from typing import Iterator

import numpy as np


def load_predictor(cache_dir: Path):
    """Load SAM 3 VideoPredictor. Downloads checkpoint on first run."""
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
