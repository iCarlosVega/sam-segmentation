import shutil
import tempfile
import threading
from pathlib import Path

import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import StringProperty

_CACHE_DIR = Path.home() / ".lichtfeld" / "cache" / "grounded_sam2"


mask_state: dict = {
    "running": False,
    "current": 0,
    "total": 0,
    "error": None,
    "last_finished": 0,
}


class GenerateMasksOperator(Operator):
    label = "Generate All Masks"
    description = "Run Grounded SAM 2 text segmentation across all video frames"

    prompt: str = StringProperty(default="", maxlen=256)

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        if mask_state["running"]:
            lf.log.warn("SAM Segment: mask generation already running")
            return {"CANCELLED"}

        if not self.prompt:
            lf.log.warn("SAM Segment: prompt is empty — type a prompt first")
            return {"CANCELLED"}

        scene = lf.get_scene()
        camera_nodes = sorted(
            scene.get_nodes(lf.scene.NodeType.CAMERA),
            key=lambda n: n.image_path,
        )
        if not camera_nodes:
            lf.log.warn("SAM Segment: no camera nodes found in scene")
            return {"CANCELLED"}

        dataset_root = Path(lf.dataset_params().data_path)
        masks_dir = dataset_root / "masks"
        masks_dir.mkdir(exist_ok=True)

        mask_state["running"] = True
        mask_state["current"] = 0
        mask_state["total"] = len(camera_nodes)
        mask_state["error"] = None

        threading.Thread(
            target=_run_mask_generation,
            args=(camera_nodes, self.prompt, masks_dir),
            daemon=True,
        ).start()
        return {"FINISHED"}


def _run_mask_generation(camera_nodes, prompt, masks_dir):
    from PIL import Image
    from .grounded_sam2_backend import load_predictor, run_video_segmentation

    frames_tmp = tempfile.mkdtemp(prefix="sam_frames_")
    total = mask_state["total"]
    try:
        for i, node in enumerate(camera_nodes):
            ext = Path(node.image_path).suffix
            dst = Path(frames_tmp) / f"{i:05d}{ext}"
            dst.write_bytes(Path(node.image_path).read_bytes())

        predictor = load_predictor(_CACHE_DIR)

        for frame_idx, mask in run_video_segmentation(
            predictor, Path(frames_tmp), prompt
        ):
            stem = Path(camera_nodes[frame_idx].image_path).stem
            Image.fromarray(mask).save(masks_dir / (stem + ".png"))
            mask_state["current"] = frame_idx + 1
            lf.log.info(f"[{frame_idx + 1}/{total}] {stem}.png")

        mask_state["last_finished"] = total
        lf.log.info(f"Masks saved to {masks_dir}. Ready to train.")
    except Exception as e:
        mask_state["error"] = str(e)
        lf.log.error(f"SAM Segment: mask generation failed: {e}")
    finally:
        shutil.rmtree(frames_tmp, ignore_errors=True)
        mask_state["running"] = False
