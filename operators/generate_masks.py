import shutil
import tempfile
from pathlib import Path

import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import StringProperty

_CACHE_DIR = Path.home() / ".lichtfeld" / "cache" / "sam3"


class GenerateMasksOperator(Operator):
    label = "Generate All Masks"
    description = "Run SAM 3 text segmentation across all video frames"

    prompt: str = StringProperty(default="", maxlen=256)

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
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

        frames_tmp = tempfile.mkdtemp()
        try:
            for i, node in enumerate(camera_nodes):
                ext = Path(node.image_path).suffix
                dst = Path(frames_tmp) / f"{i:05d}{ext}"
                dst.write_bytes(Path(node.image_path).read_bytes())

            from .grounded_sam2_backend import load_predictor, run_video_segmentation
            predictor = load_predictor(_CACHE_DIR)
            total = len(camera_nodes)
            from PIL import Image

            for frame_idx, mask in run_video_segmentation(predictor, Path(frames_tmp), self.prompt):
                stem = Path(camera_nodes[frame_idx].image_path).stem
                Image.fromarray(mask).save(masks_dir / (stem + ".png"))
                lf.log.info(f"[{frame_idx + 1}/{total}] {stem}.png")
        finally:
            shutil.rmtree(frames_tmp)

        lf.log.info(f"Masks saved to {masks_dir}. Ready to train.")
        return {"FINISHED"}
