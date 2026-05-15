import shutil
import tempfile
from pathlib import Path

import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import StringProperty

_CACHE_DIR = Path.home() / ".lichtfeld" / "cache" / "sam3"


class PreviewMaskOperator(Operator):
    label = "Preview on Selected Camera"
    description = "Run SAM 3 on the selected camera image and log mask path"

    prompt: str = StringProperty(default="", maxlen=256)

    @classmethod
    def poll(cls, context) -> bool:
        if not lf.has_scene():
            return False
        scene = lf.get_scene()
        selected = lf.get_selected_node_names()
        return any(
            scene.get_node(n) and scene.get_node(n).type == lf.scene.NodeType.CAMERA
            for n in selected
        )

    def execute(self, context) -> set:
        if not self.prompt:
            lf.log.warn("SAM Segment: prompt is empty — type a prompt first")
            return {"CANCELLED"}

        scene = lf.get_scene()
        camera_node = None
        for name in lf.get_selected_node_names():
            node = scene.get_node(name)
            if node and node.type == lf.scene.NodeType.CAMERA:
                camera_node = node
                break

        if camera_node is None:
            return {"CANCELLED"}

        frames_tmp = tempfile.mkdtemp()
        try:
            ext = Path(camera_node.image_path).suffix
            frame_path = Path(frames_tmp) / f"00000{ext}"
            frame_path.write_bytes(Path(camera_node.image_path).read_bytes())

            from .grounded_sam2_backend import load_predictor, run_video_segmentation
            predictor = load_predictor(_CACHE_DIR)
            _frame_idx, mask = next(
                run_video_segmentation(predictor, Path(frames_tmp), self.prompt)
            )
        finally:
            shutil.rmtree(frames_tmp)

        mask_tmp = tempfile.mkdtemp()
        mask_path = Path(mask_tmp) / "preview_mask.png"
        from PIL import Image

        Image.fromarray(mask).save(mask_path)
        lf.log.info(f"Preview mask saved to {mask_path}. Open to inspect.")
        return {"FINISHED"}
