from pathlib import Path

import lichtfeld as lf

from ..operators.generate_masks import GenerateMasksOperator, mask_state
from ..operators.preview_mask import PreviewMaskOperator


class SAMPanel(lf.ui.Panel):
    id = "sam_segment.sam_panel"
    label = "SAM Segmentation"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200
    template = str(Path(__file__).resolve().with_name("sam_panel.rml"))
    update_interval_ms = 100

    def __init__(self):
        self._prompt = ""

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        ui.heading("SAM Segmentation")

        changed, new_prompt = ui.input_text_with_hint(
            "##sam_prompt", "e.g. action figure", self._prompt
        )
        if changed:
            self._prompt = new_prompt

        ui.separator()

        running = mask_state["running"]

        if not running:
            if PreviewMaskOperator.poll(None):
                if ui.button("Preview on Selected Camera", (-1, 0)):
                    lf.ui.ops.invoke(
                        PreviewMaskOperator._class_id(), prompt=self._prompt
                    )
            else:
                ui.text_disabled("Select a camera node to enable preview")

            if ui.button("Generate All Masks", (-1, 0)):
                lf.ui.ops.invoke(
                    GenerateMasksOperator._class_id(), prompt=self._prompt
                )
        else:
            ui.text_disabled("Mask generation in progress — please wait")

        ui.separator()

        if running:
            cur = mask_state["current"]
            total = mask_state["total"]
            fraction = cur / max(total, 1)
            ui.progress_bar(fraction, overlay=f"{cur}/{total} frames")
        elif mask_state["error"]:
            ui.label(f"Error: {mask_state['error']}")
        elif mask_state["last_finished"] > 0:
            ui.label(f"Done — {mask_state['last_finished']} masks saved")
