import lichtfeld as lf

from ..operators.generate_masks import GenerateMasksOperator
from ..operators.preview_mask import PreviewMaskOperator


class SAMPanel(lf.ui.Panel):
    label = "SAM Segmentation"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200

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

        ui.separator()
        ui.label("Status: check LichtFeld log for progress")
