import lichtfeld as lf
from .operators.generate_masks import GenerateMasksOperator
from .operators.preview_mask import PreviewMaskOperator
from .panels.sam_panel import SAMPanel

_classes = [GenerateMasksOperator, PreviewMaskOperator, SAMPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("SAM Segment plugin loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("SAM Segment plugin unloaded")
