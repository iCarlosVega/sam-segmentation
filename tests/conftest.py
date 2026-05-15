"""Stubs out host modules (lichtfeld, lfs_plugins) so pytest can collect
tests without LichtFeld installed."""
import enum
import sys
import types


def _noop(*a, **kw):
    return None


# --- lichtfeld stub ---
if "lichtfeld" not in sys.modules:
    _lf = types.ModuleType("lichtfeld")
    _lf.register_class = _noop
    _lf.unregister_class = _noop
    _lf.has_scene = lambda: False
    _lf.get_scene = lambda: None
    _lf.get_selected_node_names = lambda: []
    _lf.dataset_params = lambda: types.SimpleNamespace(data_path="")
    _lf.log = types.SimpleNamespace(
        info=_noop, warn=_noop, warning=_noop, error=_noop, debug=_noop
    )

    # lichtfeld.scene submodule with NodeType enum
    _scene = types.ModuleType("lichtfeld.scene")

    class _NodeType(enum.Enum):
        SPLAT = 0
        POINTCLOUD = 1
        GROUP = 2
        CAMERA = 7

    _scene.NodeType = _NodeType
    _lf.scene = _scene
    sys.modules["lichtfeld"] = _lf
    sys.modules["lichtfeld.scene"] = _scene

    # lichtfeld.ui submodule
    _ui = types.ModuleType("lichtfeld.ui")

    class _Panel:
        label = ""
        space = None
        order = 100

        def draw(self, ui):
            pass

    class _PanelSpace(enum.Enum):
        MAIN_PANEL_TAB = "MAIN_PANEL_TAB"

    _ui.Panel = _Panel
    _ui.PanelSpace = _PanelSpace
    _lf.ui = _ui
    sys.modules["lichtfeld.ui"] = _ui


# --- lfs_plugins stub ---
if "lfs_plugins" not in sys.modules:
    _lfs = types.ModuleType("lfs_plugins")
    sys.modules["lfs_plugins"] = _lfs

    # lfs_plugins.types
    _lfs_types = types.ModuleType("lfs_plugins.types")

    class _PropertyGroup:
        pass

    class _Operator(_PropertyGroup):
        label: str = ""
        description: str = ""
        options: set = set()

        @classmethod
        def _class_id(cls) -> str:
            return f"{cls.__module__}.{cls.__qualname__}"

        @classmethod
        def poll(cls, context) -> bool:
            return True

        def execute(self, context) -> set:
            return {"FINISHED"}

    _lfs_types.Operator = _Operator
    _lfs_types.PropertyGroup = _PropertyGroup
    sys.modules["lfs_plugins.types"] = _lfs_types
    _lfs.types = _lfs_types

    # lfs_plugins.props
    _lfs_props = types.ModuleType("lfs_plugins.props")

    def _StringProperty(default="", maxlen=0, **kw):
        return default

    def _FloatProperty(default=0.0, **kw):
        return default

    def _BoolProperty(default=False, **kw):
        return default

    _lfs_props.StringProperty = _StringProperty
    _lfs_props.FloatProperty = _FloatProperty
    _lfs_props.BoolProperty = _BoolProperty
    _lfs_props.PropertyGroup = _PropertyGroup
    sys.modules["lfs_plugins.props"] = _lfs_props
    _lfs.props = _lfs_props
