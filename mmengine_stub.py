"""Let torch.load open mmengine-flavoured checkpoints without installing mmengine.

The MM-GroundingDINO .pth files pickle `mmengine.config.ConfigDict` (in `meta`) and
`mmengine.logging.HistoryBuffer` (in `message_hub`). We only ever want `state_dict`,
so we fabricate permissive placeholder classes for anything under mmengine/mmdet/mmcv
rather than pulling in the whole framework.

Import this module before calling torch.load.
"""

import importlib.abc
import importlib.machinery
import sys
import types

_STUBBED_ROOTS = ("mmengine", "mmdet", "mmcv")


class _StubMeta(type):
    """Swallow class-level attribute lookups the unpickler makes (e.g. HistoryBuffer.min)."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: _Stub()


class _Stub(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def __repr__(self):
        return f"<{type(self).__name__}>"


class _StubDict(dict, metaclass=_StubMeta):
    pass


def _module_getattr(module_name, module):
    def getter(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        base = _StubDict if "Config" in attr else _Stub
        cls = _StubMeta(attr, (base,), {"__module__": module_name})
        setattr(module, attr, cls)
        return cls

    return getter


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = _module_getattr(spec.name, module)
        return module

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        return None


def install():
    """Idempotently register the stub finder."""
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


install()
