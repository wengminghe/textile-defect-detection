from typing import Optional, Callable
BACKBONES = {}


def model(name: Optional[str] = None) -> Callable:
    def _register(fn: Callable) -> Callable:
        key = name if name is not None else fn.__name__
        if key in BACKBONES:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BACKBONES[key] = fn
        return fn

    return _register


from . import resnet
