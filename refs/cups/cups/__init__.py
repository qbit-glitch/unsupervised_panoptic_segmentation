from .config import get_default_config

try:
    from .pl_model_pseudo import build_model_pseudo
    from .pl_model_self import build_model_self
except ModuleNotFoundError as exc:
    if exc.name != "detectron2":
        raise
    # Pseudo-label generation only needs the config loader. Training entrypoints
    # still import these builders directly in environments with Detectron2.
    build_model_pseudo = None
    build_model_self = None
