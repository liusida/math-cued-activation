"""Response generation backends."""

from .vllm import generate_from_config

__all__ = ["generate_from_config"]
