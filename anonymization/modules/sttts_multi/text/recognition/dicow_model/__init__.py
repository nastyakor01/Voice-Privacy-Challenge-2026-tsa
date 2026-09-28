"""DiCoW model code vendored from TS-ASR-Whisper (transformers 4.55 compatible).

The HuggingFace hub repo BUT-FIT/DiCoW_v3_2 ships remote modeling files written
for transformers ~4.42 (WHISPER_ATTENTION_CLASSES). Loading those via
trust_remote_code=True fails on 4.47/4.55. Weights still come from the hub;
only the Python classes are local.
"""

from .config import DiCoWConfig
from .modeling_dicow import DiCoWForConditionalGeneration

__all__ = ["DiCoWConfig", "DiCoWForConditionalGeneration"]
