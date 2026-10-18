from __future__ import annotations

"""
Embedder: local neural text embeddings via bge-small-en-v1.5.

- Runs fully offline from models/bge-small-en-v1.5
- GPU-accelerated when CUDA is present, batched for throughput
- L2-normalized vectors so cosine similarity == dot product
"""

import threading
from pathlib import Path

import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

DEFAULT_MODEL = "bge-small-en-v1.5"
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 256


#: Below this much GPU memory the embedder stays on the CPU. Measured live
#: on an 8 GB card: the embedder took the GPU next to the conversation model
#: and the chat model then had no room to load, so the assistant answered
#: with no reply at all. Retrieval is a background job; the conversation is
#: the product.
_GPU_MEMORY_FLOOR_MB = 12_000


def _pick_device() -> str:
    """Where the embedder runs: cuda, cpu, or NIX_SEMANTIC_DEVICE to decide."""
    import os

    wanted = (os.environ.get("NIX_SEMANTIC_DEVICE") or "auto").strip().lower()
    if wanted in {"cpu", "cuda"}:
        return wanted
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        try:
            total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except Exception:  # noqa: BLE001
            return "cpu"
        return "cuda" if total_mb >= _GPU_MEMORY_FLOOR_MB else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class Embedder:
    """
    Thread-safe local embedding engine.

    One model instance is shared process-wide; encode() may be called
    from any thread (the underlying torch forward pass is serialized
    by an internal lock).
    """

    _instances: dict[tuple, "Embedder"] = {}
    _instances_lock = threading.Lock()

    def __new__(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        prefer_finetuned: bool = True,
    ):
        key = (model_name, prefer_finetuned)
        with cls._instances_lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        prefer_finetuned: bool = True,
    ):
        if self._initialized:
            return
        self._initialized = True

        from sentence_transformers import SentenceTransformer

        # Production resolves to the fine-tuned backbone when one has
        # been persisted; training passes prefer_finetuned=False so it
        # always fine-tunes FROM PRISTINE (no drift across runs).
        source = None
        if prefer_finetuned:
            finetuned_dir = _MODEL_DIR / f"{model_name}-finetuned"
            if finetuned_dir.exists():
                source = str(finetuned_dir)
        if source is None:
            local_path = _MODEL_DIR / model_name
            source = (
                str(local_path) if local_path.exists() else model_name
            )

        self.model_name = model_name
        self.finetuned = prefer_finetuned and "-finetuned" in source
        self.model = SentenceTransformer(source)
        self.dimension = int(self.model.get_embedding_dimension())

        self.device = _pick_device()
        try:
            self.model.to(self.device)
        except Exception:
            # A GPU that cannot take the embedder must not take the
            # conversation model down with it: fall back to CPU.
            self.device = "cpu"
            self.model.to("cpu")

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = BATCH_SIZE,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        cleaned = [t if isinstance(t, str) else str(t) for t in texts]
        with self._lock:
            vectors = self.model.encode(
                cleaned,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> dict:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "finetuned": self.finetuned,
        }

    def save_finetuned(self) -> str:
        """Persist the (possibly fine-tuned) backbone for production use."""
        out_dir = _MODEL_DIR / f"{self.model_name}-finetuned"
        self.model.save(str(out_dir))
        return str(out_dir)
