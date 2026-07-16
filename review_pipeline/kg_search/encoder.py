from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer

from cuda_devices import default_torch_device


class QueryEncoder:
    def __init__(self, model_path: str, device: str | None = None, paper_embed_dim: int = 1024) -> None:
        if device is None:
            device = default_torch_device()
        self._model = SentenceTransformer(model_path, device=device)
        self.paper_embed_dim = paper_embed_dim

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def paper_query_vector(self, text: str) -> list[float]:
        vector = self.encode_one(text)
        if len(vector) != self.paper_embed_dim:
            raise RuntimeError(
                f"query embedding dim={len(vector)} does not match expected dim={self.paper_embed_dim}"
            )
        return vector
