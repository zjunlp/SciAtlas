from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from cuda_devices import default_torch_device


class EmbeddingReranker:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        if device is None:
            device = default_torch_device()
        self._model = CrossEncoder(model_path, device=device)

    def rerank(
        self,
        query: str,
        recalled: list[dict[str, Any]],
        *,
        text_field: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not recalled:
            return []

        pairs = [[query, str(item.get(text_field) or "")] for item in recalled]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked: list[dict[str, Any]] = []
        for item, score in zip(recalled, scores):
            ranked_item = dict(item)
            ranked_item["rerank_score"] = float(score)
            ranked.append(ranked_item)

        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]
