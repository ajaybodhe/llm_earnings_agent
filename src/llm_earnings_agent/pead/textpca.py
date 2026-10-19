"""Text embedding features for walk-forward models, fitted per fold.

The PCA projection is fitted on the training fold's embeddings only, so the
test quarter never influences the features it is scored with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TextPCA:
    """Callable extra_fn for model.walk_forward."""

    def __init__(self, text_dir: Path, components: int = 16) -> None:
        self.emb = np.load(Path(text_dir) / "embeddings.npy")
        meta = json.loads((Path(text_dir) / "embeddings_index.json").read_text(encoding="utf-8"))
        self.row = {eid: i for i, eid in enumerate(meta["event_ids"])}
        self.k = components

    def __call__(
        self, events: list[dict[str, Any]], train_idx: list[int], test_idx: list[int]
    ) -> dict[str, np.ndarray]:
        from sklearn.decomposition import PCA

        train_rows = [
            self.row[e] for i in train_idx if (e := events[i].get("event_id")) in self.row
        ]
        if len(train_rows) < self.k * 10:
            return {}
        pca = PCA(n_components=self.k, random_state=0).fit(self.emb[train_rows])
        out = {}
        for i in (*train_idx, *test_idx):
            eid = events[i].get("event_id")
            if eid in self.row:
                out[eid] = pca.transform(self.emb[self.row[eid]][None, :])[0]
        return out
