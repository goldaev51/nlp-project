from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_kb, DEFAULT_KB_PATH


@dataclass
class TfidfKBRetriever:
    kb_df: pd.DataFrame
    max_features: int = 20000

    _vectorizer: Optional[TfidfVectorizer] = None
    _kb_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        combined_texts = (
            self.kb_df["question"].fillna("") + " " +
            self.kb_df["answer"].fillna("")
        ).tolist()

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._kb_matrix = self._vectorizer.fit_transform(combined_texts)

    @classmethod
    def from_csv(
        cls,
        path: str | None = None,
        max_features: int = 20000,
    ) -> "TfidfKBRetriever":
        kb_path = path if path is not None else str(DEFAULT_KB_PATH)
        kb_df = load_kb(kb_path)
        return cls(kb_df=kb_df, max_features=max_features)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if self._vectorizer is None or self._kb_matrix is None:
            raise RuntimeError("Retriever is not initialized properly.")

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._kb_matrix)[0]

        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = self.kb_df.iloc[idx]
            score = float(sims[idx])
            results.append(
                {
                    "id": row["id"],
                    "section": row["section"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "tags": row.get("tags", ""),
                    "score": score,
                }
            )
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.tfidf_retriever \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]
    retriever = TfidfKBRetriever.from_csv()
    hits = retriever.retrieve(query, top_k=3)

    print(f"Query: {query!r}")
    print("\nTop results:")
    for i, h in enumerate(hits, start=1):
        print(f"\n[{i}] (score={h['score']:.4f})")
        print(f"Q: {h['question']}")
        print(f"A: {h['answer']}")
        if h.get("tags"):
            print(f"Tags: {h['tags']}")
