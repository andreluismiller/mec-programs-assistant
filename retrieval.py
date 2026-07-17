
from pathlib import Path

import numpy as np

from embeder import Embeder
from sqlitesearch import TextSearchIndex, VectorSearchIndex

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_ROOT / "faq_mec.db")

TEXT_FIELDS = ["nome", "pergunta", "resposta"]
KEYWORD_FIELDS = ["sigla", "termos", "sinonimos"]
ID_FIELD = "doc_id"

embeder = Embeder()

text_index = TextSearchIndex(
    text_fields=TEXT_FIELDS,
    keyword_fields=KEYWORD_FIELDS,
    id_field=ID_FIELD,
    db_path=DB_PATH,
)

vector_index = VectorSearchIndex(
    keyword_fields=KEYWORD_FIELDS,
    id_field=ID_FIELD,
    mode="ivf",
    db_path=DB_PATH,
)


def text_search(query: str, k: int = 5) -> list[dict]:
    """Busca puramente textual (BM25 via FTS5)."""
    results = text_index.search(query, output_ids=True)
    return results[:k]


def vector_search(query: str, k: int = 5, query_vector: np.ndarray | None = None) -> list[dict]:
    """
    Busca puramente vetorial (similaridade de cosseno sobre os embeddings).

    Se query_vector já vier calculado (ex.: pré-computado em batch para
    avaliação em massa), pula o encode individual da query.
    """
    if query_vector is None:
        query_vector = embeder.encode("query: " + query)
    results = vector_index.search(query_vector, output_ids=True)
    return results[:k]


def hybrid_search(
    query: str, k: int = 5, rrf_k: int = 60, query_vector: np.ndarray | None = None
) -> list[dict]:
    """
    Combina busca textual e vetorial via Reciprocal Rank Fusion (RRF):
    cada resultado ganha 1 / (rrf_k + posição no ranking + 1) em cada
    lista em que aparece; os escores são somados e reordenados.

    Aceita query_vector pré-computado pelo mesmo motivo de vector_search.
    """
    text_results = text_index.search(query, output_ids=True)
    if query_vector is None:
        query_vector = embeder.encode("query: " + query)
    vector_results = vector_index.search(query_vector, output_ids=True)

    scores: dict[str, float] = {}
    for rank, r in enumerate(text_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (rrf_k + rank + 1)
    for rank, r in enumerate(vector_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (rrf_k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{"id": doc_id, "score": scores[doc_id]} for doc_id in ranked_ids]


if __name__ == "__main__":
    query = "Quem pode ser atendido pela educação especial?"

    print("--- Busca textual ---")
    for r in text_search(query):
        print("-", r["id"], "| score:", r["score"])

    print("\n--- Busca vetorial ---")
    for r in vector_search(query):
        print("-", r["id"], "| score:", r["score"])

    print("\n--- Busca híbrida (RRF) ---")
    for r in hybrid_search(query):
        print("-", r["id"], "| score:", r["score"])