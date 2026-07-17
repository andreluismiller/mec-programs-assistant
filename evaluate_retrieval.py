
import inspect
import json
import re
from pathlib import Path

from tqdm.auto import tqdm

from retrieval import embeder, text_search, vector_search, hybrid_search

PROJECT_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"

EMBED_BATCH_SIZE = 50  # mesmo tamanho de batch usado na geração dos embeddings do índice


# ------------------------------------------------------------------
# 1. Carga combinada dos batches de ground truth
# ------------------------------------------------------------------
def _batch_number(path: Path) -> int:
    match = re.search(r"batch(\d+)\.json$", path.name)
    return int(match.group(1)) if match else 0


def load_ground_truth(directory: Path = GROUND_TRUTH_DIR) -> list[dict]:
    batch_files = sorted(directory.glob("batch*.json"), key=_batch_number)
    if not batch_files:
        raise FileNotFoundError(f"Nenhum arquivo batch*.json encontrado em {directory}")

    ground_truth = []
    for path in batch_files:
        with open(path, encoding="utf-8") as f:
            ground_truth.extend(json.load(f))

    print(
        f"{len(ground_truth)} perguntas carregadas de {len(batch_files)} "
        f"arquivos ({batch_files[0].name}..{batch_files[-1].name}) em {directory}"
    )
    return ground_truth


# ------------------------------------------------------------------
# 2. Pré-computação em batch dos embeddings das perguntas
#    Evita uma chamada ao modelo por pergunta: com >1.000 perguntas e
#    duas funções que usam vetor (busca vetorial e híbrida), isso troca
#    ~2.000 chamadas individuais ao onnxruntime por ~20 chamadas em
#    batch, reaproveitadas pelas duas avaliações.
# ------------------------------------------------------------------
def precompute_query_vectors(ground_truth: list[dict], batch_size: int = EMBED_BATCH_SIZE):
    queries = ["query: " + item["pergunta"] for item in ground_truth]

    vectors = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Pré-computando embeddings"):
        batch = queries[i : i + batch_size]
        batch_vectors = embeder.encode_batch(batch)
        vectors.extend(batch_vectors)

    return vectors


# ------------------------------------------------------------------
# 3. Métricas
# ------------------------------------------------------------------
def hit_rate(relevance_total: list[list[bool]]) -> float:
    acertos = sum(1 for relevance in relevance_total if any(relevance))
    return acertos / len(relevance_total)


def mrr(relevance_total: list[list[bool]]) -> float:
    total_score = 0.0
    for relevance in relevance_total:
        for rank, is_relevant in enumerate(relevance):
            if is_relevant:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


# ------------------------------------------------------------------
# 4. Avaliação
# ------------------------------------------------------------------
def evaluate(ground_truth: list[dict], search_function, k: int = 5, query_vectors=None) -> dict:
    """
    Roda search_function para cada pergunta do ground truth (com barra de
    progresso tqdm), monta a lista de relevância e calcula hit_rate e mrr.

    Se search_function aceitar o parâmetro query_vector e query_vectors
    for fornecido, reaproveita os embeddings pré-computados em vez de
    reembedar a pergunta a cada chamada.
    """
    accepts_query_vector = "query_vector" in inspect.signature(search_function).parameters

    relevance_total = []
    for idx, item in enumerate(tqdm(ground_truth, desc=search_function.__name__)):
        expected_id = item["id"]
        query = item["pergunta"]

        if accepts_query_vector and query_vectors is not None:
            results = search_function(query, k=k, query_vector=query_vectors[idx])
        else:
            results = search_function(query, k=k)

        relevance = [r["id"] == expected_id for r in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    query_vectors = precompute_query_vectors(ground_truth)

    K = 5
    metodos = {
        "Busca textual": text_search,
        "Busca vetorial": vector_search,
        "Busca híbrida": hybrid_search,
    }

    resultados = {
        nome: evaluate(ground_truth, fn, k=K, query_vectors=query_vectors)
        for nome, fn in metodos.items()
    }

    print(f"\n{'Método':<16} {'Hit Rate':>10} {'MRR':>10}")
    print("-" * 38)
    for nome, metricas in resultados.items():
        print(f"{nome:<16} {metricas['hit_rate']:>10.3f} {metricas['mrr']:>10.3f}")