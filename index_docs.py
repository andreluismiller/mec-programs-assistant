"""
Gera embeddings das FAQs do MEC em batches de 50 e alimenta os índices
híbridos (texto + vetor) do sqlitesearch.

Pré-requisitos:
- models/Xenova/multilingual-e5-base já baixado (ver download.py)
- pip install sqlitesearch tqdm numpy tokenizers onnxruntime
"""
from pathlib import Path
import json
import numpy as np
from tqdm.auto import tqdm

from embeder import Embeder
from sqlitesearch import TextSearchIndex, VectorSearchIndex


# ------------------------------------------------------------------
# 1. Carrega o dataset
#    (ajuste o caminho para o seu arquivo json real)
# ------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "data" / "raw" / "dataset_part.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    documents = json.load(f)

print(f"{len(documents)} documentos carregados de {DATASET_PATH}")


# ------------------------------------------------------------------
# 2. Monta o texto que será embedado, com os MESMOS campos usados no
#    índice textual (nome, pergunta, resposta).
#
#    Modelos da família E5 (multilingual-e5-base é um deles) esperam
#    prefixos de instrução:
#      - "passage: " para textos que vão para o índice (documentos)
#      - "query: "   para a pergunta do usuário na hora da busca
#    Sem isso a qualidade da recuperação cai bastante.
# ------------------------------------------------------------------
def build_passage_text(doc: dict) -> str:
    partes = [doc.get("nome", ""), doc.get("pergunta", ""), doc.get("resposta", "")]
    return "passage: " + " ".join(p for p in partes if p)


texts = [build_passage_text(doc) for doc in documents]


# ------------------------------------------------------------------
# 3. Gera os embeddings em batches de 50
# ------------------------------------------------------------------
embeder = Embeder()  # usa models/Xenova/multilingual-e5-base por padrão

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = embeder.encode_batch(batch)  # já normalizado (ver embeder.py)
    vectors.extend(batch_vectors)

print(len(vectors))
X = np.array(vectors)
print("Shape de X:", X.shape)  # (n_documentos, dim)


# ------------------------------------------------------------------
# 4. Indexação híbrida (texto + vetor) no mesmo arquivo .db
#
#    Atenção:
#    - id_field é uma STRING ("doc_id"), não uma lista, e é passado
#      no construtor do índice (não no .fit()).
#    - keyword_fields aqui usam "sinonimos" (sem acento), pois é
#      assim que a chave aparece nos documentos do dataset. Se algum
#      documento tiver a chave "sinônimos" (com acento) por engano,
#      o índice simplesmente vai ignorar aquele campo nele.
#    - text_index e vector_index precisam apontar para o MESMO
#      db_path e usar o MESMO id_field para que os resultados das
#      duas buscas possam ser combinados (RRF) pelo doc_id.
# ------------------------------------------------------------------
DB_PATH = "faq_mec.db"

vector_index = VectorSearchIndex(
    keyword_fields=["sigla", "termos", "sinonimos"],
    id_field="doc_id",
    db_path=DB_PATH,
)
vector_index.fit(X, documents)

text_index = TextSearchIndex(
    text_fields=["nome", "pergunta", "resposta"],
    keyword_fields=["sigla", "termos", "sinonimos"],
    id_field="doc_id",
    db_path=DB_PATH,
)
text_index.fit(documents)

print(f"Índice híbrido salvo em {DB_PATH}")


# ------------------------------------------------------------------
# 5. (Bônus) Busca híbrida simples com Reciprocal Rank Fusion (RRF)
#    Útil para já deixar pronto o componente de retrieval que vai
#    entrar na etapa de avaliação (recall@k, MRR, etc.)
# ------------------------------------------------------------------
def hybrid_search(query: str, k: int = 5, rrf_k: int = 60):
    query_vector = embeder.encode("query: " + query)

    text_results = text_index.search(query, output_ids=True)
    vector_results = vector_index.search(query_vector, output_ids=True)

    scores = {}
    for rank, r in enumerate(text_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (rrf_k + rank + 1)
    for rank, r in enumerate(vector_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (rrf_k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    by_id = {doc["doc_id"]: doc for doc in documents}
    return [by_id[i] for i in ranked_ids if i in by_id]


if __name__ == "__main__":
    exemplo = hybrid_search("Quem pode ser atendido pela educação especial?")
    for doc in exemplo:
        print("-", doc["nome"], "|", doc["pergunta"])



# import json
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from fastembed import TextEmbedding

# # 1. Nossa classe customizada e à prova de falhas para o FastEmbed
# class FastEmbedCustomFunction(EmbeddingFunction):
#     def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
#         # Inicializa o modelo ONNX diretamente pela biblioteca oficial
#         self.model = TextEmbedding(model_name=model_name)

#     def __call__(self, input: Documents) -> Embeddings:
#         # Gera os vetores e converte para listas de floats (padrão exigido pelo Chroma)
#         vetores = self.model.embed(input)
#         return [vetor.tolist() for vetor in vetores]

# def ingerir_dados_faq(caminho_json: str, caminho_db: str, nome_colecao: str, tamanho_batch: int = 100):
#     client = chromadb.PersistentClient(path=caminho_db)
    
#     # 2. Utilizamos a nossa função ao invés daquela nativa do Chroma que causou erro
#     funcao_embedding = FastEmbedCustomFunction()
    
#     collection = client.get_or_create_collection(
#         name=nome_colecao,
#         embedding_function=funcao_embedding
#     )
    
#     # ... (O RESTANTE DO CÓDIGO CONTINUA EXATAMENTE IGUAL) ...
#     with open(caminho_json, 'r', encoding='utf-8') as f:
#         dados = json.load(f)
        
#     total_documentos = len(dados)
#     print(f"Iniciando a ingestão de {total_documentos} documentos...")
    
#     for i in range(0, total_documentos, tamanho_batch):
#         lote_atual = dados[i:i+tamanho_batch]
#         ids = []
#         documentos = []
#         metadados = []
        
#         for item in lote_atual:
#             texto_documento = f"Pergunta: {item['pergunta']} \nResposta: {item['resposta']}"
#             meta = {
#                 "programa": item.get("programa", ""),
#                 "agrupamento": item.get("agrupamento", ""),
#                 "nome": item.get("nome", ""),
#                 "termos": item.get("termos", ""),
#                 "sinonimos": item.get("sinonimos", "")
#             }
#             ids.append(item["id"])
#             documentos.append(texto_documento)
#             metadados.append(meta)
            
#         collection.add(ids=ids, documents=documentos, metadatas=metadados)
#         print(f"Lote processado: {i + len(lote_atual)}/{total_documentos} itens inseridos.")
        
#     print("Ingestão concluída com sucesso!")

# if __name__ == "__main__":
#     ARQUIVO_JSON = "mec_faq.json"
#     DIRETORIO_DB = "./chroma_db"
#     COLECAO = "faq_mec_collection"
    
#     ingerir_dados_faq(caminho_json=ARQUIVO_JSON, caminho_db=DIRETORIO_DB, nome_colecao=COLECAO)