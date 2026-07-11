import pickle
import chromadb
from src.config import CHROMA_PATH, MINSEARCH_PATH
from src.embeddings.embedder import Embedder
# from src.embeddings.embedder import get_embedding_function


class BuscadorMEC:
    def __init__(self):
        # 1. Inicializa o nosso gerador de vetores manual
        self.embedder = Embedder()
        
        # 2. Configura a conexão com o ChromaDB (apenas armazenamento)
        self.cliente_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.colecao_chroma = self.cliente_chroma.get_collection(name="mec_faq")

        # 3. Carrega o índice do minsearch (busca lexical)
        try:
            with open(MINSEARCH_PATH, 'rb') as f:
                self.index_lexico = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Índice do minsearch não encontrado. Execute o indexer.py primeiro.")

    def busca_semantica(self, query, k=5):
        """Busca usando vetores calculados manualmente através da nossa classe Embedder."""
        # Geramos o vetor da query usando o prefixo exigido pelo E5 (query: )
        # A função gerar_vetores retorna uma lista; pegamos o primeiro item [0]
        vetor_query = list(self.embedder.gerar_vetores([f"query: {query}"]))[0]
        
        # Buscamos no Chroma passando o vetor já calculado
        resultados = self.colecao_chroma.query(
            query_embeddings=[vetor_query],
            n_results=k
        )
        
        if not resultados['ids'] or not resultados['ids'][0]:
            return []

        ids_retornados = resultados['ids'][0]
        return [{"id": id_doc, "rank": i + 1, "origem": "densa"} for i, id_doc in enumerate(ids_retornados)]

    def busca_lexica(self, query, k=5):
        """Busca puramente por palavras-chave via minsearch."""
        resultados_minsearch = self.index_lexico.search(query=query, num_results=k)
        return [{"id": doc["id"], "rank": i + 1, "origem": "lexica"} for i, doc in enumerate(resultados_minsearch)]

    def busca_hibrida(self, query, k=5, k_rrf=60):
        """Fusão dos resultados (RRF) entre semântica e léxica."""
        resultados_densos = self.busca_semantica(query, k=k)
        resultados_lexicos = self.busca_lexica(query, k=k)

        scores_rrf = {}
        
        # Reciprocal Rank Fusion
        for doc in resultados_densos:
            id_doc = doc["id"]
            scores_rrf[id_doc] = scores_rrf.get(id_doc, 0.0) + (1.0 / (k_rrf + doc["rank"]))
            
        for doc in resultados_lexicos:
            id_doc = doc["id"]
            scores_rrf[id_doc] = scores_rrf.get(id_doc, 0.0) + (1.0 / (k_rrf + doc["rank"]))

        # Ordena pelo maior score
        resultados_finais = sorted(scores_rrf.items(), key=lambda item: item[1], reverse=True)
        return [{"id": id_doc, "score_rrf": score} for id_doc, score in resultados_finais[:k]]



# class BuscadorMEC:
#     def __init__(self):
#         # Configurando ChromaDB (Busca Semântica)
#         self.cliente_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
#         self.colecao_chroma = self.cliente_chroma.get_collection(
#             name="mec_faq",
#             embedding_function=get_embedding_function()
#         )

#         # Configurando minsearch (Busca Lexical)
#         try:
#             with open(MINSEARCH_PATH, 'rb') as f:
#                 self.index_lexico = pickle.load(f)
#         except FileNotFoundError:
#             raise FileNotFoundError("Índice do minsearch não encontrado. Execute o indexer.py primeiro.")

#     def busca_semantica(self, query, k=5):
#         # Prefixo 'query:' essencial para a família de modelos E5
#         resultados = self.colecao_chroma.query(
#             query_texts=[f"query: {query}"], 
#             n_results=k
#         )
        
#         if not resultados['ids'] or not resultados['ids'][0]:
#             return []

#         ids_retornados = resultados['ids'][0]
#         return [{"id": id_doc, "rank": i + 1, "origem": "densa"} for i, id_doc in enumerate(ids_retornados)]

#     def busca_lexica(self, query, k=5):
#         resultados_minsearch = self.index_lexico.search(query=query, num_results=k)
#         return [{"id": doc["id"], "rank": i + 1, "origem": "lexica"} for i, doc in enumerate(resultados_minsearch)]

#     def busca_hibrida(self, query, k=5, k_rrf=60):
#         resultados_densos = self.busca_semantica(query, k=k)
#         resultados_lexicos = self.busca_lexica(query, k=k)

#         scores_rrf = {}
        
#         for doc in resultados_densos:
#             id_doc = doc["id"]
#             scores_rrf[id_doc] = scores_rrf.get(id_doc, 0.0) + (1.0 / (k_rrf + doc["rank"]))
            
#         for doc in resultados_lexicos:
#             id_doc = doc["id"]
#             scores_rrf[id_doc] = scores_rrf.get(id_doc, 0.0) + (1.0 / (k_rrf + doc["rank"]))

#         resultados_finais = sorted(scores_rrf.items(), key=lambda item: item[1], reverse=True)
#         return [{"id": id_doc, "score_rrf": score} for id_doc, score in resultados_finais[:k]]