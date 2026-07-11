from fastembed import TextEmbedding

class Embedder:
    def __init__(self):
        # Inicializa o modelo diretamente via FastEmbed
        self.model = TextEmbedding(model_name="intfloat/multilingual-e5-base")

    def gerar_vetores(self, textos):
        """Recebe uma lista de textos e retorna uma lista de vetores."""
        # O fastembed retorna um gerador, convertemos para lista
        return list(self.model.embed(textos))



# import chromadb.utils.embedding_functions as embedding_functions

# def get_embedding_function():
#     """
#     Tenta instanciar o embedding function de forma robusta.
#     Se o caminho do Chroma mudou, tentamos a alternativa via biblioteca.
#     """
#     try:
#         # Tenta a importação padrão (que deve funcionar se o ambiente estiver limpo)
#         return embedding_functions.FastEmbedEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
#     except AttributeError:
#         # Se a classe não estiver disponível em utils, tentamos forçar o uso do modelo
#         # via a classe genérica do Chroma se disponível, ou raise informativo.
#         raise ImportError(
#             "Não foi possível encontrar 'FastEmbedEmbeddingFunction' no ChromaDB 1.5.9. "
#             "Verifique se o pacote 'chromadb-embeddings' está instalado ou use "
#             "uma classe de embedding manual."
#         )


# from chromadb.utils.embedding_functions import FastEmbedEmbeddingFunction

# def get_embedding_function():
#     """
#     Retorna a função de embedding configurada.
#     Utiliza o modelo multilíngue da família E5.
#     """
#     return FastEmbedEmbeddingFunction(model_name="intfloat/multilingual-e5-base")