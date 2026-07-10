from chromadb.utils.embedding_functions import FastEmbedEmbeddingFunction

def get_embedding_function():
    """
    Retorna a função de embedding configurada.
    Utiliza o modelo multilíngue da família E5.
    """
    return FastEmbedEmbeddingFunction(model_name="intfloat/multilingual-e5-base")