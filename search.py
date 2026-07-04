import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# 1. Recriação da Função de Embedding Customizada
# Deve ser idêntica à utilizada na ingestão para manter a compatibilidade dos vetores
class MiniLMEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # Carrega o modelo na memória para vetorizar as perguntas de busca
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

# 2. Função de Busca Semântica
def realizar_busca_faq(termo_busca: str, caminho_db: str, nome_colecao: str, n_resultados: int = 3):
    """
    Conecta ao banco persistido, realiza a busca semântica e exibe os resultados formatados.
    """
    # Conecta ao cliente ChromaDB persistido localmente
    client = chromadb.PersistentClient(path=caminho_db)
    
    # Instancia a função de embedding para a busca
    funcao_embedding = MiniLMEmbeddingFunction()
    
    try:
        # Recupera a coleção existente vinculando a função de embedding
        collection = client.get_collection(name=nome_colecao, embedding_function=funcao_embedding)
    except Exception as e:
        print(f"Erro ao acessar a coleção '{nome_colecao}': {e}")
        print("Certifique-se de rodar o script de ingestão primeiro.")
        return

    print(f"🔍 Realizando busca semântica para: '{termo_busca}'\n")
    
    # O ChromaDB converte automaticamente o texto de busca em vetor usando nossa classe
    resultados = collection.query(
        query_texts=[termo_busca],
        n_results=n_resultados
    )
    
    # Se nenhum resultado for retornado
    if not resultados or not resultados['ids'] or len(resultados['ids'][0]) == 0:
        print("Nenhum resultado relevante encontrado.")
        return

    # Varre os resultados retornados (Chroma retorna uma lista de listas)
    for i in range(len(resultados['ids'][0])):
        doc_id = resultados['ids'][0][i]
        documento = resultados['documents'][0][i]
        metadados = resultados['metadatas'][0][i]
        distancia = resultados['distances'][0][i] # Menor distância = maior similaridade semântica
        
        print(f"🏆 [Resultado #{i+1}] - Distância Vetorial: {distancia:.4f}")
        print(f"🆔 ID do Documento: {doc_id}")
        print(f"📋 Programa: {metadados.get('nome')} ({metadados.get('programa')})")
        print(f"📖 Conteúdo Recuperado:\n{documento}")
        print(f"🏷️  Termos Relacionados: {metadados.get('termos')}")
        print(f"🔗 Sinônimos Mapeados: {metadados.get('sinonimos')}")
        print("-" * 50 + "\n")

# --- Execução de Exemplo ---
if __name__ == "__main__":
    DIRETORIO_DB = "./chroma_db"  # Deve apontar para a mesma pasta gerada na ingestão
    COLECAO = "faq_mec_collection"
    
    # Pergunta de exemplo (não precisa usar as mesmas palavras do arquivo, a busca é semântica)
    PERGUNTA_USUARIO = "Quem faz parte do público-alvo atendido pela educação especial?"
    
    realizar_busca_faq(
        termo_busca=PERGUNTA_USUARIO,
        caminho_db=DIRETORIO_DB,
        nome_colecao=COLECAO,
        n_resultados=2  # Quantidade de respostas mais próximas que deseja retornar
    )