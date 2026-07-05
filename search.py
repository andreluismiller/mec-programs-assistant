import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

# 1. A mesma classe customizada precisa estar aqui
class FastEmbedCustomFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        vetores = self.model.embed(input)
        return [vetor.tolist() for vetor in vetores]

def realizar_busca_faq(termo_busca: str, caminho_db: str, nome_colecao: str, n_resultados: int = 3):
    client = chromadb.PersistentClient(path=caminho_db)
    
    # 2. Inicializamos a nossa função
    funcao_embedding = FastEmbedCustomFunction()
    
    try:
        collection = client.get_collection(name=nome_colecao, embedding_function=funcao_embedding)
    except Exception as e:
        print(f"Erro ao acessar a coleção: {e}")
        return

    print(f"🔍 Realizando busca semântica para: '{termo_busca}'\n")
    
    resultados = collection.query(
        query_texts=[termo_busca],
        n_results=n_resultados
    )
    
    if not resultados or not resultados['ids'] or len(resultados['ids'][0]) == 0:
        print("Nenhum resultado relevante encontrado.")
        return

    # ... (O RESTANTE DA FORMATAÇÃO DOS PRINTS FICA IGUAL) ...
    for i in range(len(resultados['ids'][0])):
        doc_id = resultados['ids'][0][i]
        documento = resultados['documents'][0][i]
        metadados = resultados['metadatas'][0][i]
        distancia = resultados['distances'][0][i]
        
        print(f"🏆 [Resultado #{i+1}] - Distância Vetorial: {distancia:.4f}")
        print(f"🆔 ID: {doc_id}")
        print(f"📋 Programa: {metadados.get('nome')} ({metadados.get('programa')})")
        print(f"📖 Conteúdo:\n{documento}")
        print(f"🏷️  Termos: {metadados.get('termos')}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    DIRETORIO_DB = "./chroma_db"
    COLECAO = "faq_mec_collection"
    PERGUNTA_USUARIO = "Quais ações do MEC atendem estudantes indígenas ou comunidades quilombolas?"
    
    realizar_busca_faq(termo_busca=PERGUNTA_USUARIO, caminho_db=DIRETORIO_DB, nome_colecao=COLECAO, n_resultados=2)