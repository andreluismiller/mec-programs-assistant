import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# 1. Criação da Função de Embedding Customizada
class MiniLMEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # O modelo é carregado apenas uma vez na memória quando a classe é instanciada
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Transforma o texto (documentos ou perguntas) em vetores
        return self.model.encode(input).tolist()

# 2. Função de Ingestão em Lotes
def ingerir_dados_faq(caminho_json: str, caminho_db: str, nome_colecao: str, tamanho_batch: int = 50):
    """
    Lê o JSON, prepara os documentos e os metadados, e insere no ChromaDB em lotes.
    """
    # Inicializa o cliente ChromaDB persistente
    client = chromadb.PersistentClient(path=caminho_db)
    
    # Inicializa a função de embedding customizada
    funcao_embedding = MiniLMEmbeddingFunction()
    
    # Cria a coleção ou recupera se já existir, vinculando o modelo a ela
    collection = client.get_or_create_collection(
        name=nome_colecao,
        embedding_function=funcao_embedding
    )
    
    # Carrega os dados do arquivo JSON
    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        
    total_documentos = len(dados)
    print(f"Iniciando a ingestão de {total_documentos} documentos...")
    
    # Processamento em Batches (Lotes)
    for i in range(0, total_documentos, tamanho_batch):
        lote_atual = dados[i:i+tamanho_batch]
        
        ids = []
        documentos = []
        metadados = []
        
        for item in lote_atual:
            # Concatena a pergunta e a resposta para o documento principal
            texto_documento = f"Pergunta: {item['pergunta']} \nResposta: {item['resposta']}"
            
            # Prepara os metadados (removendo id, pergunta e resposta para não duplicar)
            meta = {
                "programa": item.get("programa", ""),
                "agrupamento": item.get("agrupamento", ""),
                "nome": item.get("nome", ""),
                "termos": item.get("termos", ""),
                "sinonimos": item.get("sinonimos", "")
            }
            
            ids.append(item["id"])
            documentos.append(texto_documento)
            metadados.append(meta)
            
        # Adiciona o lote à coleção. 
        # O ChromaDB usará a MiniLMEmbeddingFunction automaticamente nos 'documentos'.
        collection.add(
            ids=ids,
            documents=documentos,
            metadatas=metadados
        )
        
        print(f"Lote processado: {i + len(lote_atual)}/{total_documentos} itens inseridos.")
        
    print("Ingestão concluída com sucesso!")

# --- Exemplo de Uso ---
if __name__ == "__main__":
    ARQUIVO_JSON = "mec_faq.json"
    DIRETORIO_DB = "./chroma_db" # Pasta local onde os dados serão persistidos
    COLECAO = "faq_mec_collection"
    
    # Executa a ingestão processando 50 itens por vez
    ingerir_dados_faq(
        caminho_json=ARQUIVO_JSON, 
        caminho_db=DIRETORIO_DB, 
        nome_colecao=COLECAO, 
        tamanho_batch=50
    )