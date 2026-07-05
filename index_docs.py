import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

# 1. Nossa classe customizada e à prova de falhas para o FastEmbed
class FastEmbedCustomFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Inicializa o modelo ONNX diretamente pela biblioteca oficial
        self.model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Gera os vetores e converte para listas de floats (padrão exigido pelo Chroma)
        vetores = self.model.embed(input)
        return [vetor.tolist() for vetor in vetores]

def ingerir_dados_faq(caminho_json: str, caminho_db: str, nome_colecao: str, tamanho_batch: int = 100):
    client = chromadb.PersistentClient(path=caminho_db)
    
    # 2. Utilizamos a nossa função ao invés daquela nativa do Chroma que causou erro
    funcao_embedding = FastEmbedCustomFunction()
    
    collection = client.get_or_create_collection(
        name=nome_colecao,
        embedding_function=funcao_embedding
    )
    
    # ... (O RESTANTE DO CÓDIGO CONTINUA EXATAMENTE IGUAL) ...
    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        
    total_documentos = len(dados)
    print(f"Iniciando a ingestão de {total_documentos} documentos...")
    
    for i in range(0, total_documentos, tamanho_batch):
        lote_atual = dados[i:i+tamanho_batch]
        ids = []
        documentos = []
        metadados = []
        
        for item in lote_atual:
            texto_documento = f"Pergunta: {item['pergunta']} \nResposta: {item['resposta']}"
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
            
        collection.add(ids=ids, documents=documentos, metadatas=metadados)
        print(f"Lote processado: {i + len(lote_atual)}/{total_documentos} itens inseridos.")
        
    print("Ingestão concluída com sucesso!")

if __name__ == "__main__":
    ARQUIVO_JSON = "mec_faq.json"
    DIRETORIO_DB = "./chroma_db"
    COLECAO = "faq_mec_collection"
    
    ingerir_dados_faq(caminho_json=ARQUIVO_JSON, caminho_db=DIRETORIO_DB, nome_colecao=COLECAO)