import json
import pickle
import chromadb
import minsearch
from src.config import RAW_DATA_PATH, CHROMA_PATH, MINSEARCH_PATH
from src.embeddings.embedder import get_embedding_function

def executar_ingestao():
    """Lê os dados da pasta raw e cria os índices em disco."""
    print("Iniciando processo de ingestão de dados...")
    
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        documentos = json.load(f)

    ids = []
    textos_ricos = []
    metadados = []
    dados_lexicos = []

    for doc in documentos:
        # Concatenando os campos relevantes
        texto = f"{doc.get('pergunta', '')} {doc.get('resposta', '')} {doc.get('termos', '')} {doc.get('sinonimos', '')} {doc.get('sigla', '')}"
        
        # Preparação Chroma
        ids.append(doc["id"])
        textos_ricos.append(texto)
        metadados.append({"programa": doc.get("programa", ""), "pergunta": doc.get("pergunta", "")})
        
        # Preparação Minsearch
        dados_lexicos.append({"id": doc["id"], "texto_rico": texto})

    # 1. Ingestão Semântica (ChromaDB)
    cliente_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    colecao = cliente_chroma.get_or_create_collection(
        name="mec_faq",
        embedding_function=get_embedding_function()
    )
    colecao.upsert(documents=textos_ricos, metadatas=metadados, ids=ids)
    print(f"✅ {len(ids)} documentos indexados no ChromaDB.")

    # 2. Ingestão Lexical (minsearch via Pickle)
    index_lexico = minsearch.Index(
        text_fields=["texto_rico"],
        keyword_fields=["id"]
    )
    index_lexico.fit(dados_lexicos)
    
    with open(MINSEARCH_PATH, 'wb') as f:
        pickle.dump(index_lexico, f)
    print(f"✅ Índice léxico salvo em {MINSEARCH_PATH}.")

if __name__ == "__main__":
    executar_ingestao()