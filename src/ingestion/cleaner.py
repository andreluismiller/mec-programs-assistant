import chromadb
from src.config import CHROMA_PATH

def deletar_colecao(nome_colecao="mec_faq"):
    """
    Remove a coleção do banco vetorial ChromaDB.
    """
    cliente = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        cliente.delete_collection(name=nome_colecao)
        print(f"✅ Coleção '{nome_colecao}' deletada com sucesso.")
    except ValueError:
        print(f"⚠️ A coleção '{nome_colecao}' não existe ou já foi deletada.")

if __name__ == "__main__":
    deletar_colecao()