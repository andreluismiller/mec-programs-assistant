import json
from minsearch import Index

def load_faq_data():
    """Carrega os dados do FAQ do MEC a partir do arquivo JSON."""
    with open("mec_faq.json", encoding="utf-8") as f:
        documents = json.load(f)
    return documents


def build_index(documents):
    """Cria um índice de busca a partir dos documentos do FAQ."""
    index = Index(
        text_fields=["pergunta", "resposta", "nome", "termos", "sinonimos", "agrupamento"],
        keyword_fields=["id", "programa"]
    )
    index.fit(documents)
    return index