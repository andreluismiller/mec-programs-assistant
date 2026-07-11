from src.rag.retriever import BuscadorMEC

buscador = BuscadorMEC()
resultado = buscador.busca_hibrida("Como funciona o PNEEI?")
print(resultado)