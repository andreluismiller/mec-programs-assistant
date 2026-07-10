import json
from pathlib import Path

def unificar_arquivos_json():
    # 1. Mapeamento dinâmico de caminhos
    # Path(__file__).resolve().parent.parent aponta exatamente para a pasta 'src'
    # Adicionando .parent novamente chegamos na raiz do projeto
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Define o diretório onde estão os arquivos (ajuste o nome se necessário)
    dir_dados = base_dir / "data" / "ground_truth"
    caminho_saida = dir_dados / "perguntas.json"
    
    lista_unificada = []
    
    print(f"Buscando arquivos em: {dir_dados}\n")
    
    # 2. Loop para ler os arquivos de ground1.json até ground8.json
    for i in range(1, 9):
        nome_arquivo = f"ground{i}.json"
        caminho_arquivo = dir_dados / nome_arquivo
        
        # Verifica se o arquivo realmente existe antes de tentar abrir
        if caminho_arquivo.exists():
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                
                # Usamos .extend() porque 'dados' já é uma lista de dicionários.
                # Se usássemos .append(), criaríamos uma lista de listas.
                lista_unificada.extend(dados)
                
            print(f"✅ Lidos {len(dados)} registros de {nome_arquivo}")
        else:
            print(f"⚠️ Aviso: Arquivo {nome_arquivo} não encontrado.")

    # 3. Validação e Escrita do arquivo final
    if not lista_unificada:
        print("\nNenhum dado foi encontrado para unificar. Verifique a pasta.")
        return

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        # ensure_ascii=False garante que acentos (á, ç, ã) sejam salvos corretamente
        # indent=4 deixa o arquivo formatado e legível (pretty-print)
        json.dump(lista_unificada, f, ensure_ascii=False, indent=4)
        
    print(f"\n🚀 Sucesso! Arquivo gerado: {caminho_saida.name}")
    print(f"Total de dicionários unificados: {len(lista_unificada)}")

if __name__ == "__main__":
    unificar_arquivos_json()