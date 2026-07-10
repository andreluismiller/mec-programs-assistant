import os
from pathlib import Path

# Definindo a raiz do projeto (supondo que o script rode a partir da raiz)
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos de Dados
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.json"
EVAL_DATA_PATH = BASE_DIR / "data" / "ground_truth" / "perguntas.json"

# Caminhos de Processamento (Persistência)
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_PATH = PROCESSED_DIR / "chroma_data"
MINSEARCH_PATH = PROCESSED_DIR / "minsearch_index.pkl"

# Criar diretórios se não existirem
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)