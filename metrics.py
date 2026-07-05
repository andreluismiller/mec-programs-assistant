import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase

# ---------------------------------------------------------------------------
# Preços Groq (USD por milhão de tokens) — atualize se necessário
# Fonte: https://groq.com/pricing
# ---------------------------------------------------------------------------
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {
        "input":  0.59,   # USD / 1M tokens de entrada
        "output": 0.79,   # USD / 1M tokens de saída
    },
}


@dataclass
class LLMCallRecord:
    model:             str
    prompt:            str
    instructions:      str
    answer:            str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    response_time:     float
    cost:              float
    timestamp:         datetime = field(default_factory=datetime.now)


def calculate_cost(model: str, usage) -> float:
    """
    Calcula o custo da chamada em USD com base no modelo e no uso de tokens.

    Args:
        model: Nome do modelo utilizado.
        usage: Objeto usage retornado pela API (atributos prompt_tokens e
               completion_tokens).

    Returns:
        Custo estimado em USD, ou 0.0 se o modelo não estiver na tabela.
    """
    pricing = GROQ_PRICING.get(model)
    if not pricing:
        return 0.0

    return (
        usage.prompt_tokens     * pricing["input"]
        + usage.completion_tokens * pricing["output"]
    ) / 1_000_000


class RAGWithMetrics(RAGBase):
    """
    Subclasse de RAGBase que intercepta a chamada ao LLM para registrar
    métricas de uso: tokens, tempo de resposta e custo estimado.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord | None = None

    def llm(self, prompt: str) -> str:
        """
        Sobrescreve RAGBase.llm() para capturar métricas sem alterar
        o valor de retorno (texto da resposta).
        """
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time

        answer = response.choices[0].message.content
        self._log_response(prompt, response, answer, response_time)

        return answer

    def _call_llm(self, prompt: str):
        """Executa a chamada à API Groq via compatibilidade OpenAI."""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user",   "content": prompt},
        ]
        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

    def _log_response(self, prompt: str, response, answer: str, response_time: float) -> None:
        """Constrói e armazena o registro de métricas da última chamada."""
        usage = response.usage
        cost  = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=             self.model,
            prompt=            prompt,
            instructions=      self.instructions,
            answer=            answer,
            prompt_tokens=     usage.prompt_tokens,
            completion_tokens= usage.completion_tokens,
            total_tokens=      usage.total_tokens,
            response_time=     response_time,
            cost=              cost,
        )

        print(call_record)
        self.last_call = call_record