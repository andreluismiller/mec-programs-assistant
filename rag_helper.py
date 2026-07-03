INSTRUCTIONS = '''
Você é um assistente especializado nos programas e ações do Ministério da Educação (MEC).
Sua tarefa é responder perguntas com base no contexto fornecido, extraído das páginas de
Dúvidas Frequentes dos programas do MEC.
Utilize o contexto para encontrar informações relevantes e forneça respostas precisas.
Responda sempre em português. Se a resposta não estiver presente no contexto,
responda com "Não encontrei essa informação nas perguntas frequentes disponíveis."
'''.strip()

PROMPT_TEMPLATE = '''
PERGUNTA: {question}
CONTEXTO:
{context}
'''.strip()

BOOST_DICT = {
    'pergunta':    3.0,
    'nome':        2.0,
    'termos':      1.5,
    'sinonimos':   1.0,
    'resposta':    1.0,
    'agrupamento': 0.5,
}


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        programa=None,          # filtragem opcional por programa específico
        model='llama-3.3-70b-versatile'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.programa = programa
        self.model = model

    def search(self, query, num_results=5):
        # Aplica filtro por programa apenas quando especificado
        filter_dict = {'programa': self.programa} if self.programa else {}
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=BOOST_DICT,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"Programa:     {doc['programa']}")
            if doc.get('agrupamento') and doc['agrupamento'] != 'geral':
                lines.append(f"Agrupamento:  {doc['agrupamento']}")
            lines.append(f"P: {doc['pergunta']}")
            lines.append(f"R: {doc['resposta']}")
            lines.append('')
        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        """
        Chama o modelo via API compatível com OpenAI (Groq).

        Returns:
            dict com as chaves:
                - 'texto':          resposta gerada pelo modelo
                - 'tokens_input':   tokens consumidos no prompt
                - 'tokens_output':  tokens gerados na resposta
                - 'tokens_total':   soma de input + output
        """
        messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user',   'content': prompt},
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        usage = response.usage
        return {
            'texto':         response.choices[0].message.content,
            'tokens_input':  usage.prompt_tokens,
            'tokens_output': usage.completion_tokens,
            'tokens_total':  usage.total_tokens,
        }

    def rag(self, query):
        """
        Executa o pipeline completo de RAG.

        Returns:
            dict com as chaves:
                - 'resposta':       texto final gerado pelo modelo
                - 'tokens_input':   tokens consumidos no prompt
                - 'tokens_output':  tokens gerados na resposta
                - 'tokens_total':   soma de input + output
        """
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        llm_result = self.llm(prompt)
        return {
            'resposta':      llm_result['texto'],
            'tokens_input':  llm_result['tokens_input'],
            'tokens_output': llm_result['tokens_output'],
            'tokens_total':  llm_result['tokens_total'],
        }