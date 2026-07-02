"""
MEC FAQ Scraper
===============
Extrai perguntas e respostas da seção "Dúvidas Frequentes" de páginas de
Programas e Ações do Ministério da Educação (gov.br/mec) e retorna um
array JSON com as chaves: "id", "programa", "agrupamento", "pergunta" e "resposta".

A chave "agrupamento" recebe:
  - "geral"        → página sem seções (ex.: PDDE)
  - <nome da seção> → página com perguntas agrupadas (ex.: PNEI)

Uso:
    python mec_faq_scraper.py

    Ou importe a função principal:
        from mec_faq_scraper import scrape_faq_pages
        results = scrape_faq_pages(urls)
"""

import json
import logging
import time
import uuid
from typing import Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_TIMEOUT = 20   # segundos
REQUEST_DELAY   = 1.5  # pausa entre requisições (respeita o servidor)


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def _get_html(url: str, session: requests.Session) -> Optional[str]:
    """Baixa o HTML de uma URL. Retorna None em caso de erro."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException as exc:
        log.error("Erro ao acessar '%s': %s", url, exc)
        return None


def _clean(text: str) -> str:
    """
    Colapsa espaços excessivos e remove &nbsp; (\xa0).
    Funciona tanto para textos simples quanto para saídas do Word/Office Online.
    """
    return " ".join(text.replace("\xa0", " ").split())


def _is_eop_span(tag: Tag) -> bool:
    """Verifica se o elemento é um marcador de fim-de-parágrafo do Word (EOP)."""
    return tag.name == "span" and any("EOP" in c for c in tag.get("class", []))


def _extract_paragraph_text(p_tag: Tag) -> str:
    """
    Extrai texto limpo de um <p>, cobrindo dois casos:

    1. Texto direto no <p> (padrão simples / PNEI):
       <p dir="ltr">Texto direto aqui.</p>

    2. Spans aninhados do Word/Office Online (PDDE):
       <p><span class="TextRun …"><span class="NormalTextRun …">Texto</span></span>
          <span class="EOP …">&nbsp;</span></p>
       → spans EOP (apenas &nbsp;) são descartados.
    """
    parts = []
    for node in p_tag.descendants:
        if not isinstance(node, NavigableString):
            continue
        # Descarta nós filhos de spans EOP (&nbsp; inúteis do Word)
        if any(isinstance(a, Tag) and _is_eop_span(a) for a in node.parents if a != p_tag):
            continue
        text = str(node)
        if _clean(text) == "":   # vazio ou só &nbsp;
            continue
        parts.append(text)
    return _clean("".join(parts))


def _extract_answer_text(conteudo_div: Tag) -> str:
    """
    Converte <div class="conteudo"> em texto plano preservando quebras de
    parágrafo como '\\n'.

    Cobre:
    - Texto direto em <p> (ex.: PNEI)
    - Spans aninhados do Word dentro de <div> > <p> (ex.: PDDE)
    - Listas <ul>/<ol>
    - Sub-títulos <h2>–<h6>
    - Contêineres <div> recursivos
    """
    paragraphs: list[str] = []

    def walk(element: Tag) -> None:
        for child in element.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "p":
                text = _extract_paragraph_text(child)
                if text:
                    paragraphs.append(text)
            elif child.name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    text = _clean(li.get_text(" ", strip=True))
                    if text:
                        paragraphs.append(f"- {text}")
            elif child.name in ("h2", "h3", "h4", "h5", "h6"):
                text = _clean(child.get_text(" ", strip=True))
                if text:
                    paragraphs.append(text)
            elif child.name in ("div", "section", "blockquote", "article"):
                walk(child)   # desce recursivamente

    walk(conteudo_div)
    return "\n".join(paragraphs).strip()


def _get_program_name(soup: BeautifulSoup) -> str:
    """Retorna o nome do programa a partir de <h1 class="documentFirstHeading">."""
    h1 = soup.find("h1", class_="documentFirstHeading")
    if h1:
        return _clean(h1.get_text(" ", strip=True))
    h1 = soup.find("h1")
    return _clean(h1.get_text(" ", strip=True)) if h1 else "Desconhecido"


# ---------------------------------------------------------------------------
# Estratégia primária: toggable-content (gov.br/MEC)
# ---------------------------------------------------------------------------

def _direct_children_tags(element: Tag) -> list[Tag]:
    """Retorna apenas os filhos-Tag diretos de um elemento."""
    return [c for c in element.children if isinstance(c, Tag)]


def _direct_toggle_link(li: Tag) -> Optional[Tag]:
    """
    Retorna o <a class="toggle"> que é filho DIRETO do <li>,
    parando antes de entrar em qualquer <ul> aninhado.
    Isso distingue cabeçalho de seção de link de pergunta em nível inferior.
    """
    for child in li.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "ul":
            break   # não desce para o ul interno
        if child.name == "a" and "toggle" in child.get("class", []):
            return child
    return None


def _extract_qa_from_li(li: Tag, agrupamento: str) -> Optional[dict]:
    """
    Tenta extrair um par Q&A de um <li> de pergunta.
    Retorna dict com agrupamento/pergunta/resposta ou None.
    """
    q_link = li.find("a", class_="toggle")
    conteudo = li.find("div", class_="conteudo")
    if not (q_link and conteudo):
        return None
    pergunta = _clean(q_link.get_text(" ", strip=True))
    resposta = _extract_answer_text(conteudo)
    if pergunta and resposta:
        return {"agrupamento": agrupamento, "pergunta": pergunta, "resposta": resposta}
    return None


def _strategy_toggable_content(soup: BeautifulSoup) -> list[dict]:
    """
    Estratégia principal — detecta automaticamente dois sub-padrões:

    ┌─ SEM AGRUPAMENTO (ex.: PDDE) ─────────────────────────────────────┐
    │  ul.toggable-content                                               │
    │    └─ li  (sem <a class="toggle"> direto)                          │
    │         └─ ul                                                      │
    │              └─ li  ← item Q&A                                     │
    │                   ├─ a.toggle.closed  ← pergunta                  │
    │                   └─ div.conteudo     ← resposta                  │
    └────────────────────────────────────────────────────────────────────┘
    → agrupamento = "geral" para todos os itens

    ┌─ COM AGRUPAMENTO (ex.: PNEI) ──────────────────────────────────────┐
    │  ul.toggable-content                                               │
    │    └─ li  ← seção                                                  │
    │         ├─ a.toggle        ← nome da seção (filho direto do li)   │
    │         └─ ul                                                      │
    │              └─ li  ← item Q&A                                     │
    │                   ├─ a.toggle.closed  ← pergunta                  │
    │                   └─ div.conteudo     ← resposta                  │
    └────────────────────────────────────────────────────────────────────┘
    → agrupamento = nome da seção para cada grupo de perguntas
    """
    toggable = soup.find("ul", class_="toggable-content")
    if not toggable:
        return []

    pairs: list[dict] = []
    top_lis = _direct_children_tags(toggable)

    for top_li in top_lis:
        if top_li.name != "li":
            continue

        direct_children = _direct_children_tags(top_li)
        section_link = _direct_toggle_link(top_li)
        inner_ul = next((c for c in direct_children if c.name == "ul"), None)
        direct_conteudo = next(
            (c for c in direct_children if c.name == "div" and "conteudo" in c.get("class", [])),
            None,
        )

        # ── Caso 1: seção com nome + ul interno (COM agrupamento) ──────────
        if section_link and inner_ul and not direct_conteudo:
            agrupamento = _clean(section_link.get_text(" ", strip=True))
            for qa_li in _direct_children_tags(inner_ul):
                if qa_li.name != "li":
                    continue
                pair = _extract_qa_from_li(qa_li, agrupamento)
                if pair:
                    pairs.append(pair)

        # ── Caso 2: top-li tem apenas ul interno SEM link de seção ─────────
        #    (wrapper "transparente" — padrão PDDE)
        elif inner_ul and not section_link and not direct_conteudo:
            for qa_li in _direct_children_tags(inner_ul):
                if qa_li.name != "li":
                    continue
                pair = _extract_qa_from_li(qa_li, "geral")
                if pair:
                    pairs.append(pair)

        # ── Caso 3: item Q&A diretamente no nível superior ─────────────────
        elif section_link and direct_conteudo:
            pair = _extract_qa_from_li(top_li, "geral")
            if pair:
                pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Estratégias de fallback (mantêm compatibilidade com outros layouts)
# ---------------------------------------------------------------------------

def _strategy_accordion(soup: BeautifulSoup) -> list[dict]:
    """Fallback: accordions Bootstrap / Plone genéricos."""
    pairs = []
    for container in soup.select("div.accordion, div#accordion"):
        for item in container.select(
            "div.accordion-item, div.card, div[class*='accordion-item']"
        ):
            btn = item.select_one(
                "button.accordion-button, .accordion-header button, "
                ".card-header button, a[data-bs-toggle='collapse'], "
                "a[data-toggle='collapse']"
            ) or item.select_one(".accordion-header, .card-header")
            body = item.select_one(
                "div.accordion-collapse, div.accordion-body, "
                "div.collapse, div.card-body"
            )
            if btn and body:
                pergunta = _clean(btn.get_text(" ", strip=True))
                resposta = _extract_answer_text(body)
                if pergunta and resposta:
                    pairs.append({
                        "agrupamento": "geral",
                        "pergunta": pergunta,
                        "resposta": resposta,
                    })
    return pairs


def _strategy_definition_list(soup: BeautifulSoup) -> list[dict]:
    """Fallback: listas de definição <dl> / <dt> / <dd>."""
    pairs = []
    for dl in soup.find_all("dl"):
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
            pergunta = _clean(dt.get_text(" ", strip=True))
            resposta = _extract_answer_text(dd)
            if pergunta and resposta:
                pairs.append({
                    "agrupamento": "geral",
                    "pergunta": pergunta,
                    "resposta": resposta,
                })
    return pairs


def _strategy_strong_paragraph(soup: BeautifulSoup) -> list[dict]:
    """Fallback: <p><strong>Pergunta?</strong></p> seguido de parágrafos de resposta."""
    pairs = []
    content = (
        soup.find("div", id="content-core")
        or soup.find("div", class_="documentContent")
        or soup.find("article") or soup.find("main") or soup.body
    )
    if not content:
        return pairs

    block_tags = {"p", "div", "h2", "h3", "h4", "ul", "ol", "blockquote"}
    children = [
        c for c in content.descendants
        if isinstance(c, Tag) and c.name in block_tags
        and not any(
            isinstance(p, Tag) and p.name in block_tags
            for p in c.parents if p != content
        )
    ]
    i = 0
    while i < len(children):
        child = children[i]
        text = _clean(child.get_text(" ", strip=True))
        is_question = (
            child.find(["strong", "b"]) is not None
            and text.endswith("?") and len(text) < 300
        )
        if is_question:
            resp_parts, j = [], i + 1
            while j < len(children):
                sib = children[j]
                sib_text = _clean(sib.get_text(" ", strip=True))
                if sib.find(["strong", "b"]) and sib_text.endswith("?") and len(sib_text) < 300:
                    break
                if sib_text:
                    resp_parts.append(sib_text)
                j += 1
            if resp_parts:
                pairs.append({
                    "agrupamento": "geral",
                    "pergunta": text,
                    "resposta": "\n".join(resp_parts),
                })
            i = j
        else:
            i += 1
    return pairs


def _strategy_heading_based(soup: BeautifulSoup) -> list[dict]:
    """Fallback: perguntas em <h2>/<h3>/<h4>, respostas nos <p> seguintes."""
    pairs = []
    content = (
        soup.find("div", id="content-core")
        or soup.find("div", class_="documentContent")
        or soup.find("article") or soup.find("main") or soup.body
    )
    if not content:
        return pairs
    for heading in content.find_all(["h2", "h3", "h4"]):
        pergunta = _clean(heading.get_text(" ", strip=True))
        if not pergunta:
            continue
        resp_parts = []
        for sib in heading.find_next_siblings():
            if sib.name in ("h2", "h3", "h4"):
                break
            text = _clean(sib.get_text(" ", strip=True))
            if text:
                resp_parts.append(text)
        if resp_parts:
            pairs.append({
                "agrupamento": "geral",
                "pergunta": pergunta,
                "resposta": "\n".join(resp_parts),
            })
    return pairs


# Ordem de prioridade das estratégias
STRATEGIES = [
    ("toggable_content",  _strategy_toggable_content),   # padrão gov.br/MEC
    ("accordion",         _strategy_accordion),
    ("definition_list",   _strategy_definition_list),
    ("strong_paragraph",  _strategy_strong_paragraph),
    ("heading_based",     _strategy_heading_based),
]


def _extract_faq(soup: BeautifulSoup) -> list[dict]:
    """Tenta cada estratégia em ordem e retorna o primeiro resultado não-vazio."""
    for name, fn in STRATEGIES:
        pairs = fn(soup)
        if pairs:
            log.info("  Estratégia '%s' → %d par(es) encontrado(s).", name, len(pairs))
            return pairs
        log.debug("  Estratégia '%s' → sem resultados.", name)
    log.warning("  Nenhuma estratégia encontrou perguntas/respostas nesta página.")
    return []


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def scrape_faq_pages(urls: list[str]) -> list[dict]:
    """
    Recebe uma lista de URLs de FAQ do MEC e retorna um array de dicts com
    as chaves 'id', 'programa', 'agrupamento', 'pergunta' e 'resposta'.
    """
    results: list[dict] = []

    with requests.Session() as session:
        for idx, url in enumerate(urls):
            log.info("(%d/%d) Processando: %s", idx + 1, len(urls), url)

            html = _get_html(url, session)
            if html is None:
                log.warning("  Pulando URL por falha de requisição.")
                continue

            soup = BeautifulSoup(html, "lxml")
            programa = _get_program_name(soup)
            log.info("  Programa: %s", programa)

            page_pairs = _extract_faq(soup)
            for pair in page_pairs:
                results.append({
                    "id":          uuid.uuid4().hex,
                    "programa":    programa,
                    "agrupamento": pair["agrupamento"],
                    "pergunta":    pair["pergunta"],
                    "resposta":    pair["resposta"],
                })

            log.info("  %d registro(s) adicionado(s).", len(page_pairs))

            if idx < len(urls) - 1:
                time.sleep(REQUEST_DELAY)

    log.info("Total geral de registros extraídos: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Entry-point CLI
# ---------------------------------------------------------------------------

def main():
    urls = [
        # Sem agrupamento → agrupamento = "geral"
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/pneei",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/politica-nacional-integrada-da-primeira-infancia-pnipi",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/educacao-escolar-indigena",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-nacional-do-livro-e-do-material-didatico",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-dinheiro-direto-na-escola-pdde",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-educacao-para-a-cidadania-e-para-a-sustentabilidade",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/pacto-nacional-pela-superacao-do-analfabetismo",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-leitura-e-escrita-na-educacao-infantil-proleei",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-de-acompanhamento-e-formacao-continuada-para-o-ensino-multisseriado-praema",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-escola-e-comunidade",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-brasil-alfabetizado",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/ensino-medio-mais",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-toda-matematica",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/prilei",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/proditec",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-nacional-de-inclusao-de-jovens",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/politica-nacional-de-educacao-profissional-e-tecnologica",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/profissionais-do-futuro",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-nacional-de-acesso-ao-ensino-tecnico-e-emprego",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/juros-por-educacao",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/inovalab",
        # Com agrupamento → agrupamento = nome da seção
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/politica-nacional-de-educacao-infantil-pnei",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/politica-nacional-de-equidade-educacao-para-as-relacoes-etnico-raciais-e-educacao-escolar-quilombola-pneerq",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/pacto-nacional-pela-recomposicao-das-aprendizagens",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/pe-de-meia",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/programa-universidade-para-todos-prouni",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/sisu",
        "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/politica-de-regulacao-e-supervisao-da-educacao-superior"
    ]

    data = scrape_faq_pages(urls)

    output_file = "mec_faq_raw.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info("Resultado salvo em '%s'.", output_file)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()