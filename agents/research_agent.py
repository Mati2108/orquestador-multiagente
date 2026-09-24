"""Agente Investigador: consulta la base vectorial (ChromaDB) del corpus Orbital.

Herramientas acotadas y de solo lectura: `search_docs` y `list_sources`.
No calcula ni redacta la respuesta final: transcribe datos con su cita.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool

from agents.common import new_contribution, previous_attempt_block, run_react_agent
from config import Settings, settings as default_settings
from state import OrchestratorState, get_question

RESEARCH_PROMPT = """Sos el Investigador de un equipo multi-agente. Tu única fuente es la base de \
conocimiento de Orbital (plataforma interna de despliegue de microservicios), a la que accedés con tus \
herramientas.

Cómo trabajás:
1. Buscá con `search_docs` usando consultas cortas y específicas. Si la instrucción pide varios datos, \
hacé una búsqueda por dato.
2. Reportá SOLO datos que aparecen en los fragmentos recuperados, cada uno con su cita [archivo#n] \
tal como figura en el encabezado del fragmento.
3. No hagas cálculos ni conversiones: eso lo hace el Analista. Transcribí los valores tal cual \
(por ejemplo "500m", "512Mi", "10").
4. Si un dato no está en los documentos, decilo en "No encontrado". Nunca completes con conocimiento \
general.

Formato de salida (sin preámbulo):
Hallazgos:
- <dato concreto> [archivo#n]
No encontrado:
- <dato pedido que no aparece> (o "nada")
"""


def make_research_tools(store: Chroma, top_k: int) -> list[BaseTool]:
    """Las herramientas se construyen sobre un vectorstore concreto (inyectable en tests)."""

    @tool(response_format="content_and_artifact")
    def search_docs(query: str) -> tuple[str, list[dict]]:
        """Búsqueda semántica en la documentación de Orbital. Devuelve los fragmentos más \
parecidos a la consulta, cada uno con su id citable [archivo#n]."""
        hits = store.similarity_search_with_relevance_scores(query, k=top_k)
        if not hits:
            return "Sin resultados.", []
        blocks, artifact = [], []
        for doc, score in hits:
            ref = doc.metadata["id"]
            blocks.append(f"[{ref}] sección: {doc.metadata.get('section', '')} · similitud {score:.2f}\n{doc.page_content}")
            artifact.append({"id": ref, "score": round(score, 3), "content": doc.page_content})
        return "\n\n---\n\n".join(blocks), artifact

    @tool
    def list_sources() -> str:
        """Lista los documentos disponibles en la base de conocimiento y cuántos fragmentos tiene cada uno."""
        metadatas = store.get(include=["metadatas"])["metadatas"]
        counts = Counter(m["source"] for m in metadatas)
        return "\n".join(f"- {source}: {n} fragmentos" for source, n in sorted(counts.items()))

    return [search_docs, list_sources]


def build_research_agent(llm: BaseChatModel, store: Chroma, cfg: Settings = default_settings) -> Runnable:
    # create_agent es el sucesor de create_react_agent en LangGraph v1: el mismo loop ReAct.
    return create_agent(llm, make_research_tools(store, cfg.top_k), system_prompt=RESEARCH_PROMPT, name="researcher")


def make_research_node(agent: Runnable, cfg: Settings = default_settings) -> Callable[..., dict]:
    def researcher(state: OrchestratorState, config: RunnableConfig) -> dict:
        # Contexto mínimo: la consulta original + la instrucción puntual del supervisor.
        task = (
            f"Consulta original del usuario:\n{get_question(state)}\n\n"
            f"Instrucción del supervisor:\n{state.get('current_instruction', '')}"
            f"{previous_attempt_block(state, 'researcher')}"
        )
        run = run_react_agent(agent, task, config, cfg.agent_recursion_limit)
        retrieved = [hit["id"] for hits in run.artifacts if isinstance(hits, list) for hit in hits]
        contribution = new_contribution(
            state,
            "researcher",
            run.content,
            sources=list(dict.fromkeys(retrieved)),  # únicos, en orden de aparición
            tool_calls=run.tool_calls,
            evidence=run.evidence,
            status=run.status,
        )
        return {"contributions": [contribution], "messages": [AIMessage(content=run.content, name="researcher")]}

    return researcher
