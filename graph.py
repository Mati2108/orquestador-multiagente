"""Grafo principal: topología jerárquica con un Supervisor en el centro.

    START -> supervisor -(arista condicional)-> researcher | analyst | synthesizer | END
    researcher -> supervisor
    analyst    -> supervisor
    synthesizer -> validator -> supervisor

Los especialistas nunca se hablan entre sí: toda la coordinación pasa por el
supervisor y por el estado compartido.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import build_analyst_agent, make_analyst_node
from agents.research_agent import build_research_agent, make_research_node
from agents.supervisor import Decider, build_supervisor_decider, make_supervisor_node, route_from_supervisor
from agents.synthesizer import make_synthesizer_node
from agents.validator import validator
from config import ROOT_DIR, Settings, build_chat_model, settings as default_settings
from state import OrchestratorState


def build_graph(
    *,
    decide: Decider | None = None,
    research_agent: Runnable | None = None,
    analyst_agent: Runnable | None = None,
    synthesizer_llm: Runnable | None = None,
    cfg: Settings = default_settings,
) -> CompiledStateGraph:
    """Arma y compila el grafo. Cada pieza es inyectable (los tests usan dobles sin API)."""
    if None in (decide, research_agent, analyst_agent, synthesizer_llm):
        worker_llm = build_chat_model(cfg)  # especialistas
        coordinator_llm = build_chat_model(cfg, cfg.supervisor_model)  # supervisor y sintetizador
        if research_agent is None:
            from knowledge_base import get_store

            research_agent = build_research_agent(worker_llm, get_store(cfg), cfg)
        decide = decide or build_supervisor_decider(coordinator_llm)
        analyst_agent = analyst_agent or build_analyst_agent(worker_llm)
        synthesizer_llm = synthesizer_llm or coordinator_llm

    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", make_supervisor_node(decide, cfg))
    builder.add_node("researcher", make_research_node(research_agent, cfg))
    builder.add_node("analyst", make_analyst_node(analyst_agent, cfg))
    builder.add_node("synthesizer", make_synthesizer_node(synthesizer_llm))
    builder.add_node("validator", validator)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor)  # destinos tomados del Literal
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("synthesizer", "validator")
    builder.add_edge("validator", "supervisor")
    return builder.compile(name="orquestador")


def build_diagram_graph() -> CompiledStateGraph:
    """El mismo grafo con piezas vacías: alcanza para dibujarlo sin clave de API."""
    stub = RunnableLambda(lambda _: {"messages": []})
    return build_graph(decide=lambda *_: None, research_agent=stub, analyst_agent=stub, synthesizer_llm=stub)


def export_diagram(graph: CompiledStateGraph, out_dir: Path = ROOT_DIR / "docs") -> tuple[Path, Path | None]:
    """Guarda el diagrama Mermaid (texto) y, si hay red, el PNG renderizado por mermaid.ink."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mermaid_path = out_dir / "graph.mmd"
    mermaid_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    try:
        png_path = out_dir / "graph.png"
        png_path.write_bytes(graph.get_graph().draw_mermaid_png())
    except Exception:  # sin conexión a mermaid.ink: queda el .mmd, que GitHub renderiza igual
        png_path = None
    return mermaid_path, png_path
