"""Punto de entrada: corre el orquestador e imprime la traza de delegación.

Uso:
    python main.py "tu consulta"
    python main.py --demo        # consulta de demostración (Investigador -> Analista -> síntesis)
    python main.py --diagram     # regenera docs/graph.mmd y docs/graph.png
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph.state import CompiledStateGraph

from config import Settings, settings as default_settings
from state import OrchestratorState, initial_state

DEMO_QUESTION = (
    "Tomando el manifiesto de ejemplo de pagos-api de la documentación de Orbital: ¿cuánta CPU y memoria "
    "reserva en total cuando escala al máximo de réplicas? ¿Cuánto necesitaría durante una transición "
    "blue-green en ese máximo? ¿Cada pod respeta el límite por pod de la plataforma?"
)
OUT_OF_SCOPE_QUESTION = "¿Cuánto cuesta por mes la licencia de Orbital para un equipo de 80 desarrolladores?"

ICONS = {"supervisor": "🧭", "researcher": "🔎", "analyst": "🧮", "synthesizer": "✍️ ", "validator": "🛡️ "}


class UsageTracker(BaseCallbackHandler):
    """Cuenta llamadas al LLM y tokens reales (lo que reporta la API, no estimaciones)."""

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = self.reasoning_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.reasoning_tokens += usage.get("output_token_details", {}).get("reasoning", 0)

    def summary(self) -> str:
        return (
            f"{self.calls} llamadas al LLM · {self.input_tokens:,} tokens de entrada · "
            f"{self.output_tokens:,} de salida ({self.reasoning_tokens:,} de razonamiento)"
        ).replace(",", ".")


@dataclass
class RunResult:
    state: OrchestratorState
    usage: UsageTracker
    seconds: float


# --------------------------------------------------------------------------- traza legible
def _indent(text: str, max_lines: int = 40) -> str:
    lines = text.strip().splitlines() or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… ({len(lines) - max_lines} líneas más)"]
    return "\n".join("   │ " + line for line in lines)


def format_update(node: str, update: dict) -> str:
    icon = ICONS.get(node, "•")
    if node == "supervisor":
        d = update["decisions"][0]
        out = [f"{icon} supervisor · decisión {d['step']} → {d['route']}", f"   evaluación: {d['evaluation']}", f"   motivo: {d['reason']}"]
        if d["guard"]:
            out.append(f"   ⛔ regla dura: {d['guard']}")
        if d["instruction"]:
            out.append("   instrucción: " + textwrap.shorten(d["instruction"], 400, placeholder=" […]"))
        return "\n".join(out)
    if node == "validator":
        v = update["validation"]
        if v["passed"]:
            head = f"{icon} validator · APROBADA: {v['checked_numbers']} números con respaldo · citas válidas: {len(v['cited_sources'])}"
        else:
            head = (
                f"{icon} validator · RECHAZADA: números sin respaldo {v['unsupported_numbers']} · "
                f"citas inexistentes {v['invalid_citations']}"
            )
        return head + "".join(f"\n   aviso: {w}" for w in v["warnings"])
    c = update["contributions"][0]
    tools = Counter(call["tool"] for call in c["tool_calls"])
    meta = [f"intento {c['attempt']}"]
    if tools:
        meta.append("herramientas: " + ", ".join(f"{name}×{n}" for name, n in tools.items()))
    if node == "researcher" and c["sources"]:
        meta.append("fragmentos: " + ", ".join(c["sources"]))
    if c["status"] == "error":
        meta.append("ESTADO: error")
    return f"{icon} {node} · " + " · ".join(meta) + "\n" + _indent(c["content"])


def run_with_trace(
    graph: CompiledStateGraph,
    question: str,
    cfg: Settings = default_settings,
    printer: Callable[[str], None] | None = print,
) -> RunResult:
    """Ejecuta una consulta mostrando cada paso del grafo a medida que ocurre."""
    tracker = UsageTracker()
    config = {"recursion_limit": cfg.graph_recursion_limit, "callbacks": [tracker]}
    final: OrchestratorState = initial_state(question)
    started = time.perf_counter()
    for mode, chunk in graph.stream(initial_state(question), config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif printer:
            for node, update in chunk.items():
                printer(format_update(node, update) + "\n")
    return RunResult(state=final, usage=tracker, seconds=time.perf_counter() - started)


def contributions_table(state: OrchestratorState) -> str:
    """Tabla Markdown de quién aportó qué (para el notebook y el README)."""
    rows = ["| # | agente | intento | herramientas | fragmentos recuperados | inicio del aporte |", "|---|---|---|---|---|---|"]
    for c in state["contributions"]:
        tools = ", ".join(f"{k}×{v}" for k, v in Counter(call["tool"] for call in c["tool_calls"]).items()) or "—"
        lines = [line.strip("-• ").strip() for line in c["content"].splitlines()]
        first = next((line for line in lines if line and not line.endswith(":")), "")  # salta "Hallazgos:" y similares
        first = textwrap.shorten(first, 80, placeholder="…").replace("|", "/")
        sources = ", ".join(c["sources"]) if c["agent"] == "researcher" else "—"
        rows.append(f"| {c['id']} | {c['agent']} | {c['attempt']} | {tools} | {sources or '—'} | {first} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente (Supervisor + Investigador + Analista).")
    parser.add_argument("question", nargs="?", help="Consulta a resolver.")
    parser.add_argument("--demo", action="store_true", help="Corre la consulta de demostración.")
    parser.add_argument("--diagram", action="store_true", help="Exporta el diagrama del grafo a docs/.")
    args = parser.parse_args(argv)

    from graph import build_graph, export_diagram

    graph = build_graph()
    if args.diagram:
        mermaid_path, png_path = export_diagram(graph)
        print(f"Diagrama: {mermaid_path}" + (f" y {png_path}" if png_path else " (PNG no disponible sin red)"))
        return 0

    question = DEMO_QUESTION if args.demo else args.question
    if not question:
        parser.error("pasá una consulta o usá --demo")
    print(f"Consulta: {question}\n")
    result = run_with_trace(graph, question)
    print("=" * 80 + "\nRESPUESTA FINAL\n" + "=" * 80)
    print(result.state["final_answer"])
    print(f"\n[{result.usage.summary()} · {result.seconds:.1f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
