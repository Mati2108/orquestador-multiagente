"""Estado compartido del orquestador: el "pizarrón" que leen y escriben todos los nodos.

Regla de oro: los campos que escriben varios nodos (messages, contributions) son
append-only con reducer, y los de reemplazo tienen un único escritor. Así ningún nodo
puede pisar lo que escribió otro y siempre se sabe quién aportó qué y en qué intento.

| Campo                                        | Lo escribe                                   |
|----------------------------------------------|----------------------------------------------|
| messages                                     | usuario (pregunta) y especialistas (su aporte)|
| next_agent, current_instruction, step,       | supervisor                                   |
| task_completed, decisions                    |                                              |
| contributions                                | researcher, analyst, synthesizer             |
| final_answer                                 | synthesizer (el supervisor solo le agrega    |
|                                              | notas si cierra con problemas pendientes)    |
| validation                                   | validator                                    |
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

AgentName = Literal["researcher", "analyst", "synthesizer"]
Route = Literal["researcher", "analyst", "synthesizer", "FINISH"]
AGENTS: tuple[AgentName, ...] = ("researcher", "analyst", "synthesizer")


class ToolCallRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    output: str


class Contribution(TypedDict):
    id: int  # orden global del aporte: 1, 2, 3...
    agent: AgentName
    attempt: int  # 1 = primer intento; 2+ = refinamiento pedido por el supervisor
    instruction: str  # lo que pidió el supervisor (el agente no ve nada más de la orquestación)
    content: str  # la salida del agente
    sources: list[str]  # ids de fragmentos recuperados (investigador) o herramientas usadas
    tool_calls: list[ToolCallRecord]  # registro de cada llamada, con la salida tal como la vio el agente
    evidence: list[str]  # lo verificable: texto de los fragmentos recuperados y resultados de cálculo
    status: Literal["ok", "error"]


class Decision(TypedDict):
    step: int
    route: Route
    instruction: str
    evaluation: str  # el último aporte evaluado contra la rúbrica del supervisor
    reason: str
    guard: str | None  # si una regla dura corrigió la decisión del LLM, por qué


class ValidationReport(TypedDict):
    passed: bool
    checked_numbers: int
    unsupported_numbers: list[str]  # números de la respuesta sin respaldo en ninguna herramienta
    invalid_citations: list[str]  # citas a fragmentos (o archivos) que el investigador nunca recuperó
    cited_sources: list[str]
    warnings: list[str]


class OrchestratorState(MessagesState):
    # --- control de flujo (solo el supervisor)
    next_agent: Route | None
    current_instruction: str
    step: int
    task_completed: bool
    # --- trazabilidad (append-only)
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    # --- resultado
    final_answer: str
    validation: ValidationReport | None


def initial_state(question: str) -> OrchestratorState:
    """Estado de arranque para una consulta nueva."""
    return OrchestratorState(
        messages=[HumanMessage(content=question)],
        next_agent=None,
        current_instruction="",
        step=0,
        task_completed=False,
        contributions=[],
        decisions=[],
        final_answer="",
        validation=None,
    )


# ----------------------------------------------------------------------------- lecturas
def get_question(state: OrchestratorState) -> str:
    """La consulta original: el primer mensaje del usuario."""
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def contributions_by(state: OrchestratorState, agent: AgentName) -> list[Contribution]:
    return [c for c in state.get("contributions", []) if c["agent"] == agent]


def attempts_used(state: OrchestratorState) -> dict[str, int]:
    """Cuántas veces trabajó cada agente. Se deriva del historial, no de un contador aparte."""
    return {agent: len(contributions_by(state, agent)) for agent in AGENTS}
