"""Piezas compartidas por los nodos especialistas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.errors import GraphRecursionError

from state import AgentName, Contribution, OrchestratorState, ToolCallRecord, contributions_by


@dataclass
class AgentRun:
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    artifacts: list[Any] = field(default_factory=list)
    status: str = "ok"


def short_error(exc: Exception) -> str:
    """Tipo y comienzo del mensaje: los errores de la API traen JSON larguísimo."""
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message[:160]}{'…' if len(message) > 160 else ''}"


def run_react_agent(agent: Runnable, task: str, config: RunnableConfig | None, recursion_limit: int) -> AgentRun:
    """Corre un agente ReAct con UNA instrucción y devuelve solo lo que importa afuera.

    El historial interno del agente (sus razonamientos y llamadas a herramientas) queda
    encapsulado: al estado global vuelve el texto final y el registro de herramientas.
    """
    run_config: RunnableConfig = {**(config or {}), "recursion_limit": recursion_limit}
    try:
        result = agent.invoke({"messages": [HumanMessage(content=task)]}, config=run_config)
    except GraphRecursionError:
        return AgentRun(content=f"Error: el agente superó el límite de {recursion_limit} pasos internos.", status="error")
    except Exception as exc:  # un especialista caído no debe tumbar al orquestador
        return AgentRun(content=f"Error del agente: {short_error(exc)}", status="error")

    messages = result["messages"]
    calls_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls_by_id[call["id"]] = call

    run = AgentRun(content="")
    for message in messages:
        if isinstance(message, ToolMessage):
            call = calls_by_id.get(message.tool_call_id, {})
            run.tool_calls.append(
                ToolCallRecord(tool=message.name or call.get("name", "?"), args=call.get("args", {}), output=message.text)
            )
            if message.artifact is not None:
                run.artifacts.append(message.artifact)

    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    run.content = final.text.strip() if final and final.text.strip() else "Error: el agente no devolvió texto final."
    if not (final and final.text.strip()):
        run.status = "error"
    return run


def new_contribution(state: OrchestratorState, agent: AgentName, content: str, **extra: Any) -> Contribution:
    return Contribution(
        id=len(state.get("contributions", [])) + 1,
        agent=agent,
        attempt=len(contributions_by(state, agent)) + 1,
        instruction=state.get("current_instruction", ""),
        content=content,
        sources=extra.get("sources", []),
        tool_calls=extra.get("tool_calls", []),
        status=extra.get("status", "ok"),
    )


def previous_attempt_block(state: OrchestratorState, agent: AgentName) -> str:
    """Si es un refinamiento, el agente ve su propio aporte anterior (y nada más del resto)."""
    previous = contributions_by(state, agent)
    if not previous:
        return ""
    return f"\n\nTu aporte anterior, que el supervisor pidió refinar:\n{previous[-1]['content']}"


def findings_block(state: OrchestratorState, agent: AgentName, title: str) -> str:
    """Los aportes de un agente como texto, del más viejo al más nuevo."""
    items = contributions_by(state, agent)
    if not items:
        return f"{title}: (sin aportes)"
    parts = [f"{title} — intento {c['attempt']}:\n{c['content']}" for c in items]
    return "\n\n".join(parts)
