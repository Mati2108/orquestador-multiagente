"""Dobles guionados (sin API) para probar la orquestación de forma determinística.

Reemplazan al LLM del supervisor, a los agentes ReAct y al sintetizador, pero el grafo,
el estado, las reglas duras y el validador son los reales.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from agents.supervisor import SupervisorDecision

# El fragmento real del manifiesto en la base es 02_despliegue_y_cli.md#4.
MANIFEST_REF = "02_despliegue_y_cli.md#4"
MANIFEST_CONTENT = 'name: pagos-api\nresources:\n  cpu: "500m"\n  memory: "512Mi"\nreplicas:\n  min: 2\n  max: 10'
# Lo que ve el agente: encabezado con id y puntaje + contenido. Solo el contenido es evidencia.
MANIFEST_TOOL_OUTPUT = f"[{MANIFEST_REF}] sección: Ejemplo de manifiesto mínimo · similitud 0.81\n{MANIFEST_CONTENT}"
RESEARCH_FINDINGS = (
    f"Hallazgos:\n- pagos-api pide cpu 500m y memoria 512Mi, con replicas.max 10 [{MANIFEST_REF}]\n"
    "No encontrado:\n- nada"
)
PARTIAL_FINDINGS = f"Hallazgos:\n- pagos-api pide cpu 500m y memoria 512Mi [{MANIFEST_REF}]\nNo encontrado:\n- el máximo de réplicas"
ANALYSIS = "Cálculos:\n- CPU total: 10 * 0.5 = 5 cores (datos: 10 réplicas, 500m)\nDatos faltantes:\n- ninguno"
# Con análisis: el 5 sale de la calculadora.
GOOD_ANSWER = f"En el máximo de 10 réplicas, pagos-api reserva 5 cores de CPU [{MANIFEST_REF}]."
# Número que ninguna herramienta produjo.
INVENTED_ANSWER = f"En el máximo de 10 réplicas, pagos-api reserva 7 cores de CPU [{MANIFEST_REF}]."
# Sin análisis: solo datos que están en el fragmento.
RESEARCH_ONLY_ANSWER = f"pagos-api pide 500m de CPU y 512Mi de memoria por réplica, con un máximo de 10 réplicas [{MANIFEST_REF}]."


def scripted_decider(routes: list[tuple[str, str]]):
    """Supervisor guionado: devuelve las decisiones en orden y registra lo que vio."""
    queue = list(routes)
    seen: list[str] = []

    def decide(briefing: str, config=None) -> SupervisorDecision:
        seen.append(briefing)
        route, instruction = queue.pop(0) if queue else ("FINISH", "")
        return SupervisorDecision(evaluation="(guion)", next=route, instruction=instruction, reason="(guion)")

    return decide, seen


def failing_decider():
    def decide(briefing: str, config=None) -> SupervisorDecision:
        raise RuntimeError("LLM caído")

    return decide


def scripted_agent(tool_name: str, tool_output: str, final_texts: list[str], artifact=None):
    """Agente ReAct guionado: una llamada a herramienta y una respuesta final por turno (repite la última)."""
    received: list[str] = []

    def run(inputs: dict) -> dict:
        task = inputs["messages"][0]
        received.append(task.content)
        final_text = final_texts[min(len(received), len(final_texts)) - 1]
        call = {"name": tool_name, "args": {"input": "x"}, "id": f"call-{len(received)}", "type": "tool_call"}
        return {
            "messages": [
                task,
                AIMessage(content="", tool_calls=[call]),
                ToolMessage(content=tool_output, tool_call_id=call["id"], name=tool_name, artifact=artifact),
                AIMessage(content=final_text),
            ]
        }

    return RunnableLambda(run), received


def scripted_llm(*answers: str):
    """LLM guionado para el sintetizador: devuelve las respuestas en orden (repite la última)."""
    queue = list(answers)
    received: list[str] = []

    def run(messages) -> AIMessage:
        received.append(messages[-1].content)
        return AIMessage(content=queue.pop(0) if len(queue) > 1 else queue[0])

    return RunnableLambda(run), received


def research_double(partial_first: bool = False):
    """Investigador guionado. Con `partial_first`, el primer aporte omite el máximo de réplicas."""
    artifact = [{"id": MANIFEST_REF, "score": 0.81, "content": MANIFEST_CONTENT}]
    finals = [PARTIAL_FINDINGS, RESEARCH_FINDINGS] if partial_first else [RESEARCH_FINDINGS]
    return scripted_agent("search_docs", MANIFEST_TOOL_OUTPUT, finals, artifact)


def analyst_double():
    return scripted_agent("calculate", "10 * 0.5 = 5", [ANALYSIS])


def failing_llm():
    """LLM que siempre falla, como una API caída o sin cuota."""

    def run(messages) -> AIMessage:
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    return RunnableLambda(run)
