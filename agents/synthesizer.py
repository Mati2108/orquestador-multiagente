"""Sintetizador: la fase de síntesis final. Redacta la respuesta a partir de los aportes.

No tiene herramientas ni acceso a la documentación: solo puede combinar lo que ya
trajeron el Investigador y el Analista.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

from agents.common import findings_block, new_contribution, previous_attempt_block, short_error
from state import OrchestratorState, get_question

SYNTHESIZER_PROMPT = """Sos el Sintetizador de un equipo multi-agente. Redactás la respuesta final \
para el usuario usando SOLO los aportes del equipo.

Reglas:
1. Cada número que escribas tiene que estar en los aportes (hallazgos del Investigador o resultados \
del Analista). No calcules nada nuevo ni cambies el redondeo.
2. Citá los datos de la documentación con el formato [archivo#n], copiando las citas de los hallazgos. \
No inventes citas.
3. Si un agente trabajó más de una vez, ante una contradicción vale su intento más reciente.
4. Si alguna parte de la consulta no se pudo responder, decilo explícitamente.
5. Español rioplatense, claro y breve.

Formato:
<respuesta directa en 1 a 3 oraciones>

Detalle:
- <punto con su cita o su cálculo>

Fuentes: [archivo#n], ...
"""


def _validation_feedback(state: OrchestratorState) -> str:
    report = state.get("validation")
    if not report or report["passed"]:
        return ""
    return (
        "\n\nLa validación automática rechazó tu borrador anterior:"
        f"\n- números sin respaldo en las herramientas: {report['unsupported_numbers'] or 'ninguno'}"
        f"\n- citas a fragmentos que nunca se recuperaron: {report['invalid_citations'] or 'ninguna'}"
    )


def make_synthesizer_node(llm: Runnable) -> Callable[..., dict]:
    def synthesizer(state: OrchestratorState, config: RunnableConfig) -> dict:
        task = (
            f"Consulta del usuario:\n{get_question(state)}\n\n"
            f"Instrucción del supervisor:\n{state.get('current_instruction', '')}\n\n"
            f"{findings_block(state, 'researcher', 'Hallazgos del Investigador')}\n\n"
            f"{findings_block(state, 'analyst', 'Resultados del Analista')}"
            f"{previous_attempt_block(state, 'synthesizer')}"
            f"{_validation_feedback(state)}"
        )
        error, content = None, ""
        try:
            response = llm.invoke([SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=task)], config=config)
            content = response.text.strip()
            if not content:
                error = "Error del sintetizador: respuesta vacía."
        except Exception as exc:
            error = f"Error del sintetizador: {short_error(exc)}"
        if error:  # un error nunca se convierte en "respuesta": el borrador anterior queda como estaba
            return {"contributions": [new_contribution(state, "synthesizer", error, status="error")]}
        return {
            "contributions": [new_contribution(state, "synthesizer", content)],
            "final_answer": content,
            "messages": [AIMessage(content=content, name="synthesizer")],
        }

    return synthesizer
