"""Supervisor: router inteligente y controlador de flujo del grafo.

Dos capas:
1. Un LLM con salida estructurada que evalúa el último aporte contra una rúbrica y
   elige el próximo paso con una instrucción autocontenida.
2. Reglas duras en código (guards) que el LLM no puede saltearse: presupuesto de
   pasos, intentos por agente y "no se cierra sin respuesta validada". Son las que
   garantizan que el grafo termina aunque el LLM insista.
"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from pydantic import BaseModel, Field

from config import Settings, settings as default_settings
from state import AGENTS, Decision, OrchestratorState, Route, ValidationReport, attempts_used, get_question

SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo de análisis e investigación. Coordinás: nunca \
respondés la consulta vos mismo.

Tu equipo:
- researcher (Investigador): busca en la documentación de Orbital (base vectorial). Es el único que \
puede traer datos.
- analyst (Analista): hace cálculos y conversiones de unidades con herramientas, sobre datos ya \
investigados. No busca información y no ve la consulta original: solo tu instrucción y los hallazgos \
del Investigador.
- synthesizer (Sintetizador): redacta la respuesta final con los aportes. Después, un validador \
automático revisa que cada número y cada cita tengan respaldo en las herramientas.
- FINISH: cerrar y entregar la respuesta.

Rúbrica para evaluar el último aporte:
Investigación — suficiente si:
  R1. Trae cada dato que la consulta necesita, con cita [archivo#n].
  R2. Declara lo que no encontró en lugar de completarlo.
Análisis — suficiente si:
  A1. Cada número derivado salió de una herramienta (calculate / convert_units).
  A2. Usa solo datos de la investigación, sin supuestos no declarados.
  A3. Resuelve todos los cálculos que pide la consulta.
Respuesta final — aceptable si:
  F1. La validación automática pasó.
  F2. Responde cada parte de la consulta o explica qué no se pudo responder.

Reglas de ruteo:
1. Sin datos todavía -> researcher.
2. Datos suficientes y la consulta pide cálculos -> analyst. Si no pide cálculos, o la investigación \
confirmó que el dato no está en la documentación, salteá al analyst.
3. Un aporte que no cumple la rúbrica vuelve al MISMO agente con una instrucción que diga exactamente \
qué falta o qué corregir. Solo por datos faltantes o errores, nunca por estilo.
4. Investigación y análisis suficientes -> synthesizer.
5. Borrador con la validación fallida -> synthesizer para corregir (o analyst si falta un cálculo).
6. Borrador validado que cumple F2 -> FINISH.
7. Un agente sin intentos restantes no puede volver a elegirse.

La instrucción es lo ÚNICO que el agente elegido ve de esta coordinación: tiene que ser autocontenida \
(qué buscar o qué calcular, con los valores y unidades relevantes)."""


class SupervisorDecision(BaseModel):
    evaluation: str = Field(description="Evaluación del último aporte contra la rúbrica (1-2 oraciones). 'Sin aportes' si no hay.")
    next: Route = Field(description="Próximo paso: researcher, analyst, synthesizer o FINISH.")
    instruction: str = Field(description="Instrucción autocontenida para el agente elegido. Vacía si next es FINISH.")
    reason: str = Field(description="Por qué ese próximo paso (1 oración).")


Decider = Callable[[str, RunnableConfig | None], SupervisorDecision]

DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "researcher": "Buscá en la documentación todos los datos necesarios para responder la consulta, con su cita.",
    "analyst": "Hacé con tus herramientas los cálculos que la consulta necesita, usando solo los datos investigados.",
    "synthesizer": "Redactá la respuesta final con los aportes disponibles y aclará qué no se pudo resolver.",
    "FINISH": "",
}


def build_supervisor_decider(llm: BaseChatModel) -> Decider:
    structured = llm.with_structured_output(SupervisorDecision)

    def decide(briefing: str, config: RunnableConfig | None = None) -> SupervisorDecision:
        decision = structured.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=briefing)], config=config
        )
        if decision is None:
            raise ValueError("el LLM no devolvió una decisión estructurada")
        return decision

    return decide


# --------------------------------------------------------------------------- lo que ve el supervisor
def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " […]"


def describe_validation(report: ValidationReport | None) -> str:
    if report is None:
        return "sin validar."
    warnings = f" Avisos: {' '.join(report['warnings'])}" if report["warnings"] else ""
    if report["passed"]:
        return f"APROBADA ({report['checked_numbers']} números verificados, citas: {report['cited_sources'] or 'ninguna'}).{warnings}"
    return (
        "RECHAZADA. Números sin respaldo: "
        f"{report['unsupported_numbers'] or 'ninguno'}; citas inexistentes: {report['invalid_citations'] or 'ninguna'}.{warnings}"
    )


def build_briefing(state: OrchestratorState, cfg: Settings, step: int) -> str:
    """Vista compacta del estado. No incluye los mensajes internos ni la salida cruda de las herramientas."""
    used = attempts_used(state)
    budget = ", ".join(f"{agent} {used[agent]}/{cfg.max_attempts_per_agent}" for agent in AGENTS)
    lines = [
        f"Consulta del usuario:\n{get_question(state)}",
        f"\nPresupuesto: decisión {step} de {cfg.max_steps}. Intentos usados: {budget}.",
    ]
    contributions = state.get("contributions", [])
    if not contributions:
        lines.append("\nAportes: todavía ninguno.")
    for c in contributions:
        lines.append(f"\n#{c['id']} {c['agent']} (intento {c['attempt']}, estado {c['status']})")
        lines.append(f"Instrucción recibida: {c['instruction']}")
        if c["sources"]:
            lines.append(f"Fuentes / herramientas: {', '.join(c['sources'])}")
        lines.append(_truncate(c["content"], 1800))
    if state.get("final_answer"):
        lines.append(f"\nValidación automática del último borrador: {describe_validation(state.get('validation'))}")
    lines.append("\nDada la conversación actual, ¿quién debe intervenir ahora o es momento de finalizar?")
    return "\n".join(lines)


# --------------------------------------------------------------------------- reglas duras
def default_policy(state: OrchestratorState) -> Route:
    """Plan B si el LLM falla: investigar, sintetizar, cerrar."""
    if not attempts_used(state)["researcher"]:
        return "researcher"
    return "synthesizer" if not state.get("final_answer") else "FINISH"


def draft_is_stale(state: OrchestratorState) -> bool:
    """Hay aportes de especialistas posteriores al último borrador: el borrador no los incluye."""
    contributions = [c for c in state.get("contributions", []) if c["status"] == "ok"]
    drafts = [c["id"] for c in contributions if c["agent"] == "synthesizer"]
    return bool(drafts) and any(c["id"] > drafts[-1] and c["agent"] != "synthesizer" for c in contributions)


def draft_ok(state: OrchestratorState) -> bool:
    validation = state.get("validation")
    return bool(state.get("final_answer")) and bool(validation and validation["passed"]) and not draft_is_stale(state)


def fallback_route(state: OrchestratorState, cfg: Settings) -> Route:
    """Hacia dónde ir cuando el LLM pide algo que el presupuesto no permite."""
    used = attempts_used(state)
    if not state.get("final_answer") and used["researcher"] == 0:
        return "researcher"  # nunca se responde sin haber buscado
    if not draft_ok(state) and used["synthesizer"] < cfg.max_attempts_per_agent:
        return "synthesizer"
    return "FINISH"


def apply_guards(proposed: Route, state: OrchestratorState, cfg: Settings) -> tuple[Route, str | None]:
    used = attempts_used(state)
    validation = state.get("validation")
    synth_left = used["synthesizer"] < cfg.max_attempts_per_agent
    if proposed == "FINISH":
        if not state.get("final_answer"):
            route = fallback_route(state, cfg)
            return route, f"No se puede cerrar sin una respuesta: se sigue con {route}."
        if validation and not validation["passed"] and synth_left:
            return "synthesizer", "La validación falló: se pide un borrador corregido antes de cerrar."
        if draft_is_stale(state) and synth_left:
            return "synthesizer", "Hay aportes posteriores al último borrador: se pide incorporarlos antes de cerrar."
        return "FINISH", None
    if used[proposed] >= cfg.max_attempts_per_agent:
        route = fallback_route(state, cfg)
        return route, f"{proposed} ya usó sus {cfg.max_attempts_per_agent} intentos: se sigue con {route}."
    return proposed, None


def guard_instruction(route: Route, state: OrchestratorState) -> str:
    validation = state.get("validation")
    if route == "synthesizer" and state.get("final_answer") and validation and not validation["passed"]:
        problems = []
        if validation["unsupported_numbers"]:
            problems.append(f"los números sin respaldo {validation['unsupported_numbers']}")
        if validation["invalid_citations"]:
            problems.append(f"las citas a fragmentos inexistentes {validation['invalid_citations']}")
        return (
            f"Corregí el borrador anterior: quitá o reemplazá por valores con respaldo {' y '.join(problems)}. "
            "No agregues datos nuevos."
        )
    if route == "synthesizer" and draft_is_stale(state):
        return "Actualizá la respuesta final: incorporá los aportes del equipo posteriores a tu borrador anterior."
    return DEFAULT_INSTRUCTIONS[route]


# --------------------------------------------------------------------------- nodo y arista condicional
def make_supervisor_node(decide: Decider, cfg: Settings = default_settings) -> Callable[..., dict]:
    def supervisor(state: OrchestratorState, config: RunnableConfig) -> dict:
        step = state.get("step", 0) + 1
        if step > cfg.max_steps:  # corte por presupuesto: ni siquiera se consulta al LLM
            proposed: Route = fallback_route(state, cfg)
            evaluation, reason, instruction = "Sin evaluar.", f"Se alcanzó el límite de {cfg.max_steps} decisiones.", ""
        else:
            try:
                decision = decide(build_briefing(state, cfg, step), config)
                proposed, evaluation = decision.next, decision.evaluation
                reason, instruction = decision.reason, decision.instruction
            except Exception as exc:
                proposed = default_policy(state)
                evaluation = f"El LLM del supervisor falló ({type(exc).__name__}); se aplica la política por defecto."
                reason, instruction = "Política por defecto.", ""

        route, guard = apply_guards(proposed, state, cfg)
        if step > cfg.max_steps:
            guard = f"Límite de {cfg.max_steps} decisiones alcanzado: se sigue con {route}."
        if guard or not instruction.strip():
            instruction = guard_instruction(route, state)
        if route == "FINISH":
            instruction = ""

        record = Decision(step=step, route=route, instruction=instruction, evaluation=evaluation, reason=reason, guard=guard)
        update: dict = {"step": step, "next_agent": route, "current_instruction": instruction, "decisions": [record]}
        if route == "FINISH":
            update["task_completed"] = True
            update.update(closing_answer(state))
        return update

    return supervisor


def closing_answer(state: OrchestratorState) -> dict:
    """Al cerrar sin intentos restantes, los problemas se declaran en la respuesta, no se esconden."""
    draft, validation = state.get("final_answer", ""), state.get("validation")
    if not draft:
        failed = [f"{c['agent']} (intento {c['attempt']})" for c in state.get("contributions", []) if c["status"] == "error"]
        return {"final_answer": "No se pudo generar una respuesta." + (f" Fallaron: {', '.join(failed)}." if failed else "")}
    notes = []
    if validation and not validation["passed"]:
        problems = []
        if validation["unsupported_numbers"]:
            problems.append(f"los números {validation['unsupported_numbers']}")
        if validation["invalid_citations"]:
            problems.append(f"las citas {validation['invalid_citations']}")
        notes.append(f"⚠️ Nota de validación: no se pudo verificar el respaldo de {' ni '.join(problems)}.")
    if draft_is_stale(state):
        notes.append("⚠️ Nota: la respuesta no incorpora los aportes posteriores al último borrador.")
    return {"final_answer": "\n\n".join([draft, *notes])} if notes else {}


def route_from_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "synthesizer", "__end__"]:
    """Arista condicional: traduce la decisión del supervisor al nombre del nodo destino."""
    route = state.get("next_agent")
    return route if route in AGENTS else END
