"""Orquestación de punta a punta con dobles guionados: ruteo, refinamiento y condiciones de parada."""

from config import Settings
from graph import build_graph
from state import initial_state
from tests.doubles import (
    GOOD_ANSWER,
    INVENTED_ANSWER,
    MANIFEST_CHUNK,
    analyst_double,
    failing_decider,
    failing_llm,
    research_double,
    scripted_decider,
    scripted_llm,
)

QUESTION = "¿Cuánta CPU reserva pagos-api en su máximo de réplicas?"


def run(decide, synth_answers=(GOOD_ANSWER,), cfg: Settings | None = None):
    research, research_inputs = research_double()
    analyst, analyst_inputs = analyst_double()
    synthesizer, synth_inputs = scripted_llm(*synth_answers)
    graph = build_graph(
        decide=decide, research_agent=research, analyst_agent=analyst, synthesizer_llm=synthesizer, cfg=cfg or Settings()
    )
    state = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 40})
    return state, research_inputs, analyst_inputs, synth_inputs


def routes(state):
    return [d["route"] for d in state["decisions"]]


def agents(state):
    return [c["agent"] for c in state["contributions"]]


def test_flujo_completo_investigador_analista_sintesis():
    decide, _ = scripted_decider(
        [("researcher", "Buscá cpu, memoria y réplicas máximas de pagos-api."), ("analyst", "Calculá 10 * 0.5."),
         ("synthesizer", "Redactá."), ("FINISH", "")]
    )
    state, *_ = run(decide)
    assert routes(state) == ["researcher", "analyst", "synthesizer", "FINISH"]
    assert agents(state) == ["researcher", "analyst", "synthesizer"]
    assert state["task_completed"] and state["validation"]["passed"]
    assert state["final_answer"] == GOOD_ANSWER
    assert state["contributions"][0]["sources"] == ["02_despliegue_y_cli.md#5"]
    assert all(d["guard"] is None for d in state["decisions"])


def test_el_analista_recibe_solo_instruccion_y_hallazgos():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá la CPU total."), ("synthesizer", "R."), ("FINISH", "")])
    _, _, analyst_inputs, _ = run(decide)
    received = analyst_inputs[0]
    assert "Calculá la CPU total." in received and "Hallazgos:" in received
    assert QUESTION not in received  # no ve la conversación
    assert MANIFEST_CHUNK not in received  # ni los fragmentos crudos de la base vectorial
    assert "(guion)" not in received  # ni las evaluaciones del supervisor


def test_el_supervisor_pide_refinar_al_investigador():
    decide, _ = scripted_decider(
        [("researcher", "Buscá la CPU."), ("researcher", "Falta el máximo de réplicas: buscalo."), ("synthesizer", "R."), ("FINISH", "")]
    )
    state, research_inputs, _, _ = run(decide)
    assert [c["attempt"] for c in state["contributions"] if c["agent"] == "researcher"] == [1, 2]
    assert "Falta el máximo de réplicas" in research_inputs[1]
    assert "Tu aporte anterior" in research_inputs[1]


def test_regla_dura_corta_al_supervisor_que_insiste():
    decide, _ = scripted_decider([("researcher", "Buscá más.")] * 20)
    state, *_ = run(decide)
    assert agents(state).count("researcher") == 2  # max_attempts_per_agent
    assert any(d["guard"] for d in state["decisions"])
    assert routes(state)[-1] == "FINISH"
    assert state["task_completed"]


def test_limite_de_pasos():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá.")] * 10)
    state, *_ = run(decide, cfg=Settings(max_steps=2, max_attempts_per_agent=5))
    assert "Límite de 2 decisiones" in state["decisions"][2]["guard"]
    assert routes(state)[-1] == "FINISH"


def test_validacion_fallida_obliga_a_corregir_antes_de_cerrar():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá."), ("synthesizer", "R."), ("FINISH", ""), ("FINISH", "")])
    state, _, _, synth_inputs = run(decide, synth_answers=(INVENTED_ANSWER, GOOD_ANSWER))
    assert agents(state).count("synthesizer") == 2
    assert "validación falló" in state["decisions"][3]["guard"]
    assert "'7'" in synth_inputs[1]  # el sintetizador recibe qué número no tenía respaldo
    assert state["validation"]["passed"] and state["final_answer"] == GOOD_ANSWER


def test_si_no_hay_mas_intentos_cierra_con_nota_de_validacion():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("synthesizer", "R.")] + [("FINISH", "")] * 5)
    state, *_ = run(decide, synth_answers=(INVENTED_ANSWER,))
    assert agents(state).count("synthesizer") == 2
    assert not state["validation"]["passed"]
    assert "Nota de validación" in state["final_answer"]


def test_si_el_llm_del_supervisor_falla_usa_la_politica_por_defecto():
    state, *_ = run(failing_decider())
    assert routes(state) == ["researcher", "synthesizer", "FINISH"]
    assert "falló" in state["decisions"][0]["evaluation"]


def test_si_el_sintetizador_falla_no_se_inventa_una_respuesta():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("synthesizer", "R.")] + [("FINISH", "")] * 5)
    research, _ = research_double()
    analyst, _ = analyst_double()
    graph = build_graph(decide=decide, research_agent=research, analyst_agent=analyst, synthesizer_llm=failing_llm())
    state = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 40})
    assert agents(state).count("synthesizer") == 2
    assert all(c["status"] == "error" for c in state["contributions"] if c["agent"] == "synthesizer")
    assert state["final_answer"].startswith("No se pudo generar una respuesta.")
    assert routes(state)[-1] == "FINISH"
