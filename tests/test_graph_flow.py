"""Orquestación de punta a punta con dobles guionados: ruteo, refinamiento y condiciones de parada."""

from config import Settings
from graph import build_graph
from state import initial_state
from tests.doubles import (
    GOOD_ANSWER,
    INVENTED_ANSWER,
    MANIFEST_CONTENT,
    MANIFEST_REF,
    RESEARCH_ONLY_ANSWER,
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
    assert state["contributions"][0]["sources"] == [MANIFEST_REF]
    assert all(d["guard"] is None for d in state["decisions"])


def test_el_analista_recibe_solo_instruccion_y_hallazgos():
    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá la CPU total."), ("synthesizer", "R."), ("FINISH", "")])
    _, _, analyst_inputs, _ = run(decide)
    received = analyst_inputs[0]
    assert "Calculá la CPU total." in received and "Hallazgos:" in received
    assert QUESTION not in received  # no ve la conversación
    assert MANIFEST_CONTENT not in received  # ni los fragmentos crudos de la base vectorial
    assert "(guion)" not in received  # ni las evaluaciones del supervisor


def test_el_supervisor_pide_refinar_al_investigador():
    decide, _ = scripted_decider(
        [("researcher", "Buscá la CPU."), ("researcher", "Falta el máximo de réplicas: buscalo."), ("synthesizer", "R."), ("FINISH", "")]
    )
    state, research_inputs, _, _ = run(decide, synth_answers=(RESEARCH_ONLY_ANSWER,))
    assert [c["attempt"] for c in state["contributions"] if c["agent"] == "researcher"] == [1, 2]
    assert "Falta el máximo de réplicas" in research_inputs[1]
    assert "Tu aporte anterior" in research_inputs[1]


def test_regla_dura_corta_al_supervisor_que_insiste():
    decide, _ = scripted_decider([("researcher", "Buscá más.")] * 20)
    state, *_ = run(decide, synth_answers=(RESEARCH_ONLY_ANSWER,))
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
    state, *_ = run(failing_decider(), synth_answers=(RESEARCH_ONLY_ANSWER,))
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


def test_el_supervisor_pregunta_quien_interviene_o_si_hay_que_cerrar():
    decide, briefings = scripted_decider([("researcher", "Buscá."), ("synthesizer", "R."), ("FINISH", "")])
    run(decide)
    assert briefings[0].endswith("Dada la conversación actual, ¿quién debe intervenir ahora o es momento de finalizar?")
    assert "Aportes: todavía ninguno." in briefings[0]
    assert "#1 researcher (intento 1, estado ok)" in briefings[1]  # el ledger de aportes, no el historial crudo


def test_un_error_no_le_llega_como_dato_al_siguiente_agente():
    from langchain_core.runnables import RunnableLambda

    def broken(_inputs):
        raise RuntimeError("503 UNAVAILABLE")

    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá."), ("synthesizer", "R."), ("FINISH", "")])
    analyst, analyst_inputs = analyst_double()
    synthesizer, _ = scripted_llm(GOOD_ANSWER)
    graph = build_graph(decide=decide, research_agent=RunnableLambda(broken), analyst_agent=analyst, synthesizer_llm=synthesizer)
    state = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 40})
    assert state["contributions"][0]["status"] == "error"
    assert "503" not in analyst_inputs[0] and "(sin aportes)" in analyst_inputs[0]


def test_el_grafo_tambien_corre_async():
    import asyncio

    decide, _ = scripted_decider([("researcher", "Buscá."), ("analyst", "Calculá."), ("synthesizer", "R."), ("FINISH", "")])
    research, _ = research_double()
    analyst, _ = analyst_double()
    synthesizer, _ = scripted_llm(GOOD_ANSWER)
    graph = build_graph(decide=decide, research_agent=research, analyst_agent=analyst, synthesizer_llm=synthesizer)
    state = asyncio.run(graph.ainvoke(initial_state(QUESTION), config={"recursion_limit": 40}))
    assert routes(state) == ["researcher", "analyst", "synthesizer", "FINISH"]
    assert state["validation"]["passed"]


def test_la_metadata_de_las_herramientas_no_cuenta_como_evidencia():
    # El id del fragmento termina en 5 y el puntaje es 0.81: ninguno respalda un "5" o un "0,81" de la respuesta.
    from tests.doubles import scripted_agent

    artifact = [{"id": "manifiesto.md#5", "score": 0.81, "content": MANIFEST_CONTENT}]
    research, _ = scripted_agent("search_docs", f"[manifiesto.md#5] similitud 0.81\n{MANIFEST_CONTENT}", ["Hallazgos: ..."], artifact)
    decide, _ = scripted_decider([("researcher", "Buscá."), ("synthesizer", "R."), ("FINISH", ""), ("FINISH", "")])
    analyst, _ = analyst_double()
    synthesizer, _ = scripted_llm("Reserva 5 cores, similitud 0,81 [manifiesto.md#5].")
    graph = build_graph(decide=decide, research_agent=research, analyst_agent=analyst, synthesizer_llm=synthesizer)
    state = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 40})
    assert state["validation"]["unsupported_numbers"] == ["5", "0,81"]


def test_no_se_cierra_sin_haber_investigado():
    decide, _ = scripted_decider([("FINISH", ""), ("synthesizer", "R."), ("FINISH", "")])
    state, *_ = run(decide, synth_answers=(RESEARCH_ONLY_ANSWER,))
    assert routes(state) == ["researcher", "synthesizer", "FINISH"]
    assert "No se puede cerrar sin una respuesta" in state["decisions"][0]["guard"]


def test_un_aporte_posterior_al_borrador_obliga_a_actualizarlo():
    decide, _ = scripted_decider(
        [("researcher", "Buscá."), ("synthesizer", "R."), ("analyst", "Calculá 10 * 0.5."), ("FINISH", ""), ("FINISH", "")]
    )
    state, *_ = run(decide, synth_answers=(RESEARCH_ONLY_ANSWER, GOOD_ANSWER))
    assert routes(state) == ["researcher", "synthesizer", "analyst", "synthesizer", "FINISH"]
    assert "aportes posteriores" in state["decisions"][3]["guard"]
    assert state["final_answer"] == GOOD_ANSWER and state["validation"]["passed"]
