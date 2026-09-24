"""Validador determinístico: números con respaldo y citas reales."""

from agents.validator import number_candidates, validate_answer

RETRIEVED = {"02_despliegue_y_cli.md#4"}


def test_lee_numeros_en_formato_espanol_e_ingles():
    assert {5120, 5.12} <= number_candidates("5.120")
    assert number_candidates("0,5") == {0.5}
    assert number_candidates("5.120,5") == {5120.5}
    assert number_candidates("1.2.3") == set()  # una versión, no un número


def test_aprueba_numeros_y_citas_con_respaldo():
    evidence = ["10 * 0.5 = 5", "10 * 512 = 5120", "[02_despliegue_y_cli.md#4] replicas max: 10"]
    answer = "En el máximo (10 réplicas) reserva 5 CPU y 5.120 Mi, 0,5 por pod [02_despliegue_y_cli.md#4]."
    report = validate_answer(answer, evidence + ["512Mi = 0.5 Gi"], RETRIEVED)
    assert report["passed"]
    assert report["checked_numbers"] == 4
    assert report["cited_sources"] == ["02_despliegue_y_cli.md#4"]


def test_rechaza_numero_inventado_y_cita_inexistente():
    report = validate_answer("Consume 7 CPU [03_configuracion_y_secretos.md#9].", ["10 * 0.5 = 5"], RETRIEVED)
    assert not report["passed"]
    assert report["unsupported_numbers"] == ["7"]
    assert report["invalid_citations"] == ["03_configuracion_y_secretos.md#9"]


def test_ignora_marcadores_de_lista_y_numeros_de_las_citas():
    answer = "1. Un dato [02_despliegue_y_cli.md#4]\n2. Otro dato"
    report = validate_answer(answer, [], RETRIEVED)
    assert report["passed"]
    assert report["checked_numbers"] == 0


def test_avisa_si_no_cita_nada_habiendo_fragmentos():
    report = validate_answer("No está en la documentación.", [], RETRIEVED)
    assert report["passed"]
    assert report["warnings"]


def test_cita_a_archivo_vale_solo_si_se_recupero_algo_de_ese_archivo():
    answer = "Dato [02_despliegue_y_cli.md]. Otro [03_configuracion_y_secretos.md].\n\nFuentes: [02_despliegue_y_cli.md]"
    report = validate_answer(answer, [], RETRIEVED)
    assert report["invalid_citations"] == ["03_configuracion_y_secretos.md"]
    assert report["checked_numbers"] == 0  # los "02"/"03" de los nombres de archivo no son datos
    assert not report["passed"]
