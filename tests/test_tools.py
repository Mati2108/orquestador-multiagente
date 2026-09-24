"""Herramientas del Analista: la calculadora segura y la conversión de unidades de Kubernetes."""

import pytest

from agents.analyst_agent import calculate, convert_quantity, convert_units, safe_eval


def test_calculadora_resuelve_aritmetica_y_funciones():
    assert safe_eval("10 * 0.5") == 5
    assert safe_eval("ceil(10 * 1.25)") == 13
    assert safe_eval("(2 + 3) ** 2") == 25
    assert calculate.invoke({"expression": "10 * 512"}) == "10 * 512 = 5120"


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "a + 1", "2 ** 1000", "(1).real", "1/0"])
def test_calculadora_rechaza_lo_que_no_es_aritmetica(expression):
    assert calculate.invoke({"expression": expression}).startswith("Error")


@pytest.mark.parametrize(
    ("quantity", "unit", "expected"),
    [("512Mi", "Gi", 0.5), ("5120Mi", "Gi", 5), ("500m", "cores", 0.5), ("2", "m", 2000), ("8Gi", "Mi", 8192), ("1G", "M", 1000)],
)
def test_conversiones_de_kubernetes(quantity, unit, expected):
    assert convert_quantity(quantity, unit) == pytest.approx(expected)


def test_conversion_formatea_el_resultado():
    assert convert_units.invoke({"quantity": "512Mi", "target_unit": "Gi"}) == "512Mi = 0.5 Gi"


def test_no_mezcla_cpu_con_memoria():
    assert convert_units.invoke({"quantity": "500m", "target_unit": "Gi"}).startswith("Error")
