"""CLI y diagrama: funcionan sin clave de API cuando no hace falta llamar al LLM."""

import pytest

import main
from graph import build_diagram_graph


def test_sin_consulta_muestra_el_uso_sin_pedir_la_clave(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(SystemExit) as exit_info:
        main.main([])
    assert exit_info.value.code == 2
    assert "pasá una consulta o usá --demo" in capsys.readouterr().err


def test_el_diagrama_no_necesita_clave(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    mermaid = build_diagram_graph().get_graph().draw_mermaid()
    for edge in ("supervisor -.-> researcher", "supervisor -.-> analyst", "supervisor -.-> synthesizer", "supervisor -.-> __end__"):
        assert edge in mermaid
