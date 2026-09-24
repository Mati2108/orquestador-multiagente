"""Validador: control determinístico (sin LLM) del borrador antes de que el supervisor cierre.

Dos reglas duras:
1. Todo número de la respuesta tiene que aparecer en lo que devolvió alguna herramienta
   (el texto de un fragmento recuperado de la base vectorial o un cálculo del Analista)
   o en la consulta del usuario. Lo que el texto de un agente "dice" no cuenta como
   evidencia, y tampoco la metadata del formato (ids de fragmentos, puntajes): así se
   detectan números inventados o calculados de cabeza.
2. Toda cita tiene que apuntar a algo que el Investigador realmente recuperó: una cita
   a fragmento [archivo#n], a ese fragmento; una cita a archivo [archivo.md], a un
   archivo del que se recuperó al menos un fragmento.

Es una heurística: no entiende el significado, solo verifica trazabilidad. Por eso su
veredicto vuelve al supervisor, que decide si corregir o cerrar.
"""

from __future__ import annotations

import re

from state import OrchestratorState, ValidationReport, contributions_by, get_question

CHUNK_REF_RE = re.compile(r"[\w\-]+\.md\s*#\s*\d+")
FILE_REF_RE = re.compile(r"[\w\-]+\.md(?!\s*#\s*\d)")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
NUMBER_RE = re.compile(r"(?<![A-Za-z_\d.,])\d+(?:[.,]\d+)*")


def number_candidates(token: str) -> set[float]:
    """Interpretaciones posibles de un número escrito en español o en inglés.

    "5.120" puede ser 5120 (miles) o 5,12 (decimal); "0,5" es 0.5. Se aceptan ambas
    lecturas para no castigar el formato, solo la ausencia de respaldo.
    """
    if "." in token and "," in token:
        if token.rfind(",") > token.rfind("."):  # 5.120,5 (es)
            return {float(token.replace(".", "").replace(",", "."))}
        return {float(token.replace(",", ""))}  # 5,120.5 (en)
    separator = "." if "." in token else "," if "," in token else None
    if separator is None:
        return {float(token)}
    parts = token.split(separator)
    options: set[float] = set()
    if len(parts) == 2:
        options.add(float(f"{parts[0]}.{parts[1]}"))
    if 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
        options.add(float("".join(parts)))
    return options  # vacío = no es un número (ej. una versión "1.2.3"), se ignora


def extract_numbers(text: str) -> list[tuple[str, set[float]]]:
    return [(m.group(), c) for m in NUMBER_RE.finditer(text) if (c := number_candidates(m.group()))]


def _normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref)


def validate_answer(answer: str, evidence_texts: list[str], retrieved_ids: set[str]) -> ValidationReport:
    chunk_refs = [_normalize_ref(r) for r in CHUNK_REF_RE.findall(answer)]
    file_refs = FILE_REF_RE.findall(answer)
    # Los dígitos de las citas ("02_...md#4") no son datos: se sacan antes de buscar números.
    body = LIST_MARKER_RE.sub("", FILE_REF_RE.sub(" ", CHUNK_REF_RE.sub(" ", answer)))

    evidence: set[float] = set()
    for text in evidence_texts:
        for _, candidates in extract_numbers(text):
            evidence.update(round(c, 6) for c in candidates)

    checked = extract_numbers(body)
    unsupported = [token for token, candidates in checked if not {round(c, 6) for c in candidates} & evidence]
    retrieved_files = {ref.split("#")[0] for ref in retrieved_ids}
    invalid = sorted({ref for ref in chunk_refs if ref not in retrieved_ids} | {f for f in file_refs if f not in retrieved_files})

    warnings = []
    if retrieved_ids and not chunk_refs and not file_refs:
        warnings.append("La respuesta no cita ningún fragmento.")
    return ValidationReport(
        passed=not unsupported and not invalid,
        checked_numbers=len(checked),
        unsupported_numbers=list(dict.fromkeys(unsupported)),
        invalid_citations=invalid,
        cited_sources=list(dict.fromkeys(chunk_refs + file_refs)),
        warnings=warnings,
    )


def validator(state: OrchestratorState) -> dict:
    if not state.get("final_answer"):  # el sintetizador falló y no hay borrador anterior
        return {
            "validation": ValidationReport(
                passed=False, checked_numbers=0, unsupported_numbers=[], invalid_citations=[], cited_sources=[],
                warnings=["No hay borrador para validar."],
            )
        }
    evidence = [get_question(state)]
    retrieved: set[str] = set()
    for agent in ("researcher", "analyst"):
        for contribution in contributions_by(state, agent):
            evidence.extend(contribution["evidence"])
            if agent == "researcher":
                retrieved.update(contribution["sources"])

    report = validate_answer(state.get("final_answer", ""), evidence, retrieved)
    errored = [c["agent"] for c in state.get("contributions", []) if c["status"] == "error"]
    if errored:
        report["warnings"].append(f"Aportes con error: {', '.join(errored)}.")
    return {"validation": report}
