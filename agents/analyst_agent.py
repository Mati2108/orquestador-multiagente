"""Agente Analista: procesa los datos que ya trajo el Investigador.

Herramientas acotadas de cómputo: `calculate` (aritmética segura, sin eval) y
`convert_units` (cantidades de recursos de Kubernetes). No tiene acceso a la
documentación: si le falta un dato, lo tiene que declarar, no inventarlo.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool

from agents.common import findings_block, new_contribution, previous_attempt_block, run_react_agent
from config import Settings, settings as default_settings
from state import OrchestratorState

ANALYST_PROMPT = """Sos el Analista de un equipo multi-agente. Procesás datos que ya investigó otro \
agente; no tenés acceso a la documentación.

Reglas:
1. Todo número derivado (sumas, productos, porcentajes, conversiones de unidades) sale de una \
herramienta: `calculate` o `convert_units`. Nada de cálculo mental.
2. Usá solo los datos que te pasan. Si falta un dato para un cálculo, no lo supongas: listalo en \
"Datos faltantes". Si el Investigador trabajó más de una vez y sus intentos se contradicen, vale el \
más reciente.
3. Para cada cálculo indicá la operación, el resultado y de qué dato sale.

Formato de salida (sin preámbulo):
Cálculos:
- <qué se calculó>: <operación> = <resultado> (datos: ...)
Conclusiones:
- <lectura breve de los resultados>
Datos faltantes:
- <dato> (o "ninguno")
"""

# --------------------------------------------------------------------------- calculadora segura
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt}


def safe_eval(expression: str) -> float:
    """Evalúa aritmética recorriendo el AST: solo números, operadores y funciones permitidas."""
    if len(expression) > 300:
        raise ValueError("expresión demasiado larga")
    return _eval(ast.parse(expression.replace("^", "**"), mode="eval").body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("potencia demasiado grande")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(arg) for arg in node.args])
    raise ValueError(f"elemento no permitido: {type(node).__name__}")


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- unidades de Kubernetes
CPU_UNITS = {"m": 1e-3, "millicores": 1e-3, "cores": 1.0, "core": 1.0, "cpu": 1.0}
MEMORY_UNITS = {
    "bytes": 1, "B": 1,
    "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40,
    "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40,
    "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
    "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
}
_QUANTITY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*([A-Za-z]*)\s*$")


def _lookup(unit: str) -> tuple[str, float] | None:
    """'m' (mili) y 'M' (mega) se distinguen por mayúscula; las palabras no."""
    if unit in CPU_UNITS:
        return "cpu", CPU_UNITS[unit]
    if unit in MEMORY_UNITS:
        return "memory", MEMORY_UNITS[unit]
    if unit.lower() in CPU_UNITS and len(unit) > 1:
        return "cpu", CPU_UNITS[unit.lower()]
    return None


def convert_quantity(quantity: str, target_unit: str) -> float:
    match = _QUANTITY_RE.match(quantity)
    if not match:
        raise ValueError(f"cantidad inválida: {quantity!r} (ejemplos válidos: '500m', '512Mi', '2')")
    value, unit = float(match.group(1)), match.group(2)
    target = _lookup(target_unit.strip())
    if target is None:
        raise ValueError(f"unidad destino desconocida: {target_unit!r}")
    # Sin sufijo, la magnitud la define el destino: "2" -> 2 cores o 2 bytes.
    source = _lookup(unit) if unit else (target[0], 1.0)
    if source is None:
        raise ValueError(f"unidad de origen desconocida: {unit!r}")
    if source[0] != target[0]:
        raise ValueError(f"no se puede convertir {source[0]} a {target[0]}")
    return value * source[1] / target[1]


@tool
def calculate(expression: str) -> str:
    """Evalúa una expresión aritmética de forma segura. Soporta + - * / // % **, paréntesis y las \
funciones round, min, max, abs, ceil, floor, sqrt. Usá punto decimal. Ej: "10 * 0.5", "ceil(10 * 1.25)"."""
    try:
        return f"{expression} = {format_number(safe_eval(expression))}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}. Revisá la expresión (solo números, operadores y funciones permitidas)."


@tool
def convert_units(quantity: str, target_unit: str) -> str:
    """Convierte cantidades de recursos de Kubernetes. CPU: '500m' (milicores) o '2' (cores) a 'm' o \
'cores'. Memoria: Ki, Mi, Gi, Ti (base 1024), K, M, G, T (base 1000) o 'bytes'. \
Ej: convert_units("512Mi", "Gi") -> "512Mi = 0.5 Gi"."""
    try:
        return f"{quantity} = {format_number(convert_quantity(quantity, target_unit))} {target_unit}"
    except ValueError as exc:
        return f"Error: {exc}."


ANALYST_TOOLS = [calculate, convert_units]


def build_analyst_agent(llm: BaseChatModel) -> Runnable:
    return create_agent(llm, ANALYST_TOOLS, system_prompt=ANALYST_PROMPT, name="analyst")


def make_analyst_node(agent: Runnable, cfg: Settings = default_settings) -> Callable[..., dict]:
    def analyst(state: OrchestratorState, config: RunnableConfig) -> dict:
        # Contexto mínimo: la instrucción + los hallazgos del investigador. No ve la conversación,
        # ni las evaluaciones del supervisor, ni los fragmentos crudos de la base vectorial.
        task = (
            f"Instrucción del supervisor:\n{state.get('current_instruction', '')}\n\n"
            f"{findings_block(state, 'researcher', 'Datos del Investigador')}"
            f"{previous_attempt_block(state, 'analyst')}"
        )
        run = run_react_agent(agent, task, config, cfg.agent_recursion_limit)
        contribution = new_contribution(
            state,
            "analyst",
            run.content,
            sources=sorted({call["tool"] for call in run.tool_calls}),
            tool_calls=run.tool_calls,
            evidence=run.evidence,
            status=run.status,
        )
        return {"contributions": [contribution], "messages": [AIMessage(content=run.content, name="analyst")]}

    return analyst
