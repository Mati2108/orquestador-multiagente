"""Configuración centralizada. Todo sale de variables de entorno / archivo .env.

Un solo proveedor (Google Gemini) para el LLM y para los embeddings: el proyecto
corre con una única clave, GOOGLE_API_KEY (capa gratuita en
https://aistudio.google.com/apikey).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # Modelos por rol. El supervisor (y el sintetizador, que redacta la respuesta final)
    # usan el modelo más capaz; los especialistas, con tareas acotadas y herramientas,
    # un modelo liviano. En la capa gratuita cada modelo tiene su propia cuota, así que
    # repartir los roles también reparte el consumo.
    supervisor_model: str = field(default_factory=lambda: os.getenv("SUPERVISOR_MODEL", "gemini-3-flash-preview"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"))
    # Los Gemini 3 son modelos "pensantes": el razonamiento sale del mismo max_output_tokens
    # (por eso el margen amplio) y thinking_level="low" acorta mucho cada llamada.
    thinking_level: str = field(default_factory=lambda: os.getenv("LLM_THINKING_LEVEL", "low"))
    max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 8192))
    # Ritmo máximo por modelo. La capa gratuita permite 5 pedidos por minuto; 0 = sin límite.
    llm_rpm: int = field(default_factory=lambda: _env_int("LLM_RPM", 5))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))

    # Base de conocimiento: el corpus Orbital de la Pre-entrega 3 (rag-local).
    knowledge_dir: Path = field(default_factory=lambda: Path(os.getenv("KNOWLEDGE_DIR", ROOT_DIR / "knowledge")))
    persist_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", ROOT_DIR / "vectorstore")))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "orbital_docs"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # Límites del orquestador: la defensa contra el "supervisor infinito".
    max_steps: int = field(default_factory=lambda: _env_int("MAX_STEPS", 8))  # decisiones del supervisor
    max_attempts_per_agent: int = field(default_factory=lambda: _env_int("MAX_ATTEMPTS_PER_AGENT", 2))
    agent_recursion_limit: int = field(default_factory=lambda: _env_int("AGENT_RECURSION_LIMIT", 12))  # loop ReAct
    graph_recursion_limit: int = field(default_factory=lambda: _env_int("GRAPH_RECURSION_LIMIT", 40))  # red final

    def require_api_key(self) -> None:
        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "Falta GOOGLE_API_KEY. Copiá .env.example a .env y completá la clave "
                "(nunca la escribas en el código ni la subas al repositorio)."
            )


settings = Settings()
_rate_limiters: dict[str, object] = {}  # uno por modelo: la cuota de Gemini es por modelo


def build_chat_model(cfg: Settings = settings, model: str | None = None):
    """Único punto donde se instancia un LLM. Por defecto, el de los especialistas."""
    cfg.require_api_key()
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = model or cfg.llm_model
    extra: dict = {"thinking_config": {"thinking_level": cfg.thinking_level}} if cfg.thinking_level else {}
    if cfg.llm_rpm > 0:
        if model not in _rate_limiters:
            # 10 % de margen para no rozar la ventana de un minuto de la API.
            _rate_limiters[model] = InMemoryRateLimiter(requests_per_second=cfg.llm_rpm * 0.9 / 60, check_every_n_seconds=0.2)
        extra["rate_limiter"] = _rate_limiters[model]
    # Pocos reintentos: con la cuota diaria agotada (429) reintentar solo demora el error.
    return ChatGoogleGenerativeAI(model=model, max_output_tokens=cfg.max_output_tokens, max_retries=3, timeout=120, **extra)


def build_embeddings(cfg: Settings = settings):
    """Mismo modelo de embeddings para indexar y para consultar (error #1 de la Pre-entrega 3)."""
    cfg.require_api_key()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=cfg.embedding_model)
