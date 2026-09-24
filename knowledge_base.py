"""Base vectorial del Investigador: ChromaDB local con el corpus Orbital de la Pre-entrega 3.

- Chunking por secciones Markdown (`## `): cada fragmento es una sección completa con
  el título del documento adelante, así la cita `archivo#n` apunta a algo legible.
- La colección guarda en su metadata el modelo de embeddings y un hash del corpus.
  Si cambia cualquiera de los dos, se reindexa sola: nunca se consulta con vectores
  viejos ni de otro modelo.

Uso:
    python knowledge_base.py            # indexa si hace falta
    python knowledge_base.py --rebuild  # fuerza la reindexación
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings, settings as default_settings

log = logging.getLogger("knowledge_base")

MAX_SECTION_CHARS = 2500  # si una sección es más larga, se parte por párrafos


# --------------------------------------------------------------------------- carga y chunking
def load_documents(knowledge_dir: Path) -> list[Document]:
    paths = sorted(p for p in Path(knowledge_dir).glob("*.md") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No hay archivos .md en {knowledge_dir}")
    return [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name}) for p in paths]


def _split_long(text: str, limit: int) -> list[str]:
    """Agrupa párrafos hasta `limit` caracteres (solo para secciones muy largas)."""
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) + 2 > limit:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    return parts + [current] if current else parts


def split_sections(doc: Document) -> list[Document]:
    """Un fragmento por sección `## `, con el título del documento como contexto."""
    text = doc.page_content.strip()
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else doc.metadata["source"]

    chunks: list[Document] = []
    for block in re.split(r"(?m)^(?=## )", text):
        block = block.strip()
        if not block or re.fullmatch(r"#\s+.+", block):  # vacío o solo el título del documento
            continue
        heading = re.match(r"##\s+(.+)", block)
        section = heading.group(1).strip() if heading else title
        for piece in _split_long(block, MAX_SECTION_CHARS):
            if not piece.startswith("# "):
                piece = f"# {title}\n\n{piece}"
            n = len(chunks)
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": doc.metadata["source"],
                        "chunk_id": n,
                        "section": section,
                        "id": f"{doc.metadata['source']}#{n}",
                    },
                )
            )
    return chunks


def corpus_hash(docs: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.metadata["source"].encode())
        digest.update(doc.page_content.encode())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- persistencia
def _client(persist_dir: Path) -> chromadb.ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))


def build_store(embeddings: Embeddings, cfg: Settings = default_settings, rebuild: bool = False) -> Chroma:
    """Devuelve el vectorstore listo para consultar, indexando solo si hace falta."""
    docs = load_documents(cfg.knowledge_dir)
    fingerprint = {"embedding_model": cfg.embedding_model, "corpus_hash": corpus_hash(docs)}
    client = _client(cfg.persist_dir)

    try:
        existing = client.get_collection(cfg.collection_name)
        meta = existing.metadata or {}
        stale = any(meta.get(k) != v for k, v in fingerprint.items())
        if rebuild or stale or existing.count() == 0:
            log.info("Reindexando '%s' (rebuild=%s, desactualizada=%s)", cfg.collection_name, rebuild, stale)
            client.delete_collection(cfg.collection_name)
        else:
            log.info("Colección '%s' al día: %d fragmentos", cfg.collection_name, existing.count())
            return Chroma(client=client, collection_name=cfg.collection_name, embedding_function=embeddings)
    except Exception:  # la colección no existe todavía (la excepción varía según versión de chromadb)
        pass

    store = Chroma(
        client=client,
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine", **fingerprint},
    )
    chunks = [chunk for doc in docs for chunk in split_sections(doc)]
    store.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])
    log.info("Indexados %d fragmentos de %d documentos", len(chunks), len(docs))
    return store


@lru_cache(maxsize=4)
def get_store(cfg: Settings = default_settings) -> Chroma:
    """Vectorstore compartido del proceso: uno por configuración, construido una sola vez."""
    from config import build_embeddings

    return build_store(build_embeddings(cfg), cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indexa el corpus de knowledge/ en ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Borra la colección y reindexa todo.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from config import build_embeddings

    store = build_store(build_embeddings(default_settings), default_settings, rebuild=args.rebuild)
    print(f"OK: {store._collection.count()} fragmentos en '{default_settings.collection_name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
