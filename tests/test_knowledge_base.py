"""Base vectorial del Investigador, con embeddings falsos (sin API)."""

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from agents.research_agent import make_research_tools
from config import ROOT_DIR, Settings
from knowledge_base import build_store, load_documents, split_sections


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.inner = DeterministicFakeEmbedding(size=32)
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def make_cfg(tmp_path, knowledge_dir):
    return Settings(knowledge_dir=knowledge_dir, persist_dir=tmp_path / "vs", collection_name="test", embedding_model="fake")


def test_chunking_por_secciones_del_corpus_real():
    chunks = [chunk for doc in load_documents(ROOT_DIR / "knowledge") for chunk in split_sections(doc)]
    ids = [c.metadata["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.page_content.startswith("# ") for c in chunks)  # cada fragmento lleva el título del documento
    manifest = [c for c in chunks if c.metadata["section"] == "Ejemplo de manifiesto mínimo"]
    assert manifest and "512Mi" in manifest[0].page_content


def test_indexa_una_sola_vez_y_reindexa_si_cambia_el_corpus(tmp_path):
    knowledge = tmp_path / "k"
    knowledge.mkdir()
    (knowledge / "a.md").write_text("# A\n\n## Uno\n\nDato uno: 42.\n\n## Dos\n\nDato dos: 7.", encoding="utf-8")
    embeddings = CountingEmbeddings()
    cfg = make_cfg(tmp_path, knowledge)

    assert build_store(embeddings, cfg)._collection.count() == 2
    build_store(embeddings, cfg)
    assert embeddings.embedded == 2  # la segunda vez no volvió a llamar a los embeddings

    (knowledge / "b.md").write_text("# B\n\n## Tres\n\nDato tres.", encoding="utf-8")
    assert build_store(embeddings, cfg)._collection.count() == 3


def test_search_docs_devuelve_citas_y_artefacto(tmp_path):
    knowledge = tmp_path / "k"
    knowledge.mkdir()
    (knowledge / "a.md").write_text("# A\n\n## Uno\n\nDato uno: 42.\n\n## Dos\n\nDato dos: 7.", encoding="utf-8")
    store = build_store(CountingEmbeddings(), make_cfg(tmp_path, knowledge))
    search_docs, list_sources = make_research_tools(store, top_k=2)

    message = search_docs.invoke({"name": "search_docs", "args": {"query": "dato"}, "id": "1", "type": "tool_call"})
    assert "[a.md#" in message.content
    assert {hit["id"] for hit in message.artifact} == {"a.md#0", "a.md#1"}
    assert list_sources.invoke({}) == "- a.md: 2 fragmentos"
