from pathlib import Path

from app.errors import LLMUnavailableError


class PolicyRetriever:
    """Small local Chroma retrieval adapter; no network service is required."""

    def __init__(self, persist_directory: str | Path):
        self.persist_directory = Path(persist_directory)

    def search(self, query: str, limit: int = 3) -> list[str]:
        if not self.persist_directory.exists():
            return []
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            embedding = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
            )
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_collection("supplier_policies", embedding_function=embedding)
            count = collection.count()
            if count == 0:
                return []
            result = collection.query(query_texts=[query], n_results=min(limit, count))
            return [text for text in (result.get("documents") or [[]])[0] if text]
        except Exception as exc:
            raise LLMUnavailableError(f"Could not retrieve local supplier policy documents: {exc}") from exc


def ingest_policy_folder(folder: str | Path, persist_directory: str | Path) -> int:
    source = Path(folder)
    if not source.is_dir():
        raise FileNotFoundError(f"Policy folder does not exist: {source}")
    files = sorted([*source.rglob("*.md"), *source.rglob("*.txt")])
    docs: list[str] = []
    ids: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
            ids.append(path.relative_to(source).as_posix())
    if not docs:
        return 0
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embedding = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")
    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection("supplier_policies", embedding_function=embedding)
    collection.upsert(ids=ids, documents=docs)
    return len(docs)
