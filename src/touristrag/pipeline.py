"""Index building, persistence and the end-to-end TouristRAG pipeline."""
from __future__ import annotations

import json
import os

import pandas as pd

from .chunking import make_chunks
from .config import Config
from .retrieval import (Embedder, HybridRetriever, Reranker, dedupe_by_place, make_store, places_from_hits)

SYSTEM_PROMPT = (
    "Ты — дружелюбный туристический гид по достопримечательностям. Отвечай по-русски, кратко и по делу "
    "(2–5 предложений), опираясь ТОЛЬКО на приведённый контекст. Если в контексте нет ответа, честно скажи, "
    "что информации недостаточно, и не выдумывай факты. Называй достопримечательность и город."
)


def reorder_edges(hits: list[dict]) -> list[dict]:
    """Put the most relevant chunks at the start AND end of the prompt ('lost in the middle')."""
    return hits[0::2] + hits[1::2][::-1]


def format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        coords = f" (координаты: {h['lat']:.4f}, {h['lon']:.4f})" if h.get("lat") is not None and h.get("lon") is not None else ""
        lines.append(f"[{i}] {h['text']}{coords}")
    return "\n\n".join(lines)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    user = f"Контекст:\n{format_context(hits)}\n\nВопрос: {question}\nОтвет:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_index(places: pd.DataFrame, embedder, cfg: Config, store_path: str | None = None):
    """places -> chunks -> embeddings -> vector store. Returns (chunks, store, embeddings)."""
    chunks = make_chunks(places, getattr(embedder, "count_tokens", None), cfg.chunk_tokens,
                         cfg.chunk_overlap, cfg.include_captions)
    emb = embedder.encode([c["text"] for c in chunks], kind="passage")
    store = make_store(cfg.vector_backend, store_path)
    store.add([c["chunk_id"] for c in chunks], emb, [c["text"] for c in chunks],
              [{k: v for k, v in c.items() if k != "text"} for c in chunks])
    return chunks, store, emb


class TouristRAG:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker | None, llm, cfg: Config | None = None):
        self.retriever, self.reranker, self.llm = retriever, reranker or Reranker("none"), llm
        self.cfg = cfg or Config()

    def retrieve(self, question: str, mode: str | None = None, rerank: bool = True) -> list[dict]:
        """Ranked, place-deduplicated chunk hits (reranked if a reranker is configured)."""
        hits = self.retriever.retrieve(question, k=self.cfg.k_retrieve, mode=mode or self.cfg.retrieval_mode)
        if rerank:
            hits = self.reranker.rerank(question, hits, top_k=len(hits))
        return dedupe_by_place(hits)

    def rank_places(self, question: str, mode: str | None = None, rerank: bool = True) -> list[str]:
        return places_from_hits(self.retrieve(question, mode, rerank))

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)[: self.cfg.k_context]
        ordered = reorder_edges(hits) if self.cfg.context_order == "edges" else hits
        reply = self.llm.chat(build_messages(question, ordered))
        return {"question": question, "answer": reply, "contexts": [h["text"] for h in ordered],
                "place_ids": [h["place_id"] for h in hits], "hits": ordered}


# --------------------------------------------------------------------------- persistence
def save_artifacts(out_dir: str, places: pd.DataFrame, chunks: list[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    places.to_parquet(os.path.join(out_dir, "places.parquet"), index=False)
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_artifacts(out_dir: str):
    places = pd.read_parquet(os.path.join(out_dir, "places.parquet"))
    with open(os.path.join(out_dir, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return places, chunks


def load_rag(cfg: Config, with_llm: bool = True):
    """Rebuild a ready-to-use TouristRAG from `cfg.artifacts_dir` (see `cli build`)."""
    from .llm import HFChat
    from .retrieval import ChromaStore

    places, chunks = load_artifacts(cfg.artifacts_dir)
    embedder = Embedder(cfg.embed_model)
    chroma_dir = os.path.join(cfg.artifacts_dir, "chroma")
    if cfg.vector_backend == "chroma" and os.path.isdir(chroma_dir):
        store = ChromaStore(chroma_dir, reset=False)  # reuse the persisted collection
    else:
        _, store, _ = build_index(places, embedder, cfg)
    retriever = HybridRetriever(embedder, store, chunks, cfg.rrf_k)
    reranker = Reranker(cfg.rerank_backend, cfg.rerank_model)
    llm = HFChat(cfg.llm_model, cfg.llm_4bit, cfg.max_new_tokens) if with_llm else None
    return TouristRAG(retriever, reranker, llm, cfg), places
