"""Embeddings, vector stores (Chroma / NumPy), BM25, RRF fusion, hybrid retriever and rerankers."""
from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- embedders
class Embedder:
    """sentence-transformers wrapper; adds E5 `query:`/`passage:` prefixes automatically."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._e5 = "e5" in model_name.lower()

    def encode(self, texts, kind: str = "passage") -> np.ndarray:
        texts = list(texts)
        if self._e5:
            texts = [f"{'query' if kind == 'query' else 'passage'}: {t}" for t in texts]
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 256)
        return np.asarray(emb, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder: an offline baseline and a test double."""

    def __init__(self, dim: int = 512, stem_len: int = 5):
        self.dim, self.stem_len = dim, stem_len

    def encode(self, texts, kind: str = "passage") -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t, self.stem_len):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# --------------------------------------------------------------------------- stores
class NumpyStore:
    """Brute-force cosine store (embeddings must be L2-normalised). Zero dependencies."""

    def __init__(self):
        self.ids: list[str] = []
        self.emb: np.ndarray | None = None

    def add(self, ids, embeddings, documents=None, metadatas=None):
        emb = np.asarray(embeddings, dtype=np.float32)
        self.ids += list(ids)
        self.emb = emb if self.emb is None else np.vstack([self.emb, emb])

    def query(self, embedding, k: int = 10) -> list[tuple[str, float]]:
        sims = self.emb @ np.asarray(embedding, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


class ChromaStore:
    def __init__(self, path: str | None = None, name: str = "places", reset: bool = True):
        import chromadb

        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        if reset:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, documents=None, metadatas=None, batch: int = 2048):
        ids, embeddings = list(ids), np.asarray(embeddings, dtype=np.float32)
        for s in range(0, len(ids), batch):
            sl = slice(s, s + batch)
            self.col.add(
                ids=ids[sl], embeddings=embeddings[sl].tolist(),
                documents=None if documents is None else list(documents)[sl],
                metadatas=None if metadatas is None else [_clean_meta(m) for m in list(metadatas)[sl]],
            )

    def query(self, embedding, k: int = 10) -> list[tuple[str, float]]:
        res = self.col.query(query_embeddings=[np.asarray(embedding).tolist()], n_results=k)
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


def _clean_meta(m: dict) -> dict:  # Chroma rejects None values
    return {k: v for k, v in m.items() if v is not None and k != "text"}


def make_store(backend: str = "chroma", path: str | None = None):
    if backend == "numpy":
        return NumpyStore()
    if backend == "chroma":
        return ChromaStore(path)
    raise ValueError(f"unknown vector backend {backend!r}")


# --------------------------------------------------------------------------- BM25
def tokenize(text: str, stem_len: int = 5) -> list[str]:
    """Lower-case word tokens truncated to `stem_len` chars — a cheap stemmer that works
    surprisingly well for Russian morphology without extra dependencies."""
    return [t[:stem_len] for t in re.findall(r"\w+", (text or "").lower())]


class BM25:
    """Okapi BM25 over pre-tokenised documents with an inverted index."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            counts: dict[str, int] = defaultdict(int)
            for t in d:
                counts[t] += 1
            for t, c in counts.items():
                self.postings[t].append((i, c))
        self.idf = {t: np.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(query_tokens):
            for i, tf in self.postings.get(t, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9))
                s[i] += self.idf[t] * tf * (self.k1 + 1) / norm
        return s


# --------------------------------------------------------------------------- fusion
def rrf(rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d))."""
    fused: dict[str, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def places_from_hits(hits: list[dict], n: int | None = None) -> list[str]:
    """Chunk hits -> ranked unique place ids (best chunk wins)."""
    seen, out = set(), []
    for h in hits:
        if h["place_id"] not in seen:
            seen.add(h["place_id"])
            out.append(h["place_id"])
    return out[:n] if n else out


def dedupe_by_place(hits: list[dict], n: int | None = None) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        if h["place_id"] not in seen:
            seen.add(h["place_id"])
            out.append(h)
    return out[:n] if n else out


class HybridRetriever:
    """Dense (vector store) + sparse (BM25) retrieval fused with RRF."""

    def __init__(self, embedder, store, chunks: list[dict], rrf_k: int = 60, stem_len: int = 5):
        self.embedder, self.store, self.rrf_k, self.stem_len = embedder, store, rrf_k, stem_len
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.ids = [c["chunk_id"] for c in chunks]
        self.bm25 = BM25([tokenize(c["text"], stem_len) for c in chunks])

    def _hit(self, chunk_id: str, score: float, source: str) -> dict:
        return {**self.chunks[chunk_id], "score": float(score), "source": source}

    def dense(self, query: str, k: int) -> list[dict]:
        q = self.embedder.encode([query], kind="query")[0]
        return [self._hit(i, s, "dense") for i, s in self.store.query(q, k)]

    def sparse(self, query: str, k: int) -> list[dict]:
        s = self.bm25.scores(tokenize(query, self.stem_len))
        top = np.argsort(-s)[:k]
        return [self._hit(self.ids[i], s[i], "bm25") for i in top if s[i] > 0]

    def retrieve(self, query: str, k: int = 20, mode: str = "hybrid", pool: int = 50) -> list[dict]:
        if mode == "dense":
            return self.dense(query, k)
        if mode == "bm25":
            return self.sparse(query, k)
        d, s = self.dense(query, pool), self.sparse(query, pool)
        fused = rrf({"dense": [h["chunk_id"] for h in d], "bm25": [h["chunk_id"] for h in s]}, self.rrf_k)
        return [self._hit(cid, sc, "hybrid") for cid, sc in fused[:k]]


# --------------------------------------------------------------------------- rerankers
class Reranker:
    """backend: 'none' | 'cross-encoder' (multilingual BGE) | 'colbert' (RAGatouille)."""

    def __init__(self, backend: str = "none", model: str | None = None, device: str | None = None):
        self.backend = backend
        if backend == "cross-encoder":
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model or "BAAI/bge-reranker-v2-m3", max_length=512, device=device)
        elif backend == "colbert":
            from ragatouille import RAGPretrainedModel

            self.model = RAGPretrainedModel.from_pretrained(model or "colbert-ir/colbertv2.0")
        elif backend != "none":
            raise ValueError(f"unknown reranker backend {backend!r}")

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        top_k = top_k or len(hits)
        if self.backend == "none" or not hits:
            return hits[:top_k]
        if self.backend == "cross-encoder":
            scores = self.model.predict([(query, h["text"]) for h in hits], show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))[:top_k]
            return [{**hits[i], "rerank_score": float(scores[i])} for i in order]
        # colbert via RAGatouille
        docs = [h["text"] for h in hits]
        res = self.model.rerank(query=query, documents=docs, k=min(top_k, len(docs)))
        out, used = [], set()
        for r in res:
            idx = r.get("result_index")
            if idx is None:  # older RAGatouille: map back by content
                idx = next(i for i, d in enumerate(docs) if d == r["content"] and i not in used)
            used.add(idx)
            out.append({**hits[idx], "rerank_score": float(r["score"])})
        return out
