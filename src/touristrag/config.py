from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- data ---
    data_url: str = "https://drive.google.com/uc?id=1P1BsvI2jPN3fEqjc2YZxmQ-MTs22WVUk"
    data_path: str = "data/file.csv"
    artifacts_dir: str = "artifacts"
    column_overrides: dict = field(default_factory=dict)  # {"name": "my_col", ...}

    # --- place aggregation / chunking ---
    desc_strategy: str = "concat"      # "longest" | "concat"
    max_descriptions: int = 3
    max_captions: int = 2
    chunk_tokens: int = 200
    chunk_overlap: int = 1             # sentences
    include_captions: bool = True      # separate "visual" chunk from BLIP captions

    # --- retrieval ---
    embed_model: str = "intfloat/multilingual-e5-base"
    vector_backend: str = "chroma"     # "chroma" | "numpy"
    retrieval_mode: str = "hybrid"     # "dense" | "bm25" | "hybrid"
    k_retrieve: int = 20
    k_context: int = 4
    rrf_k: int = 60
    context_order: str = "edges"       # "desc" | "edges" (lost-in-the-middle mitigation)
    rerank_backend: str = "cross-encoder"   # "none" | "cross-encoder" | "colbert"
    rerank_model: str | None = None    # defaults: bge-reranker-v2-m3 / colbert-ir/colbertv2.0

    # --- generation ---
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"   # use Qwen2.5-7B-Instruct + llm_4bit=True for quality
    llm_4bit: bool = False
    max_new_tokens: int = 300

    # --- evaluation ---
    n_eval: int = 100
    blind_ratio: float = 0.5           # share of questions that must not name the landmark
    seed: int = 42
