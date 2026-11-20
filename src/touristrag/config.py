from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    data_url: str = "https://drive.google.com/uc?id=1P1BsvI2jPN3fEqjc2YZxmQ-MTs22WVUk"
    data_path: str = "data/file.csv"
    artifacts_dir: str = "artifacts"
    column_overrides: dict = field(default_factory=dict)

    desc_strategy: str = "concat"
    max_descriptions: int = 3
    max_captions: int = 2
    chunk_tokens: int = 200
    chunk_overlap: int = 1
    include_captions: bool = True

    embed_model: str = "intfloat/multilingual-e5-base"
    vector_backend: str = "chroma"
    retrieval_mode: str = "hybrid"
    k_retrieve: int = 20
    k_context: int = 4
    rrf_k: int = 60
    context_order: str = "edges"
    rerank_backend: str = "cross-encoder"
    rerank_model: str | None = None

    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_4bit: bool = False
    max_new_tokens: int = 300

    n_eval: int = 100
    blind_ratio: float = 0.5
    seed: int = 42