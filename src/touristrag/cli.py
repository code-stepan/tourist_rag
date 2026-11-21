"""CLI:  python -m touristrag.cli {build,ask,eval} [options]"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import Config


def _cfg(a) -> Config:
    cfg = Config()
    for field, arg in [("data_path", "data"), ("artifacts_dir", "artifacts"), ("llm_model", "llm"),
                       ("rerank_backend", "rerank"), ("n_eval", "n"), ("embed_model", "embed")]:
        v = getattr(a, arg, None)
        if v is not None:
            setattr(cfg, field, v)
    return cfg


def cmd_build(a):
    from .cleaning import flag_outliers
    from .data import build_places, detect_schema, load_raw, standardize
    from .pipeline import build_index, save_artifacts
    from .retrieval import Embedder

    cfg = _cfg(a)
    raw = load_raw(cfg.data_path, cfg.data_url)
    df = standardize(raw, detect_schema(raw.columns, cfg.column_overrides))
    places = build_places(df, cfg.desc_strategy, cfg.max_descriptions, cfg.max_captions)
    print(f"{len(raw)} records -> {len(places)} unique places")
    embedder = Embedder(cfg.embed_model)
    if not a.no_clean:
        flagged = flag_outliers(places, embedder)
        print(f"flagged as junk: {int(flagged['is_junk'].sum())}")
        places = flagged[~flagged["is_junk"]].drop(columns=["rule_flag", "tfidf_atyp", "zs_margin", "junk_score",
                                                           "is_junk"]).reset_index(drop=True)
    store_path = os.path.join(cfg.artifacts_dir, "chroma") if cfg.vector_backend == "chroma" else None
    chunks, _, _ = build_index(places, embedder, cfg, store_path)
    save_artifacts(cfg.artifacts_dir, places, chunks)
    print(f"saved {len(places)} places / {len(chunks)} chunks to {cfg.artifacts_dir}/")


def cmd_ask(a):
    from .pipeline import load_rag

    rag, _ = load_rag(_cfg(a))
    q = a.question or input("Вопрос: ")
    res = rag.answer(q)
    print("\n" + res["answer"] + "\n")
    for h in res["hits"]:
        print(f" • {h['name']} ({h['city']})  score={h.get('rerank_score', h['score']):.3f}")


def cmd_eval(a):
    from .evaluation import (answer_relevancy, context_precision, context_recall, data_preprocess, evaluate_retrieval,
                             faithfulness, run_rag_on_dataset, summarize)
    from .pipeline import load_rag
    from .utils import set_seed

    cfg = _cfg(a)
    set_seed(cfg.seed)
    rag, places = load_rag(cfg)
    llm_fn = lambda p: rag.llm(p)  # noqa: E731
    embed_fn = lambda xs: rag.retriever.embedder.encode(xs, kind="query")  # noqa: E731
    ds = data_preprocess(places, llm_fn, cfg.n_eval, cfg.blind_ratio, cfg.seed)
    retrieval = evaluate_retrieval(lambda q: rag.rank_places(q), ds)
    ds = run_rag_on_dataset(ds, rag)
    scores = {"answer_relevancy": answer_relevancy(ds["question"], ds["answer"], llm_fn, embed_fn)}
    if a.full:
        scores["faithfulness"] = faithfulness(ds["question"], ds["answer"], ds["contexts"], llm_fn)
        scores["context_precision"] = context_precision(ds["question"], ds["ground_truth"], ds["contexts"], llm_fn)
        scores["context_recall"] = context_recall(ds["ground_truth"], ds["contexts"], llm_fn)
    for k, v in scores.items():
        ds[k] = v
    os.makedirs("results", exist_ok=True)
    ds.to_csv("results/eval_samples.csv", index=False)
    summary = summarize(scores)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"retrieval": retrieval, "ragas": summary["mean"].to_dict(), "config": cfg.__dict__}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(json.dumps(retrieval, indent=2)); print(summary)


def main():
    p = argparse.ArgumentParser(prog="touristrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("build", cmd_build), ("ask", cmd_ask), ("eval", cmd_eval)]:
        s = sub.add_parser(name)
        s.add_argument("--data"); s.add_argument("--artifacts"); s.add_argument("--embed")
        s.add_argument("--llm"); s.add_argument("--rerank", choices=["none", "cross-encoder", "colbert"])
        s.add_argument("-n", type=int)
        if name == "build":
            s.add_argument("--no-clean", action="store_true")
        if name == "ask":
            s.add_argument("question", nargs="?")
        if name == "eval":
            s.add_argument("--full", action="store_true", help="also faithfulness + context precision/recall")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
