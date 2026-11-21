import numpy as np
from helpers import make_rag

from touristrag.pipeline import build_messages, reorder_edges
from touristrag.retrieval import BM25, Reranker, rrf, tokenize


def test_bm25_ranks_matching_doc_first_with_russian_morphology():
    docs = [tokenize(t) for t in ["Собор с золотыми куполами", "Мост через реку", "Театр оперы и балета"]]
    bm = BM25(docs)
    assert int(np.argmax(bm.scores(tokenize("золотой купол собора")))) == 0


def test_rrf_prefers_docs_found_by_both_rankers():
    fused = rrf({"a": ["x", "y", "z"], "b": ["y", "w", "x"]})
    assert fused[0][0] in {"x", "y"} and fused[0][0] == "y"
    assert rrf({"a": ["x"]}, weights={"a": 2.0})[0][1] > rrf({"a": ["x"]})[0][1]


def test_hybrid_retrieval_finds_named_and_blind_queries():
    rag, places = make_rag()
    gold = places.set_index("name")["place_id"]
    assert rag.rank_places("Расскажи про Медный всадник")[0] == gold["Медный всадник"]
    assert gold["Мечеть Кул-Шариф"] in rag.rank_places("голубые минареты джума-мечеть Татарстана")[:3]
    for mode in ("dense", "bm25", "hybrid"):
        assert rag.rank_places("памятник Петру I Фальконе", mode=mode)[0] == gold["Медный всадник"]


def test_place_dedup_and_context_order():
    rag, _ = make_rag()
    hits = rag.retrieve("собор с куполами")
    assert len({h["place_id"] for h in hits}) == len(hits)
    assert reorder_edges([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]


def test_answer_builds_grounded_prompt_and_returns_sources():
    from helpers import FakeLLM
    llm = FakeLLM(lambda p: "Это Медный всадник.")
    rag, places = make_rag(llm=llm)
    res = rag.answer("Где стоит памятник Петру I?")
    prompt = llm.calls[-1]
    assert "Контекст:" in prompt and "Вопрос: Где стоит памятник Петру I?" in prompt
    assert res["answer"] == "Это Медный всадник." and len(res["contexts"]) == len(res["place_ids"]) <= 3
    assert "координаты" in prompt  # coordinates are exposed to the LLM


def test_reranker_none_is_passthrough_and_unknown_backend_raises():
    hits = [{"text": "a"}, {"text": "b"}]
    assert Reranker("none").rerank("q", hits, top_k=1) == hits[:1]
    try:
        Reranker("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError
