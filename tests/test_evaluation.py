import numpy as np
from helpers import js, make_rag, places_frame

from touristrag.evaluation import (answer_relevancy, context_precision, context_recall, data_preprocess,
                                   evaluate_retrieval, faithfulness, retrieval_metrics, run_rag_on_dataset, summarize)
from touristrag.retrieval import HashingEmbedder

EMB = HashingEmbedder()
embed_fn = lambda xs: EMB.encode(list(xs), kind="query")


def test_answer_relevancy_high_for_on_topic_zero_for_noncommittal_nan_for_garbage():
    q = "Где находится Медный всадник?"
    on_topic = lambda p: js({"questions": [q, "Где стоит Медный всадник?", "Где Медный всадник?"], "noncommittal": 0})
    off_topic = lambda p: js({"questions": ["Сколько стоит билет в театр?"], "noncommittal": 0})
    dunno = lambda p: js({"questions": [q], "noncommittal": 1})
    garbage = lambda p: "не json"
    hi = answer_relevancy([q], ["Медный всадник стоит на Сенатской площади."], on_topic, embed_fn)[0]
    lo = answer_relevancy([q], ["..."], off_topic, embed_fn)[0]
    assert hi > 0.7 > lo + 0.3
    assert answer_relevancy([q], ["не знаю"], dunno, embed_fn)[0] == 0.0
    assert np.isnan(answer_relevancy([q], ["x"], garbage, embed_fn)[0])


def test_answer_relevancy_averages_over_n_generated_questions():
    q = "Какой собор в Москве самый красочный?"
    fn = lambda p: js({"questions": [q, "Совсем другая тема про рыбалку"], "noncommittal": 0})
    s, d = answer_relevancy([q], ["a"], fn, embed_fn, return_details=True)
    assert 0.3 < s[0] < 0.7 and len(d[0]["generated_questions"]) == 2


def _judge(prompt: str) -> str:
    if "простые утверждения" in prompt:
        return js({"statements": ["A", "B", "C", "D"]})
    if "Следует ли утверждение" in prompt:
        return js({"verdict": 0 if "Утверждение: C" in prompt else 1})
    if "Полезен ли фрагмент" in prompt:
        return js({"verdict": 0 if "Фрагмент контекста: c2" in prompt else 1})
    if "Можно ли вывести" in prompt:
        return "1"          # bare number must be accepted too
    return ""


def test_faithfulness_precision_recall_formulas():
    assert faithfulness(["q"], ["a"], [["ctx"]], _judge)[0] == 0.75
    # relevance [1,0,1] -> precision@1=1, precision@3=2/3 -> AP = (1 + 2/3) / 2
    assert abs(context_precision(["q"], ["gt"], [["c1", "c2", "c3"]], _judge)[0] - (1 + 2 / 3) / 2) < 1e-9
    assert context_recall(["Один факт. Второй факт."], [["ctx"]], _judge)[0] == 1.0
    assert np.isnan(faithfulness(["q"], ["a"], [["c"]], lambda p: "мусор")[0])
    assert np.isnan(context_precision(["q"], ["gt"], [["c1"]], lambda p: "мусор")[0])


def test_refusals_are_noncommittal_even_if_judge_misses_them():
    from touristrag.evaluation import is_refusal
    on_topic = lambda p: js({"questions": ["Где находится Медный всадник?"], "noncommittal": 0})
    assert is_refusal("Извините, но в контексте нет информации.") and not is_refusal("Памятник стоит на площади.")
    s = answer_relevancy(["Где находится Медный всадник?"], ["Извините, но в контексте нет информации."], on_topic, embed_fn)
    assert s[0] == 0.0
    assert np.isnan(faithfulness(["q"], ["Извините, нет информации."], [["c"]], _judge)[0])


def test_answer_relevancy_recovers_questions_from_plain_text():
    q = "Где находится Медный всадник?"
    plain = lambda p: "1. Где находится Медный всадник?\n2. Где стоит Медный всадник?"
    assert answer_relevancy([q], ["Он на Сенатской площади."], plain, embed_fn)[0] > 0.7


def test_retrieval_metrics():
    m = retrieval_metrics(["a", "b", "c"], [["a", "x"], ["x", "b"], ["x", "y"]], ks=(1, 2))
    assert m["hit@1"] == 1 / 3 and m["hit@2"] == 2 / 3 and abs(m["mrr"] - (1 + 0.5 + 0) / 3) < 1e-9


def test_data_preprocess_end_to_end_with_fake_llm():
    places = places_frame()
    places = places[~places["name"].isin(["Мем про кота", "Скриншот"])]

    def gen(prompt):
        if "НЕ называя" in prompt:
            return js({"question": "Какой храм построен при Иване Грозном?", "answer": "Собор Василия Блаженного."})
        return js({"question": "Расскажи про Красная площадь", "answer": "Главная площадь Москвы."})

    ds = data_preprocess(places, gen, n=9, blind_ratio=0.5, seed=0)
    assert len(ds) == 9 and {"question", "ground_truths", "answer", "contexts", "place_id", "qtype"} <= set(ds.columns)
    assert isinstance(ds["ground_truths"].iloc[0], list)
    tpl = data_preprocess(places, None, n=3)
    assert all(q.startswith("Расскажи про") for q in tpl["question"])  # template fallback


def test_run_rag_and_summaries():
    rag, places = make_rag()
    ds = data_preprocess(places, None, n=8, seed=3)
    out = run_rag_on_dataset(ds, rag, show_progress=False)
    assert (out["answer"] == "ответ").all() and out["contexts"].map(len).between(1, 3).all()
    res = evaluate_retrieval(lambda q: rag.rank_places(q), ds)
    assert res["all"]["hit@5"] >= 0.75      # named template questions are easy
    s = summarize({"m": np.array([1.0, np.nan, 0.5])})
    assert s.loc["m", "mean"] == 0.75 and s.loc["m", "n_valid"] == 2
