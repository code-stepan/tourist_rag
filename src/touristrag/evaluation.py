"""Evaluation: synthetic QA set, RAGAS-style LLM-judge metrics (own implementation) and retrieval metrics.

All LLM-based metrics take plain callables so they are model-agnostic and unit-testable:
    llm_fn(prompt: str) -> str            embed_fn(list[str]) -> np.ndarray (n, d)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .chunking import split_sentences
from .cleaning import stratified_sample
from .utils import extract_json, progress

# =========================================================================== dataset
QA_PROMPT = """Ты помогаешь составлять тестовые вопросы для туристического справочника.
Название: {name}
Город: {city}
Описание: {description}
Как выглядит на фотографии (подпись на английском): {caption}

{style}
Вопрос должен быть конкретным и самодостаточным (не начинай его со слов «Этот объект»/«Это место»).
Ответ должен опираться ТОЛЬКО на описание и подпись к фото и занимать 1–3 предложения.
Верни СТРОГО JSON без пояснений: {{"question": "...", "answer": "..."}}"""

STYLE_NAMED = "Задай естественный вопрос туриста об этой достопримечательности (название можно упоминать)."
STYLE_BLIND = ("Задай вопрос туриста, НЕ называя саму достопримечательность: опиши объект через его тип, внешний вид "
               "или факты из описания так, чтобы правильный ответ был однозначным.")


def _named(name: str, question: str) -> bool:
    return name.lower().strip() in question.lower()


def data_preprocess(places: pd.DataFrame, llm_fn=None, n: int = 100, blind_ratio: float = 0.5,
                    seed: int = 42, by: str = "city") -> pd.DataFrame:
    """Build the RAGAS-format evaluation frame from clean, stratified places.

    Columns: question, ground_truth(s), answer, contexts (filled later by `run_rag_on_dataset`)
    plus `place_id` (gold landmark for retrieval metrics) and `qtype` (named | blind).
    Without an `llm_fn` falls back to template questions. Questions are LLM-generated from the
    description (cookbook-style synthetic QA); half are "blind" so retrieval can't just match the name.
    """
    sample = stratified_sample(places, n=n, by=by, seed=seed)
    rng = np.random.default_rng(seed)
    blind_mask = rng.random(len(sample)) < blind_ratio
    rows, failures = [], 0
    for r, blind in zip(progress(list(sample.itertuples(index=False)), desc="generating QA"), blind_mask):
        question = answer = None
        if llm_fn is not None:
            prompt = QA_PROMPT.format(name=r.name, city=r.city or "не указан", description=r.description or "—",
                                      caption=r.caption or "—", style=STYLE_BLIND if blind else STYLE_NAMED)
            parsed = extract_json(llm_fn(prompt))
            if isinstance(parsed, dict) and parsed.get("question") and parsed.get("answer"):
                question, answer = str(parsed["question"]).strip(), str(parsed["answer"]).strip()
        if not question:
            failures += 1
            question, answer = f"Расскажи про {r.name}" + (f" ({r.city})" if r.city else ""), (r.description or r.caption)
        rows.append({"place_id": r.place_id, "name": r.name, "city": r.city,
                     "qtype": "named" if _named(r.name, question) else "blind",
                     "question": question, "ground_truth": answer, "ground_truths": [answer],
                     "answer": "", "contexts": []})
    if failures:
        print(f"[data_preprocess] {failures}/{len(rows)} generations fell back to templates")
    return pd.DataFrame(rows)


def run_rag_on_dataset(df: pd.DataFrame, rag, show_progress: bool = True) -> pd.DataFrame:
    """Fill `answer`, `contexts`, `retrieved_place_ids` by running the full RAG on every question."""
    out = df.copy()
    answers, ctxs, retrieved = [], [], []
    it = progress(list(out["question"]), desc="RAG") if show_progress else out["question"]
    for q in it:
        res = rag.answer(q)
        answers.append(res["answer"])
        ctxs.append(res["contexts"])
        retrieved.append(res["place_ids"])
    out["answer"], out["contexts"], out["retrieved_place_ids"] = answers, ctxs, retrieved
    return out


# =========================================================================== helpers
_YES = {"1", "да", "yes", "true"}
REFUSAL_RE = re.compile(r"извините|нет информации|недостаточно (информации|данных)|нет данных|"
                        r"не указан\w* в|отсутству\w+ информаци|не (могу|удалось) (ответить|найти)", re.I)


def is_refusal(answer: str) -> bool:
    """Rule-based 'noncommittal' detector: small judge LLMs often miss polite refusals."""
    return bool(REFUSAL_RE.search(answer or ""))


def _call(llm_fn, prompt: str) -> str:
    try:  # a failed judge call must not kill a 100-sample run
        return llm_fn(prompt) or ""
    except Exception:
        return ""


def _judge(llm_fn, prompt: str):
    return extract_json(_call(llm_fn, prompt))


def _yesno(llm_fn, prompt: str):
    """1 / 0 / None. Accepts {"verdict": 1}, a bare 1/0, or да/нет anywhere in a short reply."""
    raw = _call(llm_fn, prompt)
    parsed = extract_json(raw)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, (int, float, bool)):
                return int(bool(v))
            if isinstance(v, str) and v.strip().lower() in _YES | {"0", "нет", "no", "false"}:
                return int(v.strip().lower() in _YES)
    m = re.search(r"\b(1|0|да|нет|yes|no)\b", raw.lower())
    return None if not m else int(m.group(1) in _YES)


def _fallback_questions(text: str) -> list[str]:
    return [q.strip(" -•*\"«»0123456789.)") + "?" for q in re.findall(r"[^\n?]{8,}\?", text or "")]


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


# =========================================================================== RAGAS: answer relevancy
AR_PROMPT = """Ниже дан ответ на некий вопрос. Сгенерируй {n} РАЗНЫХ вопроса, на которые этот ответ является прямым ответом.
Также определи noncommittal: 1, если ответ уклончивый или ничего не утверждает (например, «не знаю», «в контексте нет информации»), иначе 0.

Ответ: {answer}

Верни СТРОГО JSON: {{"questions": ["...", "..."], "noncommittal": 0}}"""


def answer_relevancy(questions, answers, llm_fn, embed_fn, n_gen: int = 3, return_details: bool = False):
    """RAGAS answer relevancy, re-implemented.

        AR_i = (1/N) * sum_j cos(E(q_gen_ij), E(q_i)),   N = n_gen
    where q_gen_ij are questions an LLM reconstructs *from the answer alone*. A noncommittal answer
    scores 0; noncommittal = judge flag OR rule-based refusal detector (`is_refusal`). If neither JSON
    nor plain-text questions can be recovered the score is NaN (not 0): a judge failure must not look
    like a bad answer. Aggregate with np.nanmean and report n_valid.
    """
    scores, details = [], []
    for q, a in zip(questions, answers):
        raw = _call(llm_fn, AR_PROMPT.format(n=n_gen, answer=a))
        parsed = extract_json(raw)
        gens, noncommittal = [], 0
        if isinstance(parsed, dict):
            gens = [str(x).strip() for x in parsed.get("questions", []) if str(x).strip()][:n_gen]
            noncommittal = int(bool(parsed.get("noncommittal", 0)))
        if not gens:
            gens = _fallback_questions(raw)[:n_gen]
        noncommittal = int(noncommittal or is_refusal(a))
        if noncommittal:
            score = 0.0
        elif not gens:
            score = float("nan")
        else:
            emb = _unit(embed_fn([q] + gens))
            score = float((emb[1:] @ emb[0]).mean())
        scores.append(score)
        details.append({"generated_questions": gens, "noncommittal": noncommittal, "score": score})
    scores = np.array(scores)
    return (scores, details) if return_details else scores


# =========================================================================== RAGAS: faithfulness
STATEMENTS_PROMPT = """Вопрос: {question}
Ответ: {answer}

Разбей ответ на отдельные простые утверждения, каждое понятно без контекста.
Верни СТРОГО JSON: {{"statements": ["...", "..."]}}"""

VERDICT_PROMPT = """Контекст:
{context}

Следует ли утверждение из контекста? Ответь 1 (да) или 0 (нет).
Утверждение: {statement}
Верни СТРОГО JSON: {{"verdict": 1}}"""


def faithfulness(questions, answers, contexts, llm_fn) -> np.ndarray:
    """|claims in the answer supported by the context| / |all claims in the answer| (one judge call per claim)."""
    out = []
    for q, a, ctx in zip(questions, answers, contexts):
        if is_refusal(a):                       # a refusal makes no claims: nothing to be unfaithful about
            out.append(float("nan"))
            continue
        st = _judge(llm_fn, STATEMENTS_PROMPT.format(question=q, answer=a))
        statements = [str(s) for s in st.get("statements", [])] if isinstance(st, dict) else []
        if not statements:
            out.append(float("nan"))
            continue
        ctx_text = "\n\n".join(ctx)
        verdicts = [_yesno(llm_fn, VERDICT_PROMPT.format(context=ctx_text, statement=s)) for s in statements]
        verdicts = [v for v in verdicts if v is not None]
        out.append(float(np.mean(verdicts)) if verdicts else float("nan"))
    return np.array(out)


# =========================================================================== retrieval-side metrics (LLM judge)
PRECISION_PROMPT = """Вопрос: {question}
Эталонный ответ: {truth}

Фрагмент контекста: {chunk}

Полезен ли фрагмент, чтобы получить эталонный ответ? Ответь 1 (да) или 0 (нет).
Верни СТРОГО JSON: {{"verdict": 1}}"""


def context_precision(questions, ground_truths, contexts, llm_fn) -> np.ndarray:
    """RAGAS context precision@K = sum_k(precision@k * rel_k) / sum_k rel_k  (rewards relevant chunks ranked first).
    One judge call per retrieved chunk (small LLMs are unreliable at returning aligned JSON lists)."""
    out = []
    for q, gt, ctx in zip(questions, ground_truths, contexts):
        rel = [_yesno(llm_fn, PRECISION_PROMPT.format(question=q, truth=gt, chunk=c)) for c in ctx]
        if not ctx or any(r is None for r in rel):
            out.append(float("nan"))
            continue
        rel = np.array(rel)
        prec_at_k = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        out.append(float((prec_at_k * rel).sum() / rel.sum()) if rel.sum() else 0.0)
    return np.array(out)


RECALL_PROMPT = """Контекст:
{context}

Можно ли вывести предложение из контекста? Ответь 1 (да) или 0 (нет).
Предложение: {sentence}
Верни СТРОГО JSON: {{"verdict": 1}}"""


def context_recall(ground_truths, contexts, llm_fn) -> np.ndarray:
    """|GT sentences attributable to the context| / |GT sentences| (one judge call per sentence)."""
    out = []
    for gt, ctx in zip(ground_truths, contexts):
        sents = split_sentences(gt) or [gt]
        ctx_text = "\n\n".join(ctx)
        flags = [_yesno(llm_fn, RECALL_PROMPT.format(context=ctx_text, sentence=s)) for s in sents]
        flags = [f for f in flags if f is not None]
        out.append(float(np.mean(flags)) if flags else float("nan"))
    return np.array(out)


# =========================================================================== retrieval metrics (no LLM)
def retrieval_metrics(gold_ids, ranked_ids, ks=(1, 3, 5, 10)) -> dict:
    """Hit@k and MRR for single-gold-landmark queries. `ranked_ids`: list of ranked place_id lists."""
    ranks = []
    for gold, ranked in zip(gold_ids, ranked_ids):
        ranks.append(ranked.index(gold) + 1 if gold in ranked else np.inf)
    ranks = np.array(ranks, dtype=float)
    m = {f"hit@{k}": float((ranks <= k).mean()) for k in ks}
    m["mrr"] = float(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).mean())
    return m


def evaluate_retrieval(rank_fn, df: pd.DataFrame, ks=(1, 3, 5, 10)) -> dict:
    """rank_fn(question) -> ranked list of place_ids. Reports metrics overall and by question type."""
    ranked = [rank_fn(q) for q in progress(list(df["question"]), desc="retrieval")]
    res = {"all": retrieval_metrics(df["place_id"].tolist(), ranked, ks)}
    for t, g in df.groupby("qtype"):
        res[t] = retrieval_metrics(g["place_id"].tolist(), [ranked[i] for i in g.index], ks)
    return res


def summarize(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = [{"metric": k, "mean": float(np.nanmean(v)), "n_valid": int(np.isfinite(v).sum()), "n": len(v)}
            for k, v in scores.items()]
    return pd.DataFrame(rows).set_index("metric")
