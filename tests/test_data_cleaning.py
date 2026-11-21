import numpy as np
from helpers import places_frame, raw_frame, std_frame

from touristrag.chunking import chunk_text, make_chunks, split_sentences
from touristrag.cleaning import flag_outliers, rule_flags, stratified_sample, tfidf_terms
from touristrag.data import build_places, detect_schema, select_description
from touristrag.utils import extract_json


def test_detect_schema_resolves_caption_before_description():
    s = detect_schema(raw_frame().columns)
    assert s["name"] == "Name" and s["wikidata_id"] == "wikidata" and s["city"] == "City"
    assert s["caption"] == "en_txt" and s["description"] == "description"
    assert s["lon"] == "longitude" and s["lat"] == "latitude"


def test_detect_schema_missing_required_raises_with_hint():
    try:
        detect_schema(["foo", "bar"])
    except KeyError as e:
        assert "column_overrides" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_places_are_deduplicated_and_strategies_differ():
    df = std_frame()
    places = build_places(df, "concat")
    assert len(places) == df["wikidata_id"].nunique() == 11
    isaac = places[places["name"] == "Исаакиевский собор"].iloc[0]
    assert isaac["n_records"] == 2 and "Монферрана" in isaac["description"] and "классицизма" in isaac["description"]
    longest = build_places(df, "longest")
    assert len(longest[longest["name"] == "Исаакиевский собор"].iloc[0]["description"]) < len(isaac["description"])


def test_select_description_dedupes_near_duplicates():
    from touristrag.data import _unique_texts
    kept = _unique_texts(["Большой старый собор в центре города.", "большой старый собор в центре города", "Совсем другое описание места."], 5)
    assert len(kept) == 2
    assert select_description([], "concat") == ""


def test_chunking_respects_budget_and_adds_header():
    text = " ".join(f"Предложение номер {i} про собор." for i in range(30))
    chunks = chunk_text(text, lambda s: len(s.split()), max_tokens=20, overlap_sentences=1)
    assert len(chunks) > 3 and all(len(c.split()) <= 20 for c in chunks)
    assert chunks[0].split(". ")[-1] != "" and split_sentences("Раз. Два! Три?") == ["Раз.", "Два!", "Три?"]
    ch = make_chunks(places_frame())
    assert all(c["text"].startswith(c["name"]) for c in ch)
    assert {"description", "caption"} <= {c["chunk_type"] for c in ch}


def test_rules_flag_junk_but_not_landmarks():
    places = places_frame()
    flags = dict(zip(places["name"], rule_flags(places)))
    assert flags["Мем про кота"] and flags["Скриншот"]
    assert flags["Исаакиевский собор"] == "" and flags["Красная площадь"] == ""


def test_flag_outliers_with_embedder_keeps_columns():
    from touristrag.retrieval import HashingEmbedder
    out = flag_outliers(places_frame(), HashingEmbedder())
    assert {"rule_flag", "tfidf_atyp", "zs_margin", "junk_score", "is_junk"} <= set(out.columns)
    assert out.set_index("name").loc["Мем про кота", "is_junk"]
    assert not out.set_index("name").loc["Медный всадник", "is_junk"]


def test_tfidf_terms_shapes():
    t = tfidf_terms(places_frame()["description"].tolist(), top_n=5, min_df=1)
    assert len(t["frequent"]) == 5 and {"term", "df", "idf"} <= set(t["rare"].columns)


def test_stratified_sample_is_distinct_and_spread():
    places = places_frame()
    s = stratified_sample(places, n=6, by="city", seed=1)
    assert len(s) == 6 and s["place_id"].is_unique
    assert s["city"].value_counts().min() >= 2   # 3 cities, round-robin -> 2 each
    assert stratified_sample(places, n=100)["place_id"].nunique() == len(places)


def test_extract_json_variants():
    assert extract_json('Вот ответ: ```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('[{"x": 1}]') == [{"x": 1}]
    assert extract_json("нет json") is None
