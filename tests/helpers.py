import json

import numpy as np
import pandas as pd

RAW = [
    # name, wikidata, city, lon, lat, description, caption
    ("Исаакиевский собор", "Q1", "Санкт-Петербург", 30.30, 59.93, "Крупнейший православный храм Санкт-Петербурга с золотым куполом. Построен по проекту Огюста Монферрана.", "a large cathedral with a golden dome"),
    ("Исаакиевский собор", "Q1", "Санкт-Петербург", 30.30, 59.93, "Собор в стиле классицизма, четвёртый по высоте купольный храм мира.", "a church with columns"),
    ("Медный всадник", "Q2", "Санкт-Петербург", 30.30, 59.94, "Памятник Петру I на Сенатской площади работы Этьена Фальконе.", "a bronze statue of a man on a horse"),
    ("Петропавловская крепость", "Q3", "Санкт-Петербург", 30.31, 59.95, "Историческое ядро города на Заячьем острове, заложена в 1703 году.", "a fortress by the river"),
    ("Красная площадь", "Q4", "Москва", 37.62, 55.75, "Главная площадь Москвы между Кремлём и Китай-городом.", "a large square with a cathedral"),
    ("Собор Василия Блаженного", "Q5", "Москва", 37.62, 55.75, "Православный храм на Красной площади с разноцветными куполами, построен при Иване Грозном.", "a colorful cathedral with domes"),
    ("Большой театр", "Q6", "Москва", 37.62, 55.76, "Исторический театр оперы и балета с колоннадой и квадригой Аполлона.", "a theater building with columns"),
    ("Казанский кремль", "Q7", "Казань", 49.10, 55.80, "Историко-архитектурный комплекс с мечетью Кул-Шариф и башней Сююмбике.", "a white fortress with a mosque"),
    ("Башня Сююмбике", "Q8", "Казань", 49.10, 55.80, "Падающая башня в Казанском кремле, символ города.", "a leaning tower"),
    ("Мечеть Кул-Шариф", "Q9", "Казань", 49.10, 55.80, "Главная джума-мечеть Татарстана с голубыми минаретами.", "a mosque with blue minarets"),
    ("Мем про кота", "Q10", "Москва", None, None, "Смешная картинка из интернета.", "a meme of a cat with text"),
    ("Скриншот", "Q11", "Казань", None, None, "Снимок экрана телефона.", "a screenshot of a cell phone"),
]


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(RAW, columns=["Name", "wikidata", "City", "longitude", "latitude", "description", "en_txt"]).assign(image="")


def std_frame():
    from touristrag.data import detect_schema, standardize

    raw = raw_frame()
    return standardize(raw, detect_schema(raw.columns))


def places_frame():
    from touristrag.data import build_places

    return build_places(std_frame())


class FakeLLM:
    """Scripted chat model: reply_fn(prompt_text) -> str."""

    def __init__(self, reply_fn):
        self.reply_fn, self.calls = reply_fn, []

    def chat(self, messages, **kw):
        text = messages[-1]["content"]
        self.calls.append(text)
        return self.reply_fn(text)

    def __call__(self, prompt, system=None, **kw):
        self.calls.append(prompt)
        return self.reply_fn(prompt)


def make_rag(cfg=None, llm=None):
    from touristrag.config import Config
    from touristrag.pipeline import TouristRAG, build_index
    from touristrag.retrieval import HashingEmbedder, HybridRetriever, Reranker

    cfg = cfg or Config(vector_backend="numpy", rerank_backend="none", k_context=3)
    places = places_frame()
    places = places[~places["name"].isin(["Мем про кота", "Скриншот"])].reset_index(drop=True)
    emb = HashingEmbedder()
    chunks, store, _ = build_index(places, emb, cfg)
    retriever = HybridRetriever(emb, store, chunks)
    return TouristRAG(retriever, Reranker("none"), llm or FakeLLM(lambda p: "ответ"), cfg), places


def js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
