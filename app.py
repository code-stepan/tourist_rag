"""Gradio demo:  python app.py   (run `python -m touristrag.cli build` first)."""
import gradio as gr

from touristrag.config import Config
from touristrag.data import decode_image
from touristrag.pipeline import load_rag

rag, places = load_rag(Config())
by_id = places.set_index("place_id")


def ask(question: str):
    res = rag.answer(question)
    gallery, rows = [], []
    for h in res["hits"]:
        row = by_id.loc[h["place_id"]]
        if row["image"]:
            gallery.append((decode_image(row["image"]), f"{row['name']} ({row['city']})"))
        rows.append([row["name"], row["city"], round(float(h.get("rerank_score", h["score"])), 3)])
    return res["answer"], gallery, rows


with gr.Blocks(title="Tourist RAG") as demo:
    gr.Markdown("## 🗺️ Туристический RAG\nСпросите о достопримечательности — ответ строится по найденным описаниям.")
    q = gr.Textbox(label="Вопрос", placeholder="Какой собор в Петербурге стоит на месте…?")
    btn = gr.Button("Спросить", variant="primary")
    ans = gr.Markdown()
    gal = gr.Gallery(label="Найденные места", columns=4, height=260)
    tbl = gr.Dataframe(headers=["Название", "Город", "Скор"], label="Источники")
    btn.click(ask, q, [ans, gal, tbl]); q.submit(ask, q, [ans, gal, tbl])

if __name__ == "__main__":
    demo.launch()
