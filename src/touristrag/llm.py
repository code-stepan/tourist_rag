"""Thin chat wrapper over a Hugging Face causal LM (optionally 4-bit quantised)."""
from __future__ import annotations


class HFChat:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = False,
                 max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw: dict = {"device_map": "auto"}
        if load_in_4bit and torch.cuda.is_available():
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()

    def chat(self, messages: list[dict], max_new_tokens: int | None = None, temperature: float = 0.0) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        gen = {"max_new_tokens": max_new_tokens or self.max_new_tokens, "pad_token_id": self.tok.eos_token_id}
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen.update(do_sample=False)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def __call__(self, prompt: str, system: str | None = None, **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)
