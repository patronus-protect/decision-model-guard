"""Local transformer models as decision models: causal LM judges and sequence classifiers."""
from __future__ import annotations

from collections.abc import Sequence

from ..util import normalize, pick_device, require

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CausalLMJudge:
    """Any causal LM (Qwen, Llama, Mistral, Gemma, ...) used as a decision model: the options are listed with letters and
    the next-token probabilities of the letters, renormalised over the options, become the option probabilities."""

    SYSTEM = "You are a decision model. Read the input and answer the question with the letter of exactly one option."

    def __init__(self, model_id: str, device: str | None = None, dtype: str | None = None, max_tokens: int = 2048,
                 system_prompt: str | None = None):
        torch = require("torch", "hf")
        tf = require("transformers", "hf")
        self.device = pick_device(device)
        self.name = model_id
        self.tok = tf.AutoTokenizer.from_pretrained(model_id)
        dtype = dtype or ("float32" if self.device == "cpu" else "bfloat16")   # bf16 is emulated and slow on Apple CPUs
        self.model = tf.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.max_tokens, self.system = max_tokens, system_prompt or self.SYSTEM

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        import torch
        ids = [self.tok.encode(" " + LETTERS[i], add_special_tokens=False)[-1] for i in range(len(options))]
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        out = []
        with torch.no_grad():
            for sc in scenes:
                body = self.tok.decode(self.tok.encode(sc, add_special_tokens=False)[-self.max_tokens:])
                msgs = [{"role": "system", "content": self.system},
                        {"role": "user", "content": f"Input:\n{body}\n\nQuestion: {question}\nOptions:\n{opts}\n\nAnswer with one letter."}]
                prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
                enc = self.tok(prompt, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits[0, -1].float()
                out.append(torch.softmax(logits[ids], -1).tolist())
        return out


class HFClassifier:
    """A Hugging Face sequence classifier as a decision model with fixed options, e.g. Wolf Defender as the decision
    "is this a prompt injection?". `option_labels` maps each option to a label index of the classifier (default: option
    i -> label i). The question is ignored: a classifier answers one fixed question."""

    def __init__(self, model_id: str, option_labels: Sequence[int] | None = None, device: str | None = None,
                 max_length: int = 2048, batch_size: int = 16, subfolder: str | None = None):
        require("torch", "hf")
        tf = require("transformers", "hf")
        self.device = pick_device(device)
        self.name = model_id
        kw = {"subfolder": subfolder} if subfolder else {}
        self.tok = tf.AutoTokenizer.from_pretrained(model_id)
        self.model = tf.AutoModelForSequenceClassification.from_pretrained(model_id, **kw).to(self.device).eval()
        self.option_labels = list(option_labels) if option_labels is not None else None
        self.max_length, self.batch_size = max_length, batch_size

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        import torch
        labels = self.option_labels or list(range(len(options)))
        if len(labels) != len(options):
            raise ValueError("option_labels must name one classifier label per option")
        out = []
        with torch.no_grad():
            for i in range(0, len(scenes), self.batch_size):
                x = self.tok(list(scenes[i: i + self.batch_size]), truncation=True, max_length=self.max_length, padding=True,
                             return_tensors="pt").to(self.device)
                p = torch.softmax(self.model(**x).logits.float(), -1)
                out += [normalize([float(row[j]) for j in labels]) for row in p]
        return out
