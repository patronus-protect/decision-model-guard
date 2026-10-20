"""Wolf Defender Jev (patronus-studio/wolf-defender-jev) as a decision model, batched by a token budget. PyTorch by default;
`onnx=` runs one of the four ONNX variants of the repository on ONNX Runtime instead (no torch)."""
from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence

from ..util import pick_device, require

REPO = "patronus-studio/wolf-defender-jev"


def _load_class(folder: str):
    """Import WolfDefenderJev from the model folder (the repo ships its own model code, wolf_defender_jev.py)."""
    path = os.path.join(folder, "wolf_defender_jev.py")
    spec = importlib.util.spec_from_file_location("wolf_defender_jev", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"wolf_defender_jev.py not found in {folder}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wolf_defender_jev"] = mod
    spec.loader.exec_module(mod)
    return mod.WolfDefenderJev


class WolfDefenderJev:
    """Decision model adapter for Wolf Defender Jev.

    Batches are cut by a token budget, not a fixed size: padded tokens per batch <= `max_tokens` and batch x length^2 <=
    `max_sq`, because attention memory grows with the square of the length. A long scene runs alone. Option vectors are
    cached per option list."""

    name = "wolf-defender-jev"

    def __init__(self, model, max_tokens: int = 8192, max_sq: int = 8_000_000):
        self.model, self.max_tokens, self.max_sq = model, max_tokens, max_sq
        self._cache: dict[tuple[str, ...], object] = {}
        self.context_tokens = model.max_case                     # longer scenes keep the first third and the last two thirds

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    @classmethod
    def from_pretrained(cls, repo_or_path: str = REPO, device: str | None = None, revision: str | None = None, onnx: str | None = None,
                        **kw):
        """PyTorch model, or with `onnx` ("fp32", "fp16", "mixed", "int8_int4_embeddings" or a path to an .onnx file) the
        ONNX Runtime adapter WolfDefenderJevONNX (keyword arguments: threads, providers, max_tokens, max_sq)."""
        if onnx is not None:
            return WolfDefenderJevONNX.from_pretrained(repo_or_path, onnx, revision=revision, **kw)
        try:                                                  # before downloading 600 MB of PyTorch weights
            require("torch", "hf")
            require("transformers", "hf")
        except ImportError as e:
            raise ImportError(f"{e}, or run the model on ONNX Runtime: WolfDefenderJev.from_pretrained(onnx='int8_int4_embeddings')") from None
        if os.path.isdir(repo_or_path):
            folder = repo_or_path
        else:
            hub = require("huggingface_hub", "hf")
            # PyTorch needs weights, configs, tokenizer and model code; skip the ONNX graphs (1.3 GB), figures and eval files
            folder = hub.snapshot_download(repo_or_path, revision=revision, ignore_patterns=["onnx/*", "assets/*", "eval/*", "*.md"])
        model = _load_class(folder).from_pretrained(folder, device=pick_device(device))
        return cls(model, **kw)

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        import torch
        m = self.model
        tok = m.tokenizer
        bos, eos = tok.bos_token_id, tok.eos_token_id
        CASE, QUESTION = 7, 8
        key = tuple(options)
        with torch.no_grad():
            if key not in self._cache:
                self._cache[key] = m.encode_options(list(options))
            ov = self._cache[key]
            q = m._ids(question, m.max_q)
            seqs, qpos = [], []
            for sc in scenes:
                c = m._ids(sc, m.max_case, keep_tail=True)
                seqs.append([bos, CASE] + c + [eos, QUESTION] + q + [eos])
                qpos.append(len(c) + 3)
            order = sorted(range(len(scenes)), key=lambda i: len(seqs[i]))
            res: dict[int, list[float]] = {}
            k = 0
            while k < len(order):
                j = k + 1
                while j < len(order):
                    L = len(seqs[order[j]])
                    if (j - k + 1) * L > self.max_tokens or (j - k + 1) * L * L > self.max_sq:
                        break
                    j += 1
                idx = order[k:j]
                s = self._scene_vectors([seqs[i] for i in idx], [qpos[i] for i in idx])
                logits = m.scorer(s, ov[None].expand(len(idx), -1, -1)).float()
                for i, p in zip(idx, logits.softmax(-1).tolist(), strict=True):
                    res[i] = p
                k = j
        return [res[i] for i in range(len(scenes))]

    def _scene_vectors(self, seqs, qpos):
        import torch
        m, tok = self.model, self.model.tokenizer
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool)
        for i, sq in enumerate(seqs):
            ids[i, : len(sq)] = torch.tensor(sq)
            mask[i, : len(sq)] = True
        ids, mask, qp = ids.to(m.device), mask.to(m.device), torch.tensor(qpos, device=m.device)
        h = m.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        rows = torch.arange(len(seqs), device=m.device)
        pool = mask.clone()
        pool[:, 1] = False
        pool[rows, qp] = False
        cls_q = torch.cat([h[:, 1], h[rows, qp]], -1)
        return m.state_head(m.readout(h, pool, qp) + m.o_tok(cls_q))


class WolfDefenderJevONNX:
    """Wolf Defender Jev on ONNX Runtime. One graph scores a batch of scenes against the options:
    input_ids / attention_mask of `<bos> [CASE] scene <eos> [QUESTION] question <eos>`, question_pos, and
    option_input_ids / option_attention_mask of `<bos> [OPTION] option <eos>` -> logits [batch, options].
    Truncation (first third and last two thirds of a long scene) and batching follow the PyTorch adapter."""

    name = "wolf-defender-jev-onnx"
    INPUTS = ["input_ids", "attention_mask", "question_pos", "option_input_ids", "option_attention_mask"]

    def __init__(self, session, tokenizer, bos: int, eos: int, max_case_tokens: int = 6144, max_question_tokens: int = 128,
                 max_option_tokens: int = 96, max_tokens: int = 8192, max_sq: int = 8_000_000):
        names = [i.name for i in session.get_inputs()]
        if names != self.INPUTS:
            raise ValueError(f"not a Wolf Defender Jev graph: inputs {names}")
        self.session, self.tok, self.bos, self.eos = session, tokenizer, bos, eos
        self.max_case, self.max_q, self.max_opt = max_case_tokens, max_question_tokens, max_option_tokens
        self.max_tokens, self.max_sq = max_tokens, max_sq
        self.context_tokens = max_case_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    @classmethod
    def from_pretrained(cls, repo_or_path: str = REPO, variant: str = "int8_int4_embeddings", revision: str | None = None,
                        threads: int | None = None, providers=None, **kw) -> WolfDefenderJevONNX:
        from .._onnx import read_json, resolve, session, tokenizer
        folder, graph = resolve(repo_or_path, variant, files=("config.json", "tokenizer.json", "tokenizer_config.json"), revision=revision)
        cfg, tcfg = read_json(folder / "config.json"), read_json(folder / "tokenizer_config.json")
        tok = tokenizer(folder)

        def special(key: str, default: str) -> int:
            v = tcfg.get(key) or default
            return tok.token_to_id(v.get("content") if isinstance(v, dict) else v)

        return cls(session(graph, threads, providers), tok, special("bos_token", "<bos>"), special("eos_token", "<eos>"),
                   cfg.get("max_case_tokens", 6144), cfg.get("max_question_tokens", 128), cfg.get("max_option_tokens", 96), **kw)

    def _ids(self, text: str, limit: int, keep_tail: bool = False) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if len(ids) > limit:
            ids = (ids[: limit // 3] + ids[-(limit - limit // 3):]) if keep_tail else ids[:limit]
        return ids

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        import numpy as np

        from .._onnx import pad, softmax
        if len(options) < 2:
            raise ValueError("a decision needs at least two options")
        q = self._ids(question, self.max_q)
        oids, omask = pad([[self.bos, 9] + self._ids(o, self.max_opt) + [self.eos] for o in options])
        seqs, qpos = [], []
        for sc in scenes:
            c = self._ids(sc, self.max_case, keep_tail=True)
            seqs.append([self.bos, 7] + c + [self.eos, 8] + q + [self.eos])
            qpos.append(len(c) + 3)
        order = sorted(range(len(scenes)), key=lambda i: len(seqs[i]))
        res: dict[int, list[float]] = {}
        k = 0
        while k < len(order):
            j = k + 1
            while j < len(order):
                L = len(seqs[order[j]])
                if (j - k + 1) * L > self.max_tokens or (j - k + 1) * L * L > self.max_sq:
                    break
                j += 1
            idx = order[k:j]
            ids, mask = pad([seqs[i] for i in idx])
            logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "question_pos": np.array([qpos[i] for i in idx], dtype=np.int64),
                                             "option_input_ids": oids, "option_attention_mask": omask})[0]
            for i, p in zip(idx, softmax(logits).tolist(), strict=True):
                res[i] = p
            k = j
        return [res[i] for i in range(len(scenes))]
