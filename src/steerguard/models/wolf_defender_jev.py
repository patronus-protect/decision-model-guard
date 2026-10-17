"""Wolf Defender Jev (patronus-studio/wolf-defender-jev) as a decision model, batched by a token budget."""
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

    @classmethod
    def from_pretrained(cls, repo_or_path: str = REPO, device: str | None = None, revision: str | None = None, **kw) -> WolfDefenderJev:
        if os.path.isdir(repo_or_path):
            folder = repo_or_path
        else:
            hub = require("huggingface_hub", "hf")
            folder = hub.snapshot_download(repo_or_path, revision=revision)
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
