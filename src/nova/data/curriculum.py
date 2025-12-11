## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Dict, Iterator, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from nova.data.text_stream import text_to_hypergraph


class CurriculumLoader:
    def __init__(
        self,
        max_seq_len: int = 128,
        cosmos_dataset: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus",
        stack_dataset: str = "bigcode/the-stack-smol",
        alpaca_dataset: str = "bacalhau-project/Turkish-Alpaca",
        cot_dataset: Optional[str] = None,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        self.sources = {}
        self.iterators = {}

        self.sources["turkish_corpus"] = self._load_stream(dataset=cosmos_dataset, split="train")
        self.sources["python_code"] = self._load_stream(dataset=stack_dataset, split="train")
        self.sources["turkish_instructions"] = self._load_stream(dataset=alpaca_dataset, split="train")

        if cot_dataset is not None:
            self.sources["turkish_cot"] = self._load_stream(dataset=cot_dataset, split="train")

        for k, ds in self.sources.items():
            self.iterators[k] = iter(ds)

        self.keys = list(self.sources.keys())
        self.ratios = np.ones(len(self.keys), dtype=np.float32) / max(1, len(self.keys))

    def set_ratios(self, ratios: Dict[str, float]):
        weights = []
        for k in self.keys:
            w = ratios.get(k, 0.0)
            weights.append(max(0.0, float(w)))
        w = np.array(weights, dtype=np.float32)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        self.ratios = w

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng()
        while True:
            idx = rng.choice(len(self.keys), p=self.ratios)
            key = self.keys[idx]
            example = None
            try:
                example = next(self.iterators[key])
            except StopIteration:
                self.iterators[key] = iter(self.sources[key])
                example = next(self.iterators[key])

            text = self._extract_text(key, example)
            if not isinstance(text, str) or len(text) == 0:
                continue

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="np",
            )
            ids = encoded["input_ids"][0].tolist()
            x, H, y = text_to_hypergraph(ids, self.max_seq_len)
            yield x, H, y

    def _load_stream(self, dataset: str, split: str):
        try:
            return load_dataset(dataset, split=split, streaming=True)
        except Exception:
            return load_dataset(dataset, split=split, streaming=True)

    @staticmethod
    def _extract_text(source_key: str, example: Dict) -> str:
        if source_key == "python_code":
            for f in ("content", "code", "text"):
                v = example.get(f)
                if isinstance(v, str) and v:
                    return v
            return ""
        if source_key in ("turkish_corpus", "turkish_cot"):
            for f in ("text", "content"):
                v = example.get(f)
                if isinstance(v, str) and v:
                    return v
            return ""
        if source_key == "turkish_instructions":
            instr = example.get("instruction") or example.get("prompt") or ""
            inp = example.get("input") or ""
            out = example.get("output") or example.get("response") or ""
            parts = []
            if instr:
                parts.append(f"Soru: {instr}")
            if inp:
                parts.append(f"Girdi: {inp}")
            if out:
                parts.append(f"Çıkış: {out}")
            return "\n".join(parts)
        return example.get("text", "")

