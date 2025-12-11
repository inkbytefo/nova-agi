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
        max_seq_len: int = 512,
        cosmos_dataset: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
        stack_dataset: str = "bigcode/the-stack-smol",
        alpaca_dataset: str = "TFLai/Turkish-Alpaca",
        cot_dataset: str = "berhaan/Turkish-gsm8k",
        cot_config: str = "main",
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        self.sources = {}
        self.iterators = {}

        try:
            self.sources["turkish_corpus"] = load_dataset(cosmos_dataset, split="train", streaming=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Turkish Corpus '{cosmos_dataset}': {e}")

        try:
            self.sources["python_code"] = load_dataset(stack_dataset, data_dir="data/python", split="train", streaming=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Python Code '{stack_dataset} (data_dir=data/python)': {e}")

        try:
            self.sources["turkish_instructions"] = load_dataset(alpaca_dataset, split="train", streaming=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Turkish Instructions '{alpaca_dataset}': {e}")

        try:
            self.sources["turkish_cot"] = load_dataset(cot_dataset, cot_config, split="train", streaming=True)
        except Exception:
            self.sources["turkish_cot"] = load_dataset(cot_dataset, split="train", streaming=True)

        # Optional English CoT fallback (not used unless explicitly weighted)
        try:
            self.sources["english_cot"] = load_dataset("openai/gsm8k", split="train", streaming=True)
        except Exception:
            pass

        for k, ds in self.sources.items():
            self.iterators[k] = iter(ds)

        self.keys = list(self.sources.keys())
        self.ratios = np.ones(len(self.keys), dtype=np.float32) / max(1, len(self.keys))

    def set_ratios(self, ratios: Dict[str, float]):
        for k, v in ratios.items():
            if v > 0.0 and k not in self.keys:
                raise ValueError(f"Ratio specifies missing dataset key '{k}' with positive weight {v}. Available: {self.keys}")
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

    # Fallback loaders removed per requirement: use exact datasets only.

    @staticmethod
    def _extract_text(source_key: str, example: Dict) -> str:
        if source_key == "python_code":
            v = example.get("content") or example.get("code") or example.get("text") or ""
            return v if isinstance(v, str) else ""
        if source_key == "turkish_corpus":
            v = example.get("text") or example.get("content") or ""
            return v if isinstance(v, str) else ""
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
        if source_key == "turkish_cot":
            q = example.get("question") or ""
            cot = example.get("cot") or ""
            ans = example.get("answer") or ""
            if not cot and isinstance(ans, str) and "####" in ans:
                parts = ans.split("####")
                cot = parts[0].strip()
                ans = "####" + parts[1].strip() if len(parts) > 1 else ans
            s = []
            if q:
                s.append(f"Soru: {q}")
            if cot:
                s.append(f"Düşünce: {cot}")
            if ans:
                s.append(f"Cevap: {ans}")
            return "\n".join(s)
        if source_key == "english_cot":
            q = example.get("question") or ""
            ans = example.get("answer") or ""
            cot = ""
            if isinstance(ans, str) and "####" in ans:
                parts = ans.split("####")
                cot = parts[0].strip()
                ans = "####" + parts[1].strip() if len(parts) > 1 else ans
            s = []
            if q:
                s.append(f"Soru (EN): {q}")
            if cot:
                s.append(f"Düşünce (EN): {cot}")
            if ans:
                s.append(f"Cevap (EN): {ans}")
            return "\n".join(s)
        return example.get("text", "")
