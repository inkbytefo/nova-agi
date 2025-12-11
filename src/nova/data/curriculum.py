## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets

from nova.data.text_stream import text_to_hypergraph

class CurriculumLoader:
    """
    Simplified Curriculum Loader using HuggingFace's interleave_datasets.
    """
    def __init__(
        self,
        max_seq_len: int = 512,
        cosmos_dataset: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
        stack_dataset: str = "bigcode/the-stack-smol",
        alpaca_dataset: str = "TFLai/Turkish-Alpaca",
        cot_dataset: str = "berhaan/Turkish-gsm8k",
        cot_config: str = "main",
        ratios: dict = None
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        
        if ratios is None:
            # Default ratios: Corpus 50%, Code 10%, Instr 20%, CoT 20%
            ratios = {
                "turkish_corpus": 0.5,
                "python_code": 0.1,
                "turkish_instructions": 0.2,
                "turkish_cot": 0.2
            }
        
        # Load Datasets (Streaming)
        try:
            d_corpus = load_dataset(cosmos_dataset, split="train", streaming=True)
            d_corpus = d_corpus.map(self._extract_corpus)
        except Exception:
            d_corpus = None
            
        try:
            d_code = load_dataset(stack_dataset, data_dir="data/python", split="train", streaming=True)
            d_code = d_code.map(self._extract_code)
        except Exception:
            d_code = None
            
        try:
            d_instr = load_dataset(alpaca_dataset, split="train", streaming=True)
            d_instr = d_instr.map(self._extract_instr)
        except Exception:
            d_instr = None
            
        try:
            d_cot = load_dataset(cot_dataset, cot_config, split="train", streaming=True)
            d_cot = d_cot.map(self._extract_cot)
        except Exception:
            d_cot = None
            
        # Filter valid datasets and probabilities
        datasets = []
        probs = []
        
        if d_corpus:
            datasets.append(d_corpus)
            probs.append(ratios.get("turkish_corpus", 0.0))
        if d_code:
            datasets.append(d_code)
            probs.append(ratios.get("python_code", 0.0))
        if d_instr:
            datasets.append(d_instr)
            probs.append(ratios.get("turkish_instructions", 0.0))
        if d_cot:
            datasets.append(d_cot)
            probs.append(ratios.get("turkish_cot", 0.0))
            
        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Fallback if all 0 or empty
            probs = [1.0 / len(datasets)] * len(datasets)
            
        if datasets:
            self.dataset = interleave_datasets(datasets, probabilities=probs, stopping_strategy="first_exhausted")
        else:
            self.dataset = []
            
    # Extraction Helpers (Standardize to 'text' field)
    @staticmethod
    def _extract_corpus(ex):
        t = ex.get("text") or ex.get("content") or ""
        return {"text": t}
        
    @staticmethod
    def _extract_code(ex):
        t = ex.get("content") or ex.get("code") or ex.get("text") or ""
        return {"text": t}
        
    @staticmethod
    def _extract_instr(ex):
        instr = ex.get("instruction") or ex.get("prompt") or ""
        inp = ex.get("input") or ""
        out = ex.get("output") or ex.get("response") or ""
        parts = []
        if instr: parts.append(f"Soru: {instr}")
        if inp: parts.append(f"Girdi: {inp}")
        if out: parts.append(f"Çıkış: {out}")
        return {"text": "\n".join(parts)}
        
    @staticmethod
    def _extract_cot(ex):
        q = ex.get("question") or ""
        cot = ex.get("cot") or ""
        ans = ex.get("answer") or ""
        # Basic parsing if cot is missing but answer has '####'
        if not cot and isinstance(ans, str) and "####" in ans:
            parts = ans.split("####")
            cot = parts[0].strip()
            ans = "####" + parts[1].strip() if len(parts) > 1 else ans
        
        s = []
        if q: s.append(f"Soru: {q}")
        if cot: s.append(f"Düşünce: {cot}")
        if ans: s.append(f"Cevap: {ans}")
        return {"text": "\n".join(s)}

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for example in self.dataset:
            text = example.get("text", "")
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
            
            if len(x) > 0:
                yield x, H, y
