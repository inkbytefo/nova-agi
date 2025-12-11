## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple, List, Dict, Any
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
import logging

from nova.data.text_stream import text_to_hypergraph

# Configure logging
logger = logging.getLogger(__name__)

class CurriculumLoader:
    """
    Curriculum Loader with progressive ratios and robust dataset handling.
    
    Phases:
    1. Foundation (Epoch < 2): 80% Corpus, 20% Code.
    2. Instruction (2 <= Epoch < 4): 50% Corpus, 30% Instruct, 20% Code.
    3. Reasoning (Epoch >= 4): 40% CoT, 40% Instruct, 20% Code.
    """
    def __init__(
        self,
        epoch: int = 0,
        max_seq_len: int = 512,
        tokenizer_name: str = "dbmdz/bert-base-turkish-cased",
        datasets_config: Dict[str, str] = None
    ):
        self.epoch = epoch
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Default Dataset Configurations
        if datasets_config is None:
            self.datasets_config = {
                # New Sources
                "instruct": "AlicanKiraz0/Turkish-SFT-Dataset-v1.0", 
                "cot": "GoktugP/Turkish-GSM8K-CoT",
                "code": "codeparrot/github-code-clean",
                # Note: 'corpus' is now handled by load_turkish_corpus in dataset.py via config
            }
        else:
            self.datasets_config = datasets_config

        # 1. Determine Ratios based on Epoch
        self.ratios = self._get_ratios(epoch)
        logger.info(f"Epoch {epoch}: Target Ratios = {self.ratios}")

        # 2. Load Datasets safely
        loaded_datasets = []
        probs = []
        
        # Helper to load and append
        def try_load(key, load_fn, weight):
            if weight <= 0:
                return
            try:
                ds = load_fn()
                # Verify accessibility by peeking at one element
                # This ensures we don't crash later in interleave_datasets
                # if the dataset requires auth or is broken.
                try:
                     _ = next(iter(ds.take(1)))
                except Exception as peek_err:
                     logger.warning(f"Dataset {key} loaded but failed to read: {peek_err}")
                     return

                if ds:
                    loaded_datasets.append(ds)
                    probs.append(weight)
                    logger.info(f"Loaded {key} (weight={weight})")
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")

        # Corpus
        # We skip the first 2048 samples to avoid overlap with validation set (if it uses train subset)
        # Corpus
        # Use the specialized loader from dataset.py to mix Cosmos/BellaTurca
        from nova.data.dataset import load_turkish_corpus
        
        # We need to construct a mini-config for the corpus loader
        # Ideally this comes from the main clean config, but we mock it here if needed
        # or we assume self.datasets_config might have it if passed properly
        corpus_config = {
             "corpus_sources": {
                 "cosmos": {"path": "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", "weight": 0.5},
                 "bellaturca": {"path": "turkish-nlp-suite/BellaTurca", "weight": 0.5},
             }
        }
        
        try_load("corpus", 
                 lambda: load_turkish_corpus(corpus_config, split="train", streaming=True).map(self._extract_corpus),
                 self.ratios.get("corpus", 0.0))
        
        # Code
        try_load("code", 
                 lambda: load_dataset(self.datasets_config["code"], data_dir="data/python", split="train", streaming=True).map(self._extract_code),
                 self.ratios.get("code", 0.0))
        
        # Instruct
        try_load("instruct", 
                 lambda: load_dataset(self.datasets_config["instruct"], split="train", streaming=True).map(self._extract_instr),
                 self.ratios.get("instruct", 0.0))
                 
        # CoT
        try_load("cot", 
                 lambda: load_dataset(self.datasets_config["cot"], split="train", streaming=True).map(self._extract_cot),
                 self.ratios.get("cot", 0.0))

        # 3. Fallback Mechanism
        if not loaded_datasets:
            logger.error("All primary datasets failed! Switching to Fallback (mc4-tr).")
            try:
                ds_fallback = load_dataset(self.datasets_config["fallback"], "tr", split="train", streaming=True)
                ds_fallback = ds_fallback.map(self._extract_corpus) # mc4 uses 'text'
                loaded_datasets.append(ds_fallback)
                probs.append(1.0)
            except Exception as e:
                logger.critical(f"Fallback dataset also failed: {e}")
                raise RuntimeError("Could not load any datasets.")

        # 4. Normalize Probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(loaded_datasets)] * len(loaded_datasets)
            
        logger.info(f"Final Probabilities: {probs}")

        # 5. Interleave
        self.dataset = interleave_datasets(
            loaded_datasets, 
            probabilities=probs, 
            stopping_strategy="first_exhausted"
        )

    def _get_ratios(self, epoch: int) -> Dict[str, float]:
        """Returns dataset ratios based on curriculum phase."""
        if epoch < 2:
            # Phase 1: Foundation
            return {"corpus": 0.8, "code": 0.2, "instruct": 0.0, "cot": 0.0}
        elif 2 <= epoch < 4:
            # Phase 2: Instruction
            return {"corpus": 0.5, "code": 0.2, "instruct": 0.3, "cot": 0.0}
        else:
            # Phase 3: Reasoning
            return {"corpus": 0.0, "code": 0.2, "instruct": 0.4, "cot": 0.4}

    # --- Extraction Helpers ---
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
        # Adapting for Turkcell/InstrucTurca or Alpaca style
        instr = ex.get("instruction") or ex.get("talimat") or ""
        inp = ex.get("input") or ex.get("giriş") or ""
        out = ex.get("output") or ex.get("çıktı") or ""
        
        parts = []
        if instr: parts.append(f"Soru: {instr}")
        if inp: parts.append(f"Girdi: {inp}")
        if out: parts.append(f"Çıkış: {out}")
        return {"text": "\n".join(parts)}
        
    @staticmethod
    def _extract_cot(ex):
        # Adapting for Turkish-GSM8K-CoT
        q = ex.get("question") or ex.get("soru") or ""
        cot = ex.get("chain_of_thought") or ex.get("cot") or ex.get("cozum_adimlari") or ""
        ans = ex.get("answer") or ex.get("cevap") or ""
        
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

            try:
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
            except Exception as e:
                # Skip faulty examples
                continue

