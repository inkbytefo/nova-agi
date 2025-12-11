from typing import Iterator, Tuple
import numpy as np
from datasets import interleave_datasets, load_dataset
from nova.data.tokenizer import HypergraphTokenizer
from nova.data.text_stream import text_to_hypergraph
import logging
import random

logger = logging.getLogger(__name__)

class CurriculumLoader:
    def __init__(
        self,
        epoch: int = 0,
        max_seq_len: int = 2048,
        datasets_config: dict = None,
    ):
        self.epoch = epoch
        self.max_seq_len = max_seq_len
        self.tokenizer = HypergraphTokenizer(vocab_size=5000)  # HARD-CODED, garanti

        # Phase ratios
        if epoch < 2:
            ratios = {"corpus": 0.8, "code": 0.2}
        elif epoch < 4:
            ratios = {"corpus": 0.5, "instruct": 0.3, "code": 0.2}
        else:
            ratios = {"corpus": 0.0, "instruct": 0.4, "code": 0.2, "cot": 0.4}

        datasets = []
        probabilities = []

        # Corpus
        if ratios.get("corpus", 0) > 0:
            ds1 = load_dataset("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", split="train", streaming=True)
            ds2 = load_dataset("turkish-nlp-suite/BellaTurca", "OzenliDerlem", split="train", streaming=True)
            corpus_ds = interleave_datasets([ds1, ds2], probabilities=[0.5, 0.5], seed=42, stopping_strategy="all_exhausted")
            datasets.append(corpus_ds)
            probabilities.append(ratios["corpus"])

        # Code
        if ratios.get("code", 0) > 0:
            # Correct usage: Filter by data_dir for specific language in 'the-stack-smol'
            code_ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
            datasets.append(code_ds)
            probabilities.append(ratios["code"])

        # Instruct
        if ratios.get("instruct", 0) > 0:
            inst_ds = load_dataset("AlicanKiraz0/Turkish-SFT-Dataset-v1.0", split="train", streaming=True)
            datasets.append(inst_ds)
            probabilities.append(ratios["instruct"])

        # CoT
        if ratios.get("cot", 0) > 0:
            cot_ds = load_dataset("bezir/gsm8k-tr", split="test", streaming=True)
            datasets.append(cot_ds)
            probabilities.append(ratios["cot"])

        self.dataset = interleave_datasets(datasets, probabilities=probabilities, seed=42, stopping_strategy="all_exhausted")
        logger.info(f"Epoch {epoch}: Target Ratios = {ratios}, Initialized Streaming Datasets.")

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for example in self.dataset:
            text = example.get("text", "") or example.get("content", "") or ""
            if not text or len(text.strip()) < 10:
                continue

            encoded = self.tokenizer(text, max_length=self.max_seq_len, padding="max_length", return_tensors="np")
            ids = encoded["input_ids"][0].tolist()

            x, H, y = text_to_hypergraph(ids, self.max_seq_len)
            if len(x) > 10:  # çok kısa olanları at
                yield x.astype(np.int32), H.astype(np.float32), y.astype(np.int32)

def build_validation_loader(max_seq_len: int = 2048):
    """Profesyonel, sabit validation seti döndürür (her run’da aynı)."""
    tokenizer = HypergraphTokenizer(vocab_size=5000)
    datasets = []

    # 1. Wikipedia TR
    try:
        ds1 = load_dataset("mc4", "tr", split="train", streaming=True).take(5000)
        datasets.append(ds1)
    except: pass

    # 2. OpenSubtitles TR
    try:
        ds2 = load_dataset("open_subtitles", "tr", split="train", streaming=True).take(2500)
        datasets.append(ds2)
    except: pass
    
    # 3. OSCAR TR
    try:
        ds3 = load_dataset("oscar", "unshuffled_deduplicated_tr", split="train", streaming=True).take(2500)
        datasets.append(ds3)
    except: pass

    if not datasets:
        logger.warning("No validation datasets loaded. Using dummy.")
        return []
    
    combined_val = interleave_datasets(datasets, seed=42, stopping_strategy="all_exhausted")

    class ValidationSet:
        def __iter__(self):
            for example in combined_val:
                text = example.get("text") or example.get("content") or ""
                if len(text) < 50: continue
                encoded = tokenizer(text, max_length=max_seq_len, padding="max_length", return_tensors="np")
                ids = encoded["input_ids"][0].tolist()
                x, H, y = text_to_hypergraph(ids, max_seq_len)
                if len(x) > 100:
                    yield x.astype(np.int32), H.astype(np.float32), y.astype(np.int32)
    
    return ValidationSet()
