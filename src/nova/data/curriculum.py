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
            ds1 = load_dataset("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", split="train")
            ds2 = load_dataset("turkish-nlp-suite/BellaTurca", "OzenliDerlem", split="train")
            corpus_ds = interleave_datasets([ds1, ds2], probabilities=[0.5, 0.5], seed=42)
            datasets.append(corpus_ds)
            probabilities.append(ratios["corpus"])

        # Code
        if ratios.get("code", 0) > 0:
            code_ds = load_dataset("bigcode/the-stack-smol", languages=["python"], split="train")
            datasets.append(code_ds)
            probabilities.append(ratios["code"])

        # Instruct
        if ratios.get("instruct", 0) > 0:
            inst_ds = load_dataset("AlicanKiraz0/Turkish-SFT-Dataset-v1.0", split="train")
            datasets.append(inst_ds)
            probabilities.append(ratios["instruct"])

        # CoT
        if ratios.get("cot", 0) > 0:
            cot_ds = load_dataset("bezir/gsm8k-tr", split="test")
            datasets.append(cot_ds)
            probabilities.append(ratios["cot"])

        self.dataset = interleave_datasets(datasets, probabilities=probabilities, seed=42)
        logger.info(f"Epoch {epoch}: Target Ratios = {ratios}")

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
    val_examples = []

    # 1. Wikipedia TR
    ds = load_dataset("mc4", "tr", split=f"train[:1%]", seed=42)
    val_examples.extend(ds.select(range(10000)))

    # 2. OpenSubtitles TR
    ds = load_dataset("open_subtitles", "tr", split="train", verification_mode="no_checks")
    val_examples.extend(ds.select(range(min(5000, len(ds)))))

    # 3. OSCAR TR
    ds = load_dataset("oscar", "unshuffled_deduplicated_tr", split="train")
    val_examples.extend(ds.select(range(min(5000, len(ds)))))

    random.seed(42)
    random.shuffle(val_examples)
    val_examples = val_examples[:10000]   # toplam 10k sabit validation

    class ValidationSet:
        def __iter__(self):
            for example in val_examples:
                text = example.get("text") or example.get("content") or ""
                if len(text) < 50: continue
                encoded = tokenizer(text, max_length=max_seq_len, padding="max_length", return_tensors="np")
                ids = encoded["input_ids"][0].tolist()
                x, H, y = text_to_hypergraph(ids, max_seq_len)
                if len(x) > 100:
                    yield x.astype(np.int32), H.astype(np.float32), y.astype(np.int32)
    
    return ValidationSet()
