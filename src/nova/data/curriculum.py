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

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for example in self.dataset:
            text = example.get("text", "") or example.get("content", "") or ""
            if not text or len(text.strip()) < 10:
                continue

            # Tokenize using HDCT (Semantic-Topo)
            ids, edges = self.tokenizer.encode_with_topology(text)
            
            # Truncate if necessary
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
                # In naive truncation, we just pass the ids. edges will be filtered by text_to_hypergraph index check.
            
            # Pad if needed (manual padding for batch consistency if required, but here we yield samples)
            # text_to_hypergraph handles the structural padding logic usually.
            
            x, H_in, H_out, y = text_to_hypergraph(ids, self.max_seq_len, topology_edges=edges)
            if len(x) > 10:  # çok kısa olanları at
                yield x.astype(np.int32), H_in.astype(np.float32), H_out.astype(np.float32), y.astype(np.int32)

def build_validation_loader(max_seq_len: int = 2048):
    tokenizer = HypergraphTokenizer(vocab_size=5000)
    
    # 10k sabit, hızlı, tekrarlanabilir validation seti
    # 10k sabit, hızlı, tekrarlanabilir validation seti
    val_sources = [
        # CulturaX (Cleaned MC4 + Oscar), artık erişim iznimiz var
        ("uonlp/CulturaX", "tr", None, 5000),
        # Dialogue / Conversational
        ("Helsinki-NLP/open_subtitles", None, "tr", 2000),
        # High Quality Corpus (Reserved part implied by streaming logic if careful, 
        # but here just grabbing diverse samples)
        ("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", None, None, 3000),
    ]
    
    examples = []
    print("Building fixed validation set...")
    for path, config, lang, n in val_sources:
        try:
            if lang:
                ds = load_dataset(path, lang, split="train", streaming=True)
            elif config:
                ds = load_dataset(path, config, split="train", streaming=True)
            else:
                ds = load_dataset(path, split="train", streaming=True)
            
            # Cache n examples in memory
            current_examples = list(ds.take(n))
            examples.extend(current_examples)
            print(f"Loaded {len(current_examples)} from {path}")
        except Exception as e:
            logger.warning(f"Failed to load validation source {path}: {e}")
            continue
    
    if not examples:
        logger.warning("All validation sources failed. Creating dummy validation set.")
        examples = [{"text": "merhaba dünya " * 10}] * 100
        
    print(f"Total validation size: {len(examples)}")
    
    class FixedVal:
        def __iter__(self):
            for ex in examples:
                text = ex.get("text") or ex.get("content") or ""
                if len(text) > 50:
                    ids, edges = tokenizer.encode_with_topology(text)
                    if len(ids) > max_seq_len:
                        ids = ids[:max_seq_len]
                        
                    x, H_in, H_out, y = text_to_hypergraph(ids, max_seq_len, topology_edges=edges)
                    # Filter too short sequences (model crashes or useless)
                    if len(x) > 10:
                        yield x.astype(np.int32), H_in.astype(np.float32), H_out.astype(np.float32), y.astype(np.int32)
    
    return FixedVal()
