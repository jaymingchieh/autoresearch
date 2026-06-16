"""
CODE-ACCORD dataset — construction regulations NER + RE.
4 entity types, 9 relation types (+ none/NO_REL).

Entities CSV: BIO-tagged sentences (one row per sentence).
Relations CSV: entity-pair-marked sentences (one row per pair).

Format compatible with SciERC pipeline (same collate_fn, same batch structure).
"""
import csv
import ast
import json
import re
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Label vocabulary ─────────────────────────────────────────────────
ENTITY_TYPES = ["Object", "Property", "Quality", "Value"]
RELATION_TYPES = [
    "selection", "necessity", "part-of", "not-part-of",
    "equal", "greater", "greater-equal", "less", "less-equal",
]

# BIO tag set: O + (B-, I-) per entity type → 1 + 2*4 = 9
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIO_TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for t, i in BIO_TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)

NO_REL_ID = 0
REL2ID = {"NO_REL": NO_REL_ID, **{r: i + 1 for i, r in enumerate(RELATION_TYPES)}}
ID2REL = {i: r for r, i in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)  # 10

# Comparison/numeric relation IDs for class-weighted loss (see A11 experiment).
# These relations are rare (equal=51, greater=32, greater-equal=74, less=7, less-equal=41
# out of 2200 training relations) yet have 0% evidence-path reachability.
COMPARISON_REL_IDS = [5, 6, 7, 8, 9]  # equal, greater, greater-equal, less, less-equal


def _doc_id_from_metadata(raw: str) -> str:
    """Extract source document ID from CODE-ACCORD metadata."""
    try:
        meta = ast.literal_eval(raw) if raw else {}
    except (SyntaxError, ValueError):
        meta = {}
    return str(meta.get("ID", "UNKNOWN"))


def _bio_tags_for_sentence(num_words: int, ner_spans: list) -> list:
    """Convert ner spans (start, end_inclusive, type) to per-word BIO ids."""
    tags = [BIO_TAG2ID["O"]] * num_words
    for s, e, t in ner_spans:
        if s >= num_words or e >= num_words:
            continue
        tag_b = f"B-{t}"
        tag_i = f"I-{t}"
        if tag_b not in BIO_TAG2ID:
            continue
        tags[s] = BIO_TAG2ID[tag_b]
        for k in range(s + 1, e + 1):
            tags[k] = BIO_TAG2ID[tag_i]
    return tags


def _bio_to_spans(bio_tags: list) -> list:
    """Convert BIO tag strings to (start, end_inclusive, type) spans."""
    spans = []
    cur_start = None
    cur_type = None
    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if cur_start is not None:
                spans.append((cur_start, i - 1, cur_type))
            cur_start = i
            cur_type = tag[2:].capitalize()
        elif tag.startswith("I-"):
            if cur_start is None:
                cur_start = i
                cur_type = tag[2:].capitalize()
        else:
            if cur_start is not None:
                spans.append((cur_start, i - 1, cur_type))
                cur_start = None
                cur_type = None
    if cur_start is not None:
        spans.append((cur_start, len(bio_tags) - 1, cur_type))
    return spans


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: collapse punctuation spacing."""
    t = text.lower()
    t = t.replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
    t = t.replace(" -", "-").replace("- ", "-")  # grid - supplied → grid-supplied
    t = t.replace(" '", "'")  # building 's → building's
    t = t.replace(" %", "%").replace(" °", "°")
    return t


def _find_span_in_words(entity_text: str, words: list, hint_start: int = 0) -> tuple:
    """Find the word-level span of entity_text in words list.
    Returns (start, end_inclusive) or None if not found.
    Handles punctuation-split words (processed_content has "grid - supplied"
    while entity text has "grid-supplied")."""
    entity_norm = _normalize_text(entity_text)
    if not entity_norm.strip():
        return None
    # Try variable-length windows
    entity_words = entity_text.split()
    min_len = len(entity_words)
    max_len = min_len + 5  # allow up to 5 extra tokens for split punctuation
    for wlen in range(min_len, min(max_len + 1, len(words) + 1)):
        for i in range(len(words) - wlen + 1):
            window_norm = _normalize_text(" ".join(words[i:i + wlen]))
            if window_norm == entity_norm:
                return (i, i + wlen - 1)
    return None


def _load_entities(csv_path: str) -> dict:
    """Load entity CSV → dict[example_id] = {"words": [...], "ner": [...]}."""
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for seq, row in enumerate(reader):
            eid = row["example_id"]
            words = row["processed_content"].split()
            bio_tags = row["label"].split()
            if len(bio_tags) != len(words):
                continue  # skip malformed
            spans = _bio_to_spans(bio_tags)
            result[eid] = {
                "words": words,
                "ner": spans,
                "doc_id": _doc_id_from_metadata(row.get("metadata", "")),
                "seq": seq,
            }
    return result


def _load_relations(csv_path: str, entity_data: dict) -> dict:
    """Load relation CSV → dict[example_id] = list of ((hs,he), (ts,te), rel_id)."""
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for seq, row in enumerate(reader):
            eid = row["example_id"]
            rel_type = row["relation_type"]
            if rel_type == "none":
                continue  # skip NO_REL pairs (implicit)
            if rel_type not in REL2ID:
                continue

            tagged = row["tagged_sentence"]
            e1_match = re.search(r"<e1>(.*?)</e1>", tagged)
            e2_match = re.search(r"<e2>(.*?)</e2>", tagged)
            if not e1_match or not e2_match:
                continue

            e1_text = e1_match.group(1)
            e2_text = e2_match.group(1)

            # Get words from entity data if available, else tokenize tagged sentence
            if eid in entity_data:
                words = entity_data[eid]["words"]
            else:
                clean = re.sub(r"</?e[12]>", "", tagged)
                # Tokenize: split punctuation from words
                words = re.findall(r"\w+|[^\w\s]", clean)

            e1_span = _find_span_in_words(e1_text, words)
            e2_span = _find_span_in_words(e2_text, words)

            if e1_span is None or e2_span is None:
                continue

            if eid not in result:
                result[eid] = {
                    "words": words,
                    "relations": [],
                    "doc_id": _doc_id_from_metadata(row.get("metadata", "")),
                    "seq": seq,
                }
            result[eid]["relations"].append(
                (e1_span, e2_span, REL2ID[rel_type])
            )
    return result


def _make_doc_windows(examples: list, window_size: int = 1, stride: int = 1) -> list:
    """Build sliding document windows from consecutive same-document sentences.

    Phase A8 uses ``window_size=2`` so each training sample is the current
    sentence plus the next sentence. Entity and relation spans remain
    word-indexed, shifted by each sentence's offset inside the joined context.
    Relations stay sentence-local because CODE-ACCORD does not annotate
    cross-sentence relation pairs.
    """
    if window_size <= 1:
        return examples

    stride = max(stride, 1)
    by_doc = {}
    for ex in examples:
        by_doc.setdefault(ex.get("doc_id", "UNKNOWN"), []).append(ex)

    windows = []
    for doc_id, doc_examples in by_doc.items():
        doc_examples = sorted(doc_examples, key=lambda ex: ex.get("seq", 0))
        if not doc_examples:
            continue
        for start in range(0, len(doc_examples), stride):
            chunk = doc_examples[start:start + window_size]
            if not chunk:
                continue

            words = []
            ner = []
            relations = []
            offset = 0
            eids = []
            for sent in chunk:
                eids.append(sent.get("example_id", ""))
                words.extend(sent["words"])
                ner.extend((s + offset, e + offset, t) for s, e, t in sent["ner"])
                relations.extend(
                    ((hs + offset, he + offset), (ts + offset, te + offset), rid)
                    for (hs, he), (ts, te), rid in sent["relations"]
                )
                offset += len(sent["words"])

            windows.append({
                "example_id": "|".join(eids),
                "doc_id": doc_id,
                "words": words,
                "ner": ner,
                "relations": relations,
            })
    return windows


class CodeAccordDataset(Dataset):
    """
    Sentence-level CODE-ACCORD for span NER+RE training.
    """
    def __init__(self, examples: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids()

        word_bio = _bio_tags_for_sentence(len(words), ex["ner"])
        token_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                token_labels.append(-100)
            elif wid != prev_word_id:
                token_labels.append(word_bio[wid])
            else:
                token_labels.append(-100)
            prev_word_id = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_ids": word_ids,
            "ner_labels": token_labels,
            "gold_entities": ex["ner"],
            "gold_relations": ex["relations"],
            "num_words": len(words),
            "words": words,
            "doc_id": ex.get("doc_id", ""),
            "example_id": ex.get("example_id", ""),
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Pad batch to max length. Same as SciERC collate_fn."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    ner_labels = torch.full((B, max_len), -100, dtype=torch.long)

    word_ids_list = []
    gold_entities_list = []
    gold_relations_list = []
    num_words_list = []
    words_list = []
    doc_id_list = []
    example_id_list = []

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
        ner_labels[i, :L] = torch.tensor(b["ner_labels"], dtype=torch.long)
        word_ids_list.append(b["word_ids"])
        gold_entities_list.append(b["gold_entities"])
        gold_relations_list.append(b["gold_relations"])
        num_words_list.append(b["num_words"])
        words_list.append(b["words"])
        doc_id_list.append(b.get("doc_id", ""))
        example_id_list.append(b.get("example_id", ""))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "word_ids": word_ids_list,
        "gold_entities": gold_entities_list,
        "gold_relations": gold_relations_list,
        "num_words": num_words_list,
        "words": words_list,
        "doc_ids": doc_id_list,
        "example_ids": example_id_list,
    }


DEFAULT_DATA_DIR = Path(__file__).parent / "code_accord"


class DocGroupDataset(Dataset):
    """
    Phase B3: Document-level dataset for Evidence GAT training.

    Each item is a list of tokenized sentence examples from the same document.
    The training loop encodes each sentence independently with DeBERTa, then
    runs the Evidence GAT over entity representations from the whole document.

    Unlike CodeAccordDataset (sentence-level), this dataset groups by doc_id
    so that cross-sentence entity interactions can be modeled.
    """

    def __init__(self, doc_groups: list, tokenizer, max_length: int = 128):
        """
        Args:
            doc_groups: list of lists — each inner list is the examples from
                        one document, already sorted by sentence sequence.
            tokenizer:  HF tokenizer for encoding words.
            max_length: max token length per sentence (not per document).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_groups = doc_groups  # list of lists of sentence examples

    def __len__(self):
        return len(self.doc_groups)

    def __getitem__(self, idx):
        """Returns list of tokenized sentence dicts for one document."""
        sents = self.doc_groups[idx]
        result = []
        for ex in sents:
            words = ex["words"]
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            word_ids = encoding.word_ids()

            word_bio = _bio_tags_for_sentence(len(words), ex["ner"])
            token_labels = []
            prev_word_id = None
            for wid in word_ids:
                if wid is None:
                    token_labels.append(-100)
                elif wid != prev_word_id:
                    token_labels.append(word_bio[wid])
                else:
                    token_labels.append(-100)
                prev_word_id = wid

            result.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_ids": word_ids,
                "ner_labels": token_labels,
                "gold_entities": ex["ner"],
                "gold_relations": ex["relations"],
                "num_words": len(words),
                "words": words,
                "doc_id": ex.get("doc_id", ""),
                "example_id": ex.get("example_id", ""),
            })
        return result


def collate_doc(docs, pad_token_id: int = 0):
    """
    Collate function for DocGroupDataset. Each call receives a list of length 1
    (since doc-level DataLoader uses batch_size=1). Returns a flat list of
    tokenized sentence dicts padded within this batch.

    Each sentence dict in the returned list has tensors ready for the model:
        input_ids:      (1, T_i)  — padded
        attention_mask: (1, T_i)
        ner_labels:     (1, T_i)
        word_ids:       list[int|None]
        gold_entities:  list of (s, e, type) tuples
        gold_relations: list of ((hs,he), (ts,te), rel_id) tuples
    """
    assert len(docs) == 1, "DocGroupDataset must use DataLoader(batch_size=1)"
    sents = docs[0]  # list of sentence dicts
    result = []
    for sent in sents:
        L = len(sent["input_ids"])
        input_ids = torch.tensor(sent["input_ids"], dtype=torch.long).unsqueeze(0)
        attn_mask = torch.tensor(sent["attention_mask"], dtype=torch.long).unsqueeze(0)
        ner_labels = torch.tensor(sent["ner_labels"], dtype=torch.long).unsqueeze(0)
        result.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "ner_labels": ner_labels,
            "word_ids": [sent["word_ids"]],
            "gold_entities": [sent["gold_entities"]],
            "gold_relations": [sent["gold_relations"]],
            "num_words": [sent["num_words"]],
            "words": [sent["words"]],
            "doc_id": sent.get("doc_id", ""),
            "example_id": sent.get("example_id", ""),
        })
    return result


def build_doc_dataloaders(tokenizer, data_dir=None, batch_size: int = 1,
                          max_length: int = 128, num_workers: int = 0,
                          dev_ratio: float = 0.15, seed: int = 42):
    """
    Phase B3: Document-level dataloaders for Evidence GAT training.

    Unlike build_dataloaders (sentence-level), each iteration yields all
    sentences from one document. batch_size is fixed to 1 document per step.

    Returns:
        (train_loader, dev_loader, test_loader) — same interface as
        build_dataloaders but each batch is a list of sentence dicts.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    pad_id = tokenizer.pad_token_id

    ent_train = _load_entities(data_dir / "entities" / "train.csv")
    rel_train = _load_relations(data_dir / "relations" / "train.csv", ent_train)
    ent_test = _load_entities(data_dir / "entities" / "test.csv")
    rel_test = _load_relations(data_dir / "relations" / "test.csv", ent_test)

    def _merge(ent_data, rel_data, require_entities=False):
        examples = []
        all_ids = list(ent_data.keys()) if require_entities else (
            list(ent_data.keys()) + [e for e in rel_data.keys() if e not in ent_data]
        )
        for eid in all_ids:
            if eid in ent_data:
                words = ent_data[eid]["words"]
                ner = ent_data[eid]["ner"]
                doc_id = ent_data[eid].get("doc_id", "UNKNOWN")
            elif eid in rel_data:
                words = rel_data[eid]["words"]
                ner = []
                doc_id = rel_data[eid].get("doc_id", "UNKNOWN")
            else:
                continue
            relations = rel_data[eid]["relations"] if eid in rel_data else []
            examples.append({
                "example_id": eid,
                "doc_id": doc_id,
                "seq": ent_data.get(eid, rel_data.get(eid, {})).get("seq", 0),
                "words": words,
                "ner": ner,
                "relations": relations,
            })
        return examples

    train_examples = _merge(ent_train, rel_train, require_entities=False)
    test_examples = _merge(ent_test, rel_test, require_entities=True)

    rng = random.Random(seed)
    # B-TW.13: if an explicit dev.csv exists, use it as a fixed (native) dev set
    # and train on ALL of train.csv. Keeps dev native when train is mixed with
    # translated silver (else the dev_ratio slice would dilute dev with noise).
    dev_csv = data_dir / "entities" / "dev.csv"
    if dev_csv.exists():
        ent_dev = _load_entities(dev_csv)
        rel_dev = _load_relations(data_dir / "relations" / "dev.csv", ent_dev)
        dev_examples = _merge(ent_dev, rel_dev, require_entities=True)
        rng.shuffle(train_examples)
    else:
        rng.shuffle(train_examples)
        n_dev = int(len(train_examples) * dev_ratio)
        dev_examples = train_examples[:n_dev]
        train_examples = train_examples[n_dev:]

    def _group_by_doc(examples):
        """Group and sort sentences by doc_id → list of doc groups."""
        by_doc = {}
        for ex in examples:
            by_doc.setdefault(ex.get("doc_id", "UNKNOWN"), []).append(ex)
        groups = []
        for doc_id, doc_exs in by_doc.items():
            doc_exs = sorted(doc_exs, key=lambda e: e.get("seq", 0))
            groups.append(doc_exs)
        return groups

    train_groups = _group_by_doc(train_examples)
    dev_groups = _group_by_doc(dev_examples)
    test_groups = _group_by_doc(test_examples)

    n_train_sents = sum(len(g) for g in train_groups)
    n_dev_sents = sum(len(g) for g in dev_groups)
    print(f"  CODE-ACCORD (doc-level): train={len(train_groups)} docs ({n_train_sents} sents) "
          f"dev={len(dev_groups)} docs ({n_dev_sents} sents) "
          f"test={len(test_groups)} docs")
    print(f"  Train rels: {sum(len(e['relations']) for g in train_groups for e in g)}")
    print(f"  Dev rels:   {sum(len(e['relations']) for g in dev_groups for e in g)}")

    def make_loader(groups, shuffle):
        ds = DocGroupDataset(groups, tokenizer, max_length)
        return DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
            collate_fn=lambda b: collate_doc(b, pad_token_id=pad_id),
        )

    return make_loader(train_groups, True), make_loader(dev_groups, False), make_loader(test_groups, False)


def build_dataloaders(tokenizer, data_dir=None, batch_size: int = 16,
                      max_length: int = 128, num_workers: int = 0,
                      dev_ratio: float = 0.15, seed: int = 42,
                      doc_window_size: int = 1, doc_window_stride: int = 1):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    pad_id = tokenizer.pad_token_id

    # Load entity and relation data
    ent_train = _load_entities(data_dir / "entities" / "train.csv")
    rel_train = _load_relations(data_dir / "relations" / "train.csv", ent_train)
    ent_test = _load_entities(data_dir / "entities" / "test.csv")
    rel_test = _load_relations(data_dir / "relations" / "test.csv", ent_test)

    def _merge(ent_data, rel_data, require_entities=False):
        """Merge entity and relation data into examples list.
        If require_entities=True, only include sentences that have entity annotations.
        This is critical for test sets where entity and relation CSVs cover
        different sentences — relation-only sentences have no NER ground truth.
        """
        examples = []
        if require_entities:
            all_ids = list(ent_data.keys())  # only entity-annotated sentences
        else:
            all_ids = list(ent_data.keys()) + [
                eid for eid in rel_data.keys() if eid not in ent_data
            ]
        for eid in all_ids:
            if eid in ent_data:
                words = ent_data[eid]["words"]
                ner = ent_data[eid]["ner"]
                doc_id = ent_data[eid].get("doc_id", "UNKNOWN")
            elif eid in rel_data:
                words = rel_data[eid]["words"]
                ner = []
                doc_id = rel_data[eid].get("doc_id", "UNKNOWN")
            else:
                continue

            if eid in rel_data:
                relations = rel_data[eid]["relations"]
            else:
                relations = []

            examples.append({
                "example_id": eid,
                "doc_id": doc_id,
                "seq": ent_data.get(eid, rel_data.get(eid, {})).get("seq", 0),
                "words": words,
                "ner": ner,
                "relations": relations,
            })
        return examples

    train_examples = _merge(ent_train, rel_train, require_entities=False)
    test_examples = _merge(ent_test, rel_test, require_entities=True)

    # Create dev split from training data
    rng = random.Random(seed)
    rng.shuffle(train_examples)
    n_dev = int(len(train_examples) * dev_ratio)
    dev_examples = train_examples[:n_dev]
    train_examples = train_examples[n_dev:]

    train_examples = _make_doc_windows(train_examples, doc_window_size, doc_window_stride)
    dev_examples = _make_doc_windows(dev_examples, doc_window_size, doc_window_stride)
    test_examples = _make_doc_windows(test_examples, doc_window_size, doc_window_stride)

    print(f"  CODE-ACCORD: train={len(train_examples)} dev={len(dev_examples)} test={len(test_examples)}")
    if doc_window_size > 1:
        print(f"  Doc windows: size={doc_window_size} stride={doc_window_stride}")
    print(f"  Train rels: {sum(len(e['relations']) for e in train_examples)}")
    print(f"  Dev rels:   {sum(len(e['relations']) for e in dev_examples)}")

    def make_loader(examples, shuffle):
        ds = CodeAccordDataset(examples, tokenizer, max_length)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
        )

    return make_loader(train_examples, True), make_loader(dev_examples, False), make_loader(test_examples, False)
