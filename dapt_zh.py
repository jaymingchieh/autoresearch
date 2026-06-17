#!/usr/bin/env python3
"""dapt_zh.py — B-TW.14 Domain-Adaptive Pre-Training for TW building regulations.

Modes:
  --stage1 N            Download N articles + 50-step smoke test (sanity check)
  --prep-data OUT_DIR   Download all 12 TW building laws → OUT_DIR/corpus.txt
  --train               Full DAPT: LoRA-FFN MLM → checkpoints/zh_dapt_xlmr.pt

Architecture: XLMRobertaForMaskedLM + manual LoRA(r=16, α=32) on FFN layers.
Output checkpoint is {"encoder": roberta.state_dict()}, compatible with
train_span.py --pretrain-ckpt (same format as cuad_pretrain_xlmr_short.pt).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, XLMRobertaForMaskedLM

# ── Constants ─────────────────────────────────────────────────────────────────

# Bulk ZIP download: https://law.moj.gov.tw/api/Ch/Law/JSON
# → ChLaw.json with {"Laws": [{"LawName":..., "LawArticles":[{"ArticleContent":...}]}]}
BULK_JSON_URL = "https://law.moj.gov.tw/api/Ch/Law/JSON"

# Filter: law names containing these keywords
LAW_KEYWORDS = ["建築", "消防", "都市計畫", "住宅", "公寓大廈", "室內裝修", "區域計畫"]
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 32
LR = 1e-4
WARMUP_STEPS = 200
TRAIN_STEPS = 3000
SPAN_GEO_P = 0.2
MAX_SPAN = 5
MLM_PROB = 0.15
LORA_R = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.1
MIN_SENT_LEN = 15
SAVE_EVERY = 500

# ── Manual LoRA ───────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    Only A and B matrices are trainable; original weight is frozen.
    Call merge() to fold LoRA back into the base weight for inference.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.drop(x) @ self.lora_A.t() @ self.lora_B.t() * self.scale
        return base + lora

    def merge(self) -> nn.Linear:
        w = self.linear.weight.data + (self.lora_B @ self.lora_A) * self.scale
        merged = nn.Linear(self.linear.in_features, self.linear.out_features,
                           bias=self.linear.bias is not None,
                           device=w.device, dtype=w.dtype)
        merged.weight.data.copy_(w)
        if self.linear.bias is not None:
            merged.bias.data.copy_(self.linear.bias.data)
        return merged


def inject_lora(model: XLMRobertaForMaskedLM, r: int, alpha: float, drop: float) -> None:
    for layer in model.roberta.encoder.layer:
        layer.intermediate.dense = LoRALinear(layer.intermediate.dense, r, alpha, drop)
        layer.output.dense = LoRALinear(layer.output.dense, r, alpha, drop)


def merge_lora(model: XLMRobertaForMaskedLM) -> None:
    for layer in model.roberta.encoder.layer:
        layer.intermediate.dense = layer.intermediate.dense.merge()
        layer.output.dense = layer.output.dense.merge()


def lora_trainable_params(model: XLMRobertaForMaskedLM) -> int:
    return sum(p.numel() for name, p in model.named_parameters()
               if p.requires_grad and ("lora_A" in name or "lora_B" in name))

# ── Data — TW law download + sentence splitting ───────────────────────────────

def _zh_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；]", text)
    return [p for p in (_zh_normalize(s) for s in parts) if len(p) >= MIN_SENT_LEN]


def prep_data(out_dir: Path, limit: int | None = None) -> Path:
    """Download TW building laws (bulk ZIP) → out_dir/corpus.txt.

    Source: law.moj.gov.tw/api/Ch/Law/JSON  (ZIP → ChLaw.json)
    Filter: law names containing LAW_KEYWORDS
    Content: LawArticles[*].ArticleContent
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"

    print("  downloading ChLaw.json from law.moj.gov.tw ...", flush=True)
    resp = requests.get(BULK_JSON_URL, timeout=120)
    resp.raise_for_status()
    print(f"  downloaded {len(resp.content)//1024}KB", flush=True)

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    data = json.loads(z.read("ChLaw.json"))
    all_laws = data["Laws"]

    sentences: list[str] = []
    for law in all_laws:
        name = law.get("LawName", "")
        if not any(kw in name for kw in LAW_KEYWORDS):
            continue
        articles = law.get("LawArticles", [])
        n = 0
        for art in articles:
            content = (art.get("ArticleContent") or "").strip()
            sents = _split_sentences(content)
            sentences.extend(sents)
            n += len(sents)
        print(f"  {name}: {len(articles)} articles → {n} sents")
        if limit and len(sentences) >= limit:
            sentences = sentences[:limit]
            break

    corpus_path.write_text("\n".join(sentences), encoding="utf-8")
    print(f"\n  corpus: {len(sentences)} sentences → {corpus_path}")
    return corpus_path

# ── Dataset ───────────────────────────────────────────────────────────────────

class CorpusDataset(Dataset):
    def __init__(self, corpus_path: Path):
        self.lines = [l.strip() for l in corpus_path.read_text("utf-8").splitlines()
                      if l.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.lines[idx]}

# ── Span masking collator ─────────────────────────────────────────────────────

class SpanMLMCollator:
    """SpanBERT-style span masking: geometric span length, 15% tokens masked."""

    def __init__(self, tokenizer, mlm_prob: float = MLM_PROB,
                 geo_p: float = SPAN_GEO_P, max_span: int = MAX_SPAN):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.geo_p = geo_p
        self.max_span = max_span
        self.special_ids = set(tokenizer.all_special_ids)
        self.vocab_size = len(tokenizer)

    def _span_len(self) -> int:
        length = 1
        while length < self.max_span and torch.rand(1).item() > self.geo_p:
            length += 1
        return length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            [ex["text"] for ex in examples],
            padding=True, truncation=True, max_length=MAX_LEN,
            return_tensors="pt",
        )
        orig_ids = enc["input_ids"].clone()
        input_ids = enc["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)

        for i in range(input_ids.shape[0]):
            seq_len = int((input_ids[i] != self.tokenizer.pad_token_id).sum())
            target = max(1, round(seq_len * self.mlm_prob))
            masked: set[int] = set()
            for _ in range(200):
                if len(masked) >= target:
                    break
                start = torch.randint(1, max(2, seq_len - 1), (1,)).item()
                if orig_ids[i, start].item() in self.special_ids:
                    continue
                for j in range(start, min(start + self._span_len(), seq_len - 1)):
                    if orig_ids[i, j].item() not in self.special_ids:
                        masked.add(j)
                    if len(masked) >= target:
                        break

            for pos in masked:
                labels[i, pos] = orig_ids[i, pos]
                r = torch.rand(1).item()
                if r < 0.8:
                    input_ids[i, pos] = self.tokenizer.mask_token_id
                elif r < 0.9:
                    input_ids[i, pos] = torch.randint(self.vocab_size, (1,)).item()
                # else: keep (label still set → loss on unchanged token)

        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"],
                "labels": labels}

# ── DAPT training ─────────────────────────────────────────────────────────────

def load_start_ckpt(model: XLMRobertaForMaskedLM, ckpt_path: str) -> None:
    if not ckpt_path:
        print("  no start ckpt — using raw xlm-roberta-base weights")
        return
    sd = torch.load(ckpt_path, map_location="cpu")
    if "encoder" in sd:
        sd = sd["encoder"]
    elif "discriminator" in sd:
        sd = sd["discriminator"]
    # CUAD checkpoint keys have "backbone.bert." prefix; strip to match XLMRobertaModel
    if any(k.startswith("backbone.bert.") for k in sd):
        sd = {k[len("backbone.bert."):]: v for k, v in sd.items()
              if k.startswith("backbone.bert.")}
    missing, unexpected = model.roberta.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt_path}: {len(sd)} keys | missing={len(missing)} unexpected={len(unexpected)}")


def train_dapt(corpus_path: Path, start_ckpt: str, out_ckpt: Path,
               steps: int = TRAIN_STEPS) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaForMaskedLM.from_pretrained(MODEL_NAME)
    load_start_ckpt(model, start_ckpt)

    inject_lora(model, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    model.to(device)

    n_trainable = lora_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_trainable:,} trainable / {n_total:,} total ({100*n_trainable/n_total:.2f}%)")

    dataset = CorpusDataset(corpus_path)
    collator = SpanMLMCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collator, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    epoch = 0
    loss_accum = 0.0
    log_every = 50

    print(f"\n  training {steps} steps (corpus={len(dataset)} sents, batch={BATCH_SIZE})")
    while step < steps:
        epoch += 1
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_accum += loss.item()
            step += 1

            if step % log_every == 0:
                avg = loss_accum / log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {step:4d}/{steps} | loss={avg:.4f} | lr={lr_now:.2e}")
                loss_accum = 0.0

    print("\n  merging LoRA weights ...")
    merge_lora(model)
    model.cpu()

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.roberta.state_dict()}, out_ckpt)
    print(f"  saved → {out_ckpt}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=int, metavar="N",
                    help="download N articles, show samples, run 50-step smoke test")
    ap.add_argument("--prep-data", metavar="OUT_DIR",
                    help="download all 12 laws → OUT_DIR/corpus.txt")
    ap.add_argument("--train", action="store_true",
                    help="full DAPT training")
    ap.add_argument("--corpus", default="data/dapt_zh_laws/corpus.txt",
                    help="path to corpus.txt (used by --train)")
    ap.add_argument("--start-ckpt", default="checkpoints/cuad_pretrain_xlmr_short.pt",
                    help="starting checkpoint (CUAD or empty for raw XLM-R)")
    ap.add_argument("--out-ckpt", default="checkpoints/zh_dapt_xlmr.pt",
                    help="output DAPT checkpoint path")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    args = ap.parse_args()

    if args.stage1:
        print(f"=== stage1: downloading {args.stage1} sentences (smoke test) ===")
        tmp = Path("data/dapt_zh_laws_stage1")
        corpus_path = prep_data(tmp, limit=args.stage1)
        lines = corpus_path.read_text("utf-8").splitlines()
        print("\nSAMPLE SENTENCES:")
        for l in lines[:5]:
            print(f"  {l[:80]}")
        print("\n=== smoke test: 50 DAPT steps ===")
        train_dapt(corpus_path, args.start_ckpt,
                   Path("checkpoints/zh_dapt_xlmr_smoke.pt"), steps=50)
        print("stage1 OK")
        return 0

    if args.prep_data:
        prep_data(Path(args.prep_data))
        return 0

    if args.train:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"corpus not found: {corpus_path}")
            print("run --prep-data first")
            return 1
        train_dapt(corpus_path, args.start_ckpt, Path(args.out_ckpt), args.steps)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
