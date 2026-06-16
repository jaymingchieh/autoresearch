#!/usr/bin/env python3
"""zh_translate_project.py — B-TW.13 translate-train label projection.

Projects English CODE-ACCORD labels into zh-Hant silver via NLLB-200
mark-then-translate (EasyProject/LabelPigeon style). The relation file already
carries <e1>...</e1> <e2>...</e2> inline markers, so relation projection =
translate the tagged sentence and re-parse the markers.

Runs on DGX (GPU + transformers). Offline data-prep only — the extractor stays
XLM-R; NLLB is never an inference-time component (privacy constraint respected).

Stage 1 (--stage1 N): translate N relation tagged_sentences, report per-triple
tag-survival rate (both <e1> and <e2> must survive). Go/no-go gate: >=85%.

Usage (on DGX):
  uv run python zh_translate_project.py --stage1 20
"""
from __future__ import annotations

import argparse
import csv
import re
import sys

MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "zho_Hant"

E1 = re.compile(r"<e1>(.*?)</e1>", re.S)
E2 = re.compile(r"<e2>(.*?)</e2>", re.S)

# Marker styles to sweep. Each = (e1_open, e1_close, e2_open, e2_close).
# Goal: distinct e1/e2 pairs that NLLB copies verbatim when targeting Chinese.
STYLES = {
    "sq_paren":   ("[", "]", "(", ")"),    # ref: 63-65%
    "sq_lentic":  ("[", "]", "〔", "〕"),    # e2 lenticular (survived in sample)
    "sq_black":   ("[", "]", "【", "】"),    # e2 black lenticular
    "sq_dquote":  ("[", "]", "《", "》"),    # e2 double angle
    "sq_brace":   ("[", "]", "「", "」"),    # e2 corner bracket
}


def restyle(xml_sent: str, style: str) -> str:
    """Convert an <e1>/<e2> tagged sentence to the given marker style."""
    o1, c1, o2, c2 = STYLES[style]
    s = E1.sub(lambda m: f"{o1}{m.group(1)}{c1}", xml_sent)
    s = E2.sub(lambda m: f"{o2}{m.group(1)}{c2}", s)
    return s


def survived_pair(zh: str, opener: str, closer: str) -> bool:
    """Whether a marker pair survived in zh with non-empty content between."""
    i = zh.find(opener)
    if i < 0:
        return False
    j = zh.find(closer, i + len(opener))
    return j > i and zh[i + len(opener):j].strip() != ""


def load_translator():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading {MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)

    def translate(texts, batch_size=8, max_length=256):
        out_all = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                     max_length=max_length, num_beams=4)
            out_all.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out_all

    return translate


def read_relations(path, limit=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def survival(zh_sent: str) -> tuple[bool, bool]:
    """Return (e1_ok, e2_ok): whether each marker pair survived with content."""
    m1, m2 = E1.search(zh_sent), E2.search(zh_sent)
    e1_ok = bool(m1 and m1.group(1).strip())
    e2_ok = bool(m2 and m2.group(1).strip())
    return e1_ok, e2_ok


def stage1(n: int, rel_path: str):
    rows = read_relations(rel_path, limit=n)
    print(f"Stage 1: translating {len(rows)} relation tagged_sentences "
          f"(EN→{TGT_LANG}) to measure tag survival\n", flush=True)
    translate = load_translator()
    srcs = [r["tagged_sentence"] for r in rows]
    zhs = translate(srcs)

    both = e1c = e2c = 0
    for r, zh in zip(rows, zhs):
        e1_ok, e2_ok = survival(zh)
        e1c += e1_ok
        e2c += e2_ok
        both += (e1_ok and e2_ok)
    n_t = len(rows)
    print("=== SAMPLES (first 5) ===")
    for r, zh in list(zip(rows, zhs))[:5]:
        e1_ok, e2_ok = survival(zh)
        print(f"[{r['relation_type']}] e1={e1_ok} e2={e2_ok}")
        print(f"  EN: {r['tagged_sentence'][:140]}")
        print(f"  ZH: {zh[:140]}\n")
    print("=== SURVIVAL ===")
    print(f"  <e1> survived: {e1c}/{n_t} ({100*e1c/n_t:.1f}%)")
    print(f"  <e2> survived: {e2c}/{n_t} ({100*e2c/n_t:.1f}%)")
    print(f"  per-triple (both): {both}/{n_t} ({100*both/n_t:.1f}%)")
    gate = 100 * both / n_t
    print(f"\n  GATE (>=85%): {'PASS ✅' if gate >= 85 else 'FAIL ❌ — switch marker style / constrained decoding'}")
    return 0


def sweep(n: int, rel_path: str):
    """Translate N sentences under each marker style; report per-triple survival."""
    rows = read_relations(rel_path, limit=n)
    print(f"Marker-style sweep: {len(rows)} sentences × {len(STYLES)} styles\n", flush=True)
    translate = load_translator()
    results = {}
    samples = {}
    for style in STYLES:
        o1, c1, o2, c2 = STYLES[style]
        srcs = [restyle(r["tagged_sentence"], style) for r in rows]
        zhs = translate(srcs)
        both = 0
        for zh in zhs:
            if survived_pair(zh, o1, c1) and survived_pair(zh, o2, c2):
                both += 1
        results[style] = 100 * both / len(rows)
        samples[style] = zhs[1]  # the 'necessity' hard-water example
    print("=== PER-TRIPLE SURVIVAL BY STYLE ===")
    for style, pct in sorted(results.items(), key=lambda x: -x[1]):
        o1, c1, o2, c2 = STYLES[style]
        print(f"  {style:10s} {pct:5.1f}%   e1={o1}..{c1}  e2={o2}..{c2}")
        print(f"      sample: {samples[style][:130]}")
    best = max(results, key=results.get)
    print(f"\n  BEST: {best} ({results[best]:.1f}%)  "
          f"{'— viable, proceed to full projection' if results[best] >= 85 else '— still < 85%, need constrained decoding or word-alignment'}")
    return 0


# Chosen marker style (Stage-1 sweep winner: [e1] (e2), 63% per-triple survival).
PROJ = ("[", "]", "(", ")")
NONE_TYPES = ("none",)  # EN relation_type for non-relations → skip as positive

# EN BIO types are lowercase (object/property/quality/value); zh CSVs match.


def parse_en_entity_types(entities_path):
    """{example_id: [(surface_lower, type_lower), ...]} from EN BIO entities."""
    out = {}
    with open(entities_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words = row["processed_content"].split()
            tags = row["label"].split()
            spans, cur, cur_t = [], [], None
            for w, t in zip(words, tags):
                if t.startswith("B-"):
                    if cur:
                        spans.append((" ".join(cur).lower(), cur_t))
                    cur, cur_t = [w], t[2:]
                elif t.startswith("I-") and cur:
                    cur.append(w)
                else:
                    if cur:
                        spans.append((" ".join(cur).lower(), cur_t))
                    cur, cur_t = [], None
            if cur:
                spans.append((" ".join(cur).lower(), cur_t))
            out[row["example_id"]] = spans
    return out


def lookup_type(en_text, ex_id, en_types, default="object"):
    """Best-effort entity type for an EN surface via the EN entity BIO."""
    spans = en_types.get(ex_id, [])
    t = en_text.strip().lower()
    for surf, typ in spans:
        if surf == t:
            return typ
    for surf, typ in spans:  # substring fallback
        if t in surf or surf in t:
            return typ
    return default


def parse_marked_zh(zh):
    """From '在 [硬水區域] 的 (措施)...' return (clean_chars, e1span, e2span) with
    spaces dropped (zh is char-level) and inclusive char spans, or None if a
    marker pair did not survive."""
    o1, c1, o2, c2 = PROJ
    clean = []
    e1s = e1e = e2s = e2e = None
    for ch in zh:
        if ch == o1: e1s = len(clean); continue
        if ch == c1: e1e = len(clean); continue
        if ch == o2: e2s = len(clean); continue
        if ch == c2: e2e = len(clean); continue
        if ch.isspace(): continue
        clean.append(ch)
    if None in (e1s, e1e, e2s, e2e) or e1e <= e1s or e2e <= e2s:
        return None
    return "".join(clean), (e1s, e1e - 1), (e2s, e2e - 1)


def build_bio(n, spans_types):
    """Char-level BIO over n chars. spans_types: [((s,e), type), ...]."""
    bio = ["O"] * n
    for (s, e), typ in spans_types:
        if 0 <= s <= e < n:
            bio[s] = f"B-{typ}"
            for i in range(s + 1, e + 1):
                bio[i] = f"I-{typ}"
    return bio


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mix_native(out_dir, native_dir, dev_frac=0.15):
    """Append native silver train into out_dir's train; carve a native dev.csv
    (by sentence id); copy native test.csv. Keeps dev/test native."""
    import os, shutil
    ent = _read_csv(f"{native_dir}/entities/train.csv")
    rel = _read_csv(f"{native_dir}/relations/train.csv")
    ids = [r["example_id"] for r in ent]
    n_dev = max(1, int(len(ids) * dev_frac))
    dev_ids = set(ids[-n_dev:])                       # deterministic native dev
    ent_tr = [r for r in ent if r["example_id"] not in dev_ids]
    ent_dv = [r for r in ent if r["example_id"] in dev_ids]
    rel_tr = [r for r in rel if r["example_id"] not in dev_ids]
    rel_dv = [r for r in rel if r["example_id"] in dev_ids]

    def append(path, rows, fields):
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)

    def write(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    EF = ["example_id", "content", "processed_content", "label", "metadata"]
    RF = ["example_id", "content", "metadata", "tagged_sentence", "relation_type"]
    append(f"{out_dir}/entities/train.csv", ent_tr, EF)
    append(f"{out_dir}/relations/train.csv", rel_tr, RF)
    write(f"{out_dir}/entities/dev.csv", ent_dv, EF)
    write(f"{out_dir}/relations/dev.csv", rel_dv, RF)
    shutil.copy(f"{native_dir}/entities/test.csv", f"{out_dir}/entities/test.csv")
    shutil.copy(f"{native_dir}/relations/test.csv", f"{out_dir}/relations/test.csv")
    print(f"  mixed: +{len(ent_tr)} native-train sents | dev={len(ent_dv)} sents (native) | test copied")


def project(limit, rel_path, ent_path, out_dir, mix_dir=None):
    import os
    rows = read_relations(rel_path)
    rows = [r for r in rows if r["relation_type"] not in NONE_TYPES]
    if limit:
        rows = rows[:limit]
    en_types = parse_en_entity_types(ent_path)
    print(f"Projecting {len(rows)} real EN relations → zh (marker {PROJ[0]}..{PROJ[1]} / {PROJ[2]}..{PROJ[3]})", flush=True)
    translate = load_translator()
    srcs = [restyle(r["tagged_sentence"], "_proj") for r in rows]
    zhs = translate(srcs)

    ent_rows, rel_rows = [], []
    kept = 0
    for r, zh in zip(rows, zhs):
        parsed = parse_marked_zh(zh)
        if not parsed:
            continue
        clean, (e1s, e1e), (e2s, e2e) = parsed
        ex = f"acc-tr-{kept:05d}"
        e1_text = re.sub(r"</?e[12]>", "", E1.search(r["tagged_sentence"]).group(1)).strip()
        e2_text = re.sub(r"</?e[12]>", "", E2.search(r["tagged_sentence"]).group(1)).strip()
        t1 = lookup_type(e1_text, r["example_id"], en_types)
        t2 = lookup_type(e2_text, r["example_id"], en_types)
        chars = list(clean)
        bio = build_bio(len(chars), [((e1s, e1e), t1), ((e2s, e2e), t2)])
        meta = '{"src": "translate", "en_id": "%s"}' % r["example_id"]
        ent_rows.append({
            "example_id": ex, "content": clean,
            "processed_content": " ".join(chars), "label": " ".join(bio),
            "metadata": meta,
        })
        e1_surf = clean[e1s:e1e + 1]
        e2_surf = clean[e2s:e2e + 1]
        tagged = clean[:e1s] + f"<e1>{e1_surf}</e1>" + clean[e1e + 1:e2s] + \
                 f"<e2>{e2_surf}</e2>" + clean[e2e + 1:] if e1s < e2s else \
                 clean[:e2s] + f"<e2>{e2_surf}</e2>" + clean[e2e + 1:e1s] + \
                 f"<e1>{e1_surf}</e1>" + clean[e1e + 1:]
        rel_rows.append({
            "example_id": ex, "content": clean, "metadata": meta,
            "tagged_sentence": tagged, "relation_type": r["relation_type"],
        })
        kept += 1
    surv = 100 * kept / max(len(rows), 1)
    print(f"  kept {kept}/{len(rows)} ({surv:.1f}% per-triple survival)")

    os.makedirs(f"{out_dir}/entities", exist_ok=True)
    os.makedirs(f"{out_dir}/relations", exist_ok=True)
    with open(f"{out_dir}/entities/train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "content", "processed_content", "label", "metadata"])
        w.writeheader(); w.writerows(ent_rows)
    with open(f"{out_dir}/relations/train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "content", "metadata", "tagged_sentence", "relation_type"])
        w.writeheader(); w.writerows(rel_rows)
    print(f"  wrote {out_dir}/{{entities,relations}}/train.csv ({kept} examples)")
    if mix_dir:
        mix_native(out_dir, mix_dir)
    print("\n  SAMPLES:")
    for er, rr in list(zip(ent_rows, rel_rows))[:3]:
        print(f"   [{rr['relation_type']}] {rr['tagged_sentence'][:90]}")
        print(f"       BIO: {er['label'][:80]}")
    return 0


def main():
    # register the projection marker style for restyle()
    STYLES["_proj"] = PROJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=int, metavar="N",
                    help="translate N relation sentences, report tag survival")
    ap.add_argument("--sweep", type=int, metavar="N",
                    help="sweep marker styles over N sentences, report survival")
    ap.add_argument("--project", nargs="?", type=int, const=0, metavar="LIMIT",
                    help="full projection → zh silver CSVs (optional LIMIT for a test)")
    ap.add_argument("--rel-path", default="data/code_accord/relations/train.csv")
    ap.add_argument("--ent-path", default="data/code_accord/entities/train.csv")
    ap.add_argument("--out-dir", default="data/accord_zh_translate")
    ap.add_argument("--mix-native", default=None,
                    help="native silver dir to mix in + carve native dev/test (e.g. data/accord_zh_v3_2)")
    args = ap.parse_args()
    if args.project is not None:
        return project(args.project, args.rel_path, args.ent_path, args.out_dir,
                       mix_dir=args.mix_native)
    if args.sweep:
        return sweep(args.sweep, args.rel_path)
    if args.stage1:
        return stage1(args.stage1, args.rel_path)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
