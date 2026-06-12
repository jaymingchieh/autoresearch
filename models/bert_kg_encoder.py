"""
KG extractor with pluggable modality adapters (Stage 2 / Stage 3 forward-compat).

Architecture:

    raw_input (input_ids | pixel_values | rdb_schema | ...)
        ↓
    Adapter[modality]                  ← swappable; lives in models/adapters/
        ↓ (inputs_embeds, attention_mask)   unified hidden dim, modality-agnostic
        ↓
    BERT body (or any transformer that supports inputs_embeds)
        ↓ (B, T, H)
        ├── NER head: token classification (BIO scheme)
        └── RE head:  span-pair classification (NUM_RELATIONS incl. NO_REL)

Adapter contract:
    forward(**raw_inputs) -> dict with at least:
        "inputs_embeds":   (B, T, hidden)   ← matches backbone hidden_size
        "attention_mask":  (B, T)           ← 1=keep, 0=pad

Stage 2:
    TextAdapter wraps BERT's own word_embeddings layer; the math is identical
    to calling bert(input_ids=...) — verified by reproducing stage2-004.

Stage 3 (future, no code changes to backbone or heads needed):
    ImageAdapter:    ViT patch embeddings + Linear projection to hidden_size
    TableAdapter:    schema cell embeddings + Linear projection
    ...

Why this matters now: when we move to engineering documents (Stage 3 in
future_implementation_plan.md), we'll need to ingest flowcharts, RDB schemas,
CAD drawings — each with its own input format. Building the adapter pattern
now means adding a modality is "implement one Adapter class + register it";
no surgery on the model body or heads.

Stage 2 training loss (unchanged from prior version):
    L = NER cross-entropy + λ · RE cross-entropy(NUM_RELATIONS=8 incl. NO_REL)

Phase B3 (ECRG-style evidence graph):
    After per-sentence DeBERTa encoding, collect entity span representations
    across all sentences of a document. Build an evidence graph (center-sentence
    heuristic: edge if entities appear in the same or adjacent sentences).
    Run 2-layer EvidenceGATLayer over entity nodes. Use enriched representations
    for cross-sentence RE prediction. Enabled via --evidence-gat flag.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from data.scierc import NUM_BIO_TAGS, NUM_RELATIONS, NO_REL_ID, BIO_TAG2ID, ID2BIO


# ─── Phase B3: Evidence Graph Attention Layer ────────────────────────────


class EvidenceGATLayer(nn.Module):
    """
    Sparse multi-head attention over an entity evidence graph (Phase B3 / ECRG).

    Each entity node attends to its graph neighbors (defined by adjacency mask).
    Pure PyTorch — no torch_geometric dependency.

    Args:
        hidden_dim: entity representation dimension (== backbone hidden_size)
        num_heads:  number of attention heads (default: 4)

    Forward:
        x:        (N, H) — stacked entity representations
        adj_mask: (N, N) bool — True where edge exists (including self-loops)

    Returns:
        (N, H) — enriched entity representations (residual connection applied)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        x:        (N, H)
        adj_mask: (N, N) bool — True = edge exists
        Returns:  (N, H)
        """
        N, H = x.shape
        nh = self.num_heads
        d = self.head_dim

        # Project to Q, K, V and reshape to (N, num_heads, head_dim)
        Q = self.q(x).view(N, nh, d)
        K = self.k(x).view(N, nh, d)
        V = self.v(x).view(N, nh, d)

        # Scaled dot-product attention: (N, N, nh)
        # attn[i,j,h] = Q[i,h] · K[j,h] / sqrt(d)
        attn = torch.einsum("ihd,jhd->ijh", Q, K) / (d ** 0.5)

        # Mask non-edges to -inf before softmax
        if adj_mask is not None:
            # adj_mask: (N, N) → broadcast over heads
            attn = attn.masked_fill(~adj_mask.unsqueeze(-1), float("-inf"))

        # Softmax over neighbors (dim=1 = over source nodes for each target)
        attn = torch.softmax(attn, dim=1)  # (N, N, nh)

        # Handle all-masked rows (isolated nodes — softmax → NaN)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Aggregate: (N, nh, d) → (N, H)
        out = torch.einsum("ijh,jhd->ihd", attn, V).reshape(N, H)
        out = self.out(out)

        return self.norm(x + out)  # residual + layer norm


class EvidenceGAT(nn.Module):
    """
    2-layer Evidence Graph Attention Network (ECRG-style, Phase B3).

    Stacks two EvidenceGATLayer modules with a feed-forward projection in
    between. Used to exchange information between entity mentions across
    sentence boundaries.

    The same adjacency mask is used for both layers (static graph).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            EvidenceGATLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        x:        (N, H) entity representations
        adj_mask: (N, N) bool adjacency matrix (self-loops included)
        Returns:  (N, H) enriched representations
        """
        for layer in self.layers:
            x = layer(x, adj_mask)
        return x


def build_evidence_graph(entity_sentence_ids: list, n_entities: int,
                         max_sentence_gap: int = 1) -> torch.Tensor:
    """
    Build adjacency matrix for the evidence graph (center-sentence heuristic).

    An edge exists between entity i and entity j if they appear in the same
    sentence OR in adjacent sentences (|sent_i - sent_j| <= max_sentence_gap).
    Self-loops are always included.

    Args:
        entity_sentence_ids: list[int] of length N — which sentence each entity
                             is in (0-indexed). Entities from the same sentence
                             share the same id.
        n_entities:          total number of entity nodes
        max_sentence_gap:    max sentence distance to connect (default: 1 = same
                             or adjacent sentences)

    Returns:
        adj_mask: (N, N) bool tensor (CPU) — True where edge exists
    """
    adj = torch.zeros(n_entities, n_entities, dtype=torch.bool)
    sent_ids = torch.tensor(entity_sentence_ids, dtype=torch.long)

    # Vectorized: |sent_i - sent_j| <= max_sentence_gap
    diff = (sent_ids.unsqueeze(0) - sent_ids.unsqueeze(1)).abs()  # (N, N)
    adj = diff <= max_sentence_gap  # includes self-loops (diff == 0)

    return adj


# ─── Modality adapters ──────────────────────────────────────────────────


class TextAdapter(nn.Module):
    """
    Wraps BERT's own word_embeddings table so the rest of the model can see
    a unified `inputs_embeds` tensor. Uses the EXACT same embedding parameters
    as the underlying BERT body, so numeric output is identical to calling
    `bert(input_ids=...)`.

    Why not just use bert.embeddings (full module = word + position + token
    type + LayerNorm + dropout)? Because BERT applies all of those internally
    when given inputs_embeds; if we pre-applied them in the adapter we'd
    double-apply LayerNorm. The adapter only does word lookup; everything
    else stays inside the backbone.
    """

    def __init__(self, bert_word_embeddings: nn.Embedding):
        super().__init__()
        # Reference (not copy) — adapter shares parameters with the backbone's
        # word embeddings. Both see the same gradients during training.
        self.word_embeddings = bert_word_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return {
            "inputs_embeds": self.word_embeddings(input_ids),
            "attention_mask": attention_mask,
        }


# ─── Backbone wrapper ────────────────────────────────────────────────────


class BertBackbone(nn.Module):
    """
    Thin wrapper around HF BertModel that always consumes inputs_embeds.
    Adapters are responsible for producing `(inputs_embeds, attention_mask)`;
    the backbone is modality-agnostic.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T, H)


# ─── Top-level extractor ─────────────────────────────────────────────────


class BertKGExtractor(nn.Module):
    """
    Pluggable KG extractor:
        - register one or more Adapters (one per modality)
        - call .encode(modality, **raw_inputs) to get contextualized hidden states
        - call .forward_ner(hidden) and .forward_re(hidden_b, word_ids_b, pairs)
          to get task-specific logits

    Stage 2 only registers a TextAdapter ("text"). Stage 3 will additionally
    register ImageAdapter, TableAdapter, etc. without touching this class.
    """

    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.1,
                 use_crf: bool = False,
                 num_bio_tags: int = None, num_relations: int = None,
                 num_entity_types: int = None, use_span_ner: bool = False,
                 max_span_width: int = 8, bio_enrich: str = "none",
                 boundary_reg: bool = False,
                 boundary_refine: bool = False):
        super().__init__()
        self.backbone = BertBackbone(model_name)
        hidden = self.backbone.hidden_size

        # Allow explicit override for multi-dataset support (train_multi.py).
        n_bio = num_bio_tags if num_bio_tags is not None else NUM_BIO_TAGS
        n_rel = num_relations if num_relations is not None else NUM_RELATIONS

        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden, n_bio)

        # Optional CRF layer for NER (stage2-012, negative result).
        self.use_crf = use_crf
        self.crf = None
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(n_bio, batch_first=True)

        # ── Span-based NER head (stage2-024) ──────────────────────────
        # Classifies candidate (start, end) spans as entity_type or NONE.
        # NONE = class 0; entity types = 1..num_entity_types.
        # The BIO head is still used for backward compat; span head is
        # an additional head that provides span-level predictions.
        self.use_span_ner = use_span_ner
        self.max_span_width = max_span_width
        self.bio_enrich = bio_enrich  # "none", "logits", or "probs"
        if use_span_ner:
            n_ent = num_entity_types if num_entity_types is not None else 6  # SciERC default
            # Span repr = [start; end; max_pool] → 3H input (+ n_bio if bio_enrich)
            span_in_dim = hidden * 3 + (n_bio if bio_enrich != "none" else 0)
            self.span_ner_head = nn.Linear(span_in_dim, n_ent + 1)
            self.span_width_emb = nn.Embedding(max_span_width, hidden)
            self.span_width_proj = nn.Linear(hidden, hidden * 3)  # project width emb to 3H

        # ── Boundary regression head (predicts Δ_start, Δ_end offsets) ───
        self.boundary_reg = boundary_reg
        if boundary_reg and use_span_ner:
            self.boundary_reg_head = nn.Linear(span_in_dim, 2)  # (Δ_start, Δ_end)

        # ── SRT-style boundary refinement (1D conv over span vectors) ───
        self.boundary_refine = boundary_refine
        if boundary_refine and use_span_ner:
            self.boundary_refine_conv = nn.Conv1d(
                span_in_dim, span_in_dim, kernel_size=3, padding=1)
            self.boundary_refine_norm = nn.LayerNorm(span_in_dim)

        # RE head — 2H concat + 2-layer MLP.
        # Optional 3H: if re_context_span=True, adds mean of tokens between the
        # head and tail spans as a third feature vector (inter-span context).
        self.re_context_span = False  # set via train_span.py --re-context-span
        re_in_dim = hidden * 2  # default; updated to 3H if re_context_span is set
        self.re_head = nn.Sequential(
            nn.Linear(re_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_rel),
        )
        self._re_hidden = hidden
        self._re_n_rel = n_rel
        self._dropout_p = dropout

        # ── GREP-style global relation prediction head (A13) ─────────────
        # Predicts which relation types are present in the document as a
        # multi-label auxiliary task over the [CLS] token representation.
        # This forces the encoder to recognize document-level relation
        # co-occurrence before per-pair classification.
        # Excludes NO_REL (class 0) — only real relation types (n_rel - 1).
        # Disabled by default; enabled when --global-rel-weight > 0.
        self.global_rel_head = None  # set to nn.Linear(hidden, n_rel-1) in train_span.py

        # ── Phase B3: ECRG-style Evidence GAT ─────────────────────────
        # 2-layer sparse multi-head attention over entity mention graph.
        # Enabled when --evidence-gat is passed to train_span.py.
        # max_sentence_gap controls which entities are connected (default: 1
        # = same or adjacent sentences only).
        self.evidence_gat = None   # set to EvidenceGAT(hidden) in train_span.py
        self.evidence_gat_gap = 1  # max sentence gap for adjacency

        # ── Pluggable adapters ─────────────────────────────────────────
        self.adapters = nn.ModuleDict()
        # Register the TextAdapter by default — Stage 2 only needs text.
        # Shares the backbone's input embedding parameters so the forward path
        # works across BERT, RoBERTa/XLM-R, DeBERTa, and other AutoModel classes.
        self.register_adapter("text", TextAdapter(self.backbone.bert.get_input_embeddings()))

    # ── Adapter registration API ────────────────────────────────────────

    def register_adapter(self, name: str, adapter: nn.Module) -> None:
        """
        Register a new modality adapter. Stage 3 use:

            extractor.register_adapter("image", ImageAdapter(vit_model, hidden))
            extractor.register_adapter("table", TableAdapter(...))

        The adapter must produce `inputs_embeds` of shape (B, T, hidden_size)
        and `attention_mask` of shape (B, T). No other constraint.
        """
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already registered. "
                             "Unregister first or use a different name.")
        self.adapters[name] = adapter

    def unregister_adapter(self, name: str) -> None:
        if name in self.adapters:
            del self.adapters[name]

    # ── Forward passes ──────────────────────────────────────────────────

    def encode(self, modality: str = "text", **raw_inputs):
        """
        Run the adapter for `modality` on `raw_inputs`, then the backbone.
        Returns (B, T, H) contextualized hidden states.
        """
        if modality not in self.adapters:
            raise KeyError(f"No adapter registered for modality '{modality}'. "
                           f"Available: {list(self.adapters.keys())}")
        adapter = self.adapters[modality]
        adapter_out = adapter(**raw_inputs)
        return self.backbone(
            inputs_embeds=adapter_out["inputs_embeds"],
            attention_mask=adapter_out["attention_mask"],
        )

    def forward_ner(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ner_head(self.dropout(hidden_states))  # (B, T, NUM_BIO_TAGS)

    @staticmethod
    def bio_guided_proposals(bio_logits_b: torch.Tensor, word_ids_b: list,
                             num_words: int, expand: int = 1,
                             conf_threshold: float = 0.0) -> list:
        """Decode BIO logits into span proposals for one example.

        Extracts B→I* sequences from argmax BIO predictions, then expands
        each by ±expand words to create boundary-variant candidates.

        Args:
            bio_logits_b: (T, NUM_BIO_TAGS) logits from BIO head.
            word_ids_b: list mapping subword positions to word indices.
            num_words: number of words in this example.
            expand: how many words to expand each proposal boundary.
            conf_threshold: minimum softmax confidence to keep a proposal.

        Returns:
            List of (start, end_inclusive) word-level spans.
        """
        preds = bio_logits_b.argmax(dim=-1).tolist()  # (T,)
        confs = bio_logits_b.softmax(dim=-1).max(dim=-1).values.tolist()

        # Map subword predictions to word-level by majority vote
        word_tags = [0] * num_words  # default O
        word_confs = [0.0] * num_words
        word_counts = [0] * num_words
        for tok_idx, wid in enumerate(word_ids_b):
            if wid is not None and 0 <= wid < num_words:
                # Take the first subword's prediction (B- tag is on first subword)
                if word_counts[wid] == 0:
                    word_tags[wid] = preds[tok_idx]
                    word_confs[wid] = confs[tok_idx]
                word_counts[wid] += 1

        # Extract B→I* spans
        bio_spans = []
        i = 0
        while i < num_words:
            tag_id = word_tags[i]
            tag_str = ID2BIO.get(tag_id, "O")
            if tag_str.startswith("B-"):
                start = i
                end = i
                avg_conf = word_confs[i]
                j = i + 1
                while j < num_words:
                    next_tag = ID2BIO.get(word_tags[j], "O")
                    if next_tag.startswith("I-"):
                        end = j
                        avg_conf += word_confs[j]
                        j += 1
                    else:
                        break
                avg_conf /= (end - start + 1)
                if avg_conf >= conf_threshold:
                    bio_spans.append((start, end))
                i = j
            else:
                i += 1

        # Expand each span by ±expand and generate boundary variants
        proposals = set()
        for (s, e) in bio_spans:
            # Original span
            proposals.add((s, e))
            # Boundary variants
            for ds in range(-expand, expand + 1):
                for de in range(-expand, expand + 1):
                    ns, ne = s + ds, e + de
                    if 0 <= ns <= ne < num_words:
                        proposals.add((ns, ne))

        return sorted(proposals)

    def _first_token_for_word(self, word_ids_b: list, word_idx: int) -> int:
        """Return the first token index mapped to word_idx."""
        for i, wid in enumerate(word_ids_b):
            if wid == word_idx:
                return i
        return 0

    def _last_token_for_word(self, word_ids_b: list, word_idx: int) -> int:
        """Return the last token index mapped to word_idx."""
        last = 0
        for i, wid in enumerate(word_ids_b):
            if wid == word_idx:
                last = i
        return last

    def span_repr_v2(self, hidden_states_b: torch.Tensor, word_ids_b: list,
                     span: tuple) -> torch.Tensor:
        """
        SpERT-style span representation:
          [start_token; end_token; max_pool(span_tokens)]
        Returns (3H,)
        """
        word_start, word_end = span
        start_tok = self._first_token_for_word(word_ids_b, word_start)
        end_tok = self._last_token_for_word(word_ids_b, word_end)
        start_vec = hidden_states_b[start_tok]
        end_vec = hidden_states_b[end_tok]
        pool_vec = self.span_repr(hidden_states_b, word_ids_b, span)
        return torch.cat([start_vec, end_vec, pool_vec], dim=-1)  # (3H,)

    def forward_span_ner(self, hidden_states_b: torch.Tensor, word_ids_b: list,
                         num_words: int, max_span_width: int = 8,
                         return_span_vecs: bool = False,
                         bio_logits_b: torch.Tensor = None,
                         bio_proposals: list = None):
        """
        Span-based NER v2: enumerate candidate spans, compute SpERT-style
        span representations [start; end; max_pool], classify as entity
        type or NONE.

        Args:
            bio_logits_b: (T, NUM_BIO_TAGS) optional BIO logits for this example.
                Used when bio_enrich != "none" to concatenate averaged BIO features
                per span into the span vector (STSN-inspired enrichment).
            bio_proposals: optional list of (start, end_inclusive) spans from BIO
                decoder. These are merged with exhaustive candidates, allowing
                spans wider than max_span_width to be considered.

        Returns:
            span_logits: (num_candidates, num_entity_types + 1)
            candidates:  list of (start, end_inclusive) word-level spans
            span_vecs:   (num_candidates, D) if return_span_vecs=True, else None
        """
        if not hasattr(self, "span_ner_head"):
            raise RuntimeError("span_ner_head not initialized. Use use_span_ner=True.")
        # Exhaustive enumeration up to max_span_width
        candidate_set = set()
        for s in range(num_words):
            for e in range(s, min(s + max_span_width, num_words)):
                candidate_set.add((s, e))
        # Merge BIO-guided proposals (can exceed max_span_width)
        if bio_proposals:
            for (s, e) in bio_proposals:
                if 0 <= s <= e < num_words:
                    candidate_set.add((s, e))
        candidates = sorted(candidate_set)
        if not candidates:
            empty = hidden_states_b.new_zeros((0, self.span_ner_head.out_features))
            return (empty, [], None) if return_span_vecs else (empty, [])

        span_vecs = []
        for (s, e) in candidates:
            vec = self.span_repr_v2(hidden_states_b, word_ids_b, (s, e))
            span_vecs.append(vec)
        span_vecs = torch.stack(span_vecs, dim=0)  # (num_candidates, 3H)

        # Add span width embedding (broadcast to 3H via linear projection)
        widths = torch.tensor(
            [min(e - s, self.span_width_emb.num_embeddings - 1) for (s, e) in candidates],
            device=hidden_states_b.device, dtype=torch.long,
        )
        width_vec = self.span_width_proj(self.span_width_emb(widths))  # (N, 3H)
        span_vecs = span_vecs + width_vec

        # STSN-inspired: concatenate BIO features per span
        if self.bio_enrich != "none" and bio_logits_b is not None:
            bio_feats = []
            for (s, e) in candidates:
                # Average BIO logits/probs over tokens in this span
                token_idx = [
                    i for i, wid in enumerate(word_ids_b)
                    if wid is not None and s <= wid <= e
                ]
                if token_idx:
                    span_bio = bio_logits_b[token_idx].mean(dim=0)  # (NUM_BIO_TAGS,)
                else:
                    span_bio = bio_logits_b.new_zeros(bio_logits_b.size(-1))
                if self.bio_enrich == "probs":
                    span_bio = F.softmax(span_bio, dim=-1)
                bio_feats.append(span_bio)
            bio_feats = torch.stack(bio_feats, dim=0)  # (N, NUM_BIO_TAGS)
            span_vecs = torch.cat([span_vecs, bio_feats], dim=-1)  # (N, 3H + NUM_BIO_TAGS)

        dropped = self.dropout(span_vecs)

        # SRT-style boundary refinement: 1D conv over span dimension
        if self.boundary_refine and hasattr(self, 'boundary_refine_conv'):
            # dropped: (N, D) -> Conv1d expects (1, D, N)
            x = dropped.unsqueeze(0).permute(0, 2, 1)
            x = self.boundary_refine_conv(x)
            x = x.permute(0, 2, 1).squeeze(0)  # back to (N, D)
            dropped = self.boundary_refine_norm(dropped + x)

        logits = self.span_ner_head(dropped)

        # Boundary regression: predict (Δ_start, Δ_end) for each candidate
        boundary_offsets = None
        if self.boundary_reg and hasattr(self, "boundary_reg_head"):
            boundary_offsets = self.boundary_reg_head(dropped)  # (N, 2)

        if return_span_vecs:
            if boundary_offsets is not None:
                return logits, candidates, span_vecs, boundary_offsets
            return logits, candidates, span_vecs
        if boundary_offsets is not None:
            return logits, candidates, boundary_offsets
        return logits, candidates

    def span_repr(self, hidden_states: torch.Tensor, word_ids_list: list, span: tuple) -> torch.Tensor:
        """
        Extract span representation by max-pooling subword embeddings whose
        word_ids fall within [span_start, span_end] (both inclusive).
        Returns (H,)
        """
        word_start, word_end = span
        token_idx = [
            i for i, wid in enumerate(word_ids_list)
            if wid is not None and word_start <= wid <= word_end
        ]
        if not token_idx:
            return hidden_states.new_zeros(hidden_states.size(-1))
        return hidden_states[token_idx].max(dim=0).values

    def _between_span_repr(self, hidden_states_b: torch.Tensor, word_ids_b: list,
                            span_a: tuple, span_b: tuple) -> torch.Tensor:
        """Mean-pool tokens strictly between two spans (exclusive of span endpoints).
        Returns (H,). Returns zero vector if spans are adjacent or overlapping.
        """
        lo = min(span_a[1], span_b[1]) + 1  # word after earlier span end
        hi = max(span_a[0], span_b[0]) - 1  # word before later span start
        if lo > hi:
            return hidden_states_b.new_zeros(hidden_states_b.size(-1))
        token_idx = [i for i, wid in enumerate(word_ids_b) if wid is not None and lo <= wid <= hi]
        if not token_idx:
            return hidden_states_b.new_zeros(hidden_states_b.size(-1))
        return hidden_states_b[token_idx].mean(dim=0)

    def forward_re(self, hidden_states_b: torch.Tensor, word_ids_b: list, pairs: list,
                   return_feats: bool = False) -> torch.Tensor:
        """
        For one example in the batch:
            hidden_states_b: (T, H)
            word_ids_b:      list[int|None] length T
            pairs:           list of ((hs, he), (ts, te))
        Returns: (num_pairs, NUM_RELATIONS)

        If self.re_context_span is True, the pair representation is 3H:
        [head_vec; tail_vec; between_span_mean] instead of 2H [head_vec; tail_vec].

        If return_feats is True, also returns the pre-logits pair representation
        (num_pairs, 2H or 3H) for pair-level contrastive loss (B-TW.11 PCL).
        """
        in_dim = hidden_states_b.size(-1) * (3 if self.re_context_span else 2)
        if not pairs:
            empty_logits = hidden_states_b.new_zeros((0, NUM_RELATIONS))
            if return_feats:
                return empty_logits, hidden_states_b.new_zeros((0, in_dim))
            return empty_logits
        feats = []
        for (hs, he), (ts, te) in pairs:
            head_vec = self.span_repr(hidden_states_b, word_ids_b, (hs, he))
            tail_vec = self.span_repr(hidden_states_b, word_ids_b, (ts, te))
            if self.re_context_span:
                ctx_vec = self._between_span_repr(
                    hidden_states_b, word_ids_b, (hs, he), (ts, te))
                feats.append(torch.cat([head_vec, tail_vec, ctx_vec], dim=-1))
            else:
                feats.append(torch.cat([head_vec, tail_vec], dim=-1))
        feats = torch.stack(feats, dim=0)  # (num_pairs, 2H or 3H)
        logits = self.re_head(self.dropout(feats))
        if return_feats:
            return logits, feats
        return logits

    def forward_re_with_graph(
        self,
        hidden_states_list: list,
        word_ids_list: list,
        entity_spans_by_sent: list,
        pairs_by_sent: list,
        entity_sent_ids: list,
    ) -> list:
        """
        Phase B3: RE prediction with ECRG-style evidence graph enrichment.

        Encodes entity spans from each sentence independently, stacks them
        into a node matrix, runs EvidenceGAT, then uses enriched representations
        for RE head predictions.

        Args:
            hidden_states_list:  list of (T_i, H) tensors — one per sentence
            word_ids_list:       list of word_ids lists — one per sentence
            entity_spans_by_sent: list of lists — entity spans per sentence.
                Each element: list of (start, end_inclusive) word-level spans.
            pairs_by_sent:       list of lists — RE pairs per sentence.
                Each element: list of ((hs,he), (ts,te), h_node_idx, t_node_idx)
                where h_node_idx and t_node_idx are indices into the global
                entity node list.
            entity_sent_ids:     list[int] — sentence id for each entity node.

        Returns:
            list of (num_pairs_in_sent, NUM_RELATIONS) tensors — one per sentence.
            Empty list if no entities.

        Notes:
            - If evidence_gat is None (not enabled), falls back to sentence-local
              span representations (equivalent to current forward_re behavior).
            - entity_spans_by_sent must be non-empty for graph construction.
        """
        if not entity_spans_by_sent or not any(entity_spans_by_sent):
            return []

        # Step 1: Compute initial span representations for all entity nodes
        all_span_reps = []
        for sent_idx, (hidden_b, wids, spans) in enumerate(
                zip(hidden_states_list, word_ids_list, entity_spans_by_sent)):
            for span in spans:
                all_span_reps.append(self.span_repr(hidden_b, wids, span))

        if not all_span_reps:
            return []

        # Step 2: Stack into entity node matrix (N, H)
        node_reps = torch.stack(all_span_reps, dim=0)  # (N, H)

        # Step 3: Run EvidenceGAT if enabled
        if self.evidence_gat is not None and len(node_reps) > 1:
            adj_mask = build_evidence_graph(
                entity_sent_ids, len(node_reps),
                max_sentence_gap=self.evidence_gat_gap,
            ).to(node_reps.device)
            node_reps = self.evidence_gat(node_reps, adj_mask)

        # Step 4: RE head using enriched node representations
        results = []
        for sent_pairs in pairs_by_sent:
            if not sent_pairs:
                results.append(None)
                continue
            feats = []
            for (hs, he), (ts, te), h_idx, t_idx in sent_pairs:
                head_vec = node_reps[h_idx]
                tail_vec = node_reps[t_idx]
                if self.re_context_span:
                    # Use the sentence's hidden states for between-span context
                    # Determine which sentence this pair belongs to
                    sent_idx = entity_sent_ids[h_idx]
                    ctx_vec = self._between_span_repr(
                        hidden_states_list[sent_idx],
                        word_ids_list[sent_idx],
                        (hs, he), (ts, te),
                    )
                    feats.append(torch.cat([head_vec, tail_vec, ctx_vec], dim=-1))
                else:
                    feats.append(torch.cat([head_vec, tail_vec], dim=-1))
            feats_t = torch.stack(feats, dim=0)
            results.append(self.re_head(self.dropout(feats_t)))

        return results


# ─── Loss ────────────────────────────────────────────────────────────────


def compute_loss(model, batch, device, re_weight: float = 1.0):
    """
    Compute Stage 2 loss = NER CE + re_weight · RE CE.

    NER: per-token cross entropy (ignore -100).
    RE:  Stage 2-004 fix — enumerate ALL ordered pairs of gold entity spans.
         Pairs that have an annotated relation get the rel id (1..7).
         Pairs without an annotated relation get NO_REL_ID = 0.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    ner_labels = batch["ner_labels"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]

    # Stage 2: text modality. Stage 3 will pass modality="image" or similar
    # from a different training loop.
    hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

    # NER loss
    ner_logits = model.forward_ner(hidden)
    if model.use_crf and model.crf is not None:
        # CRF requires: (1) mask first timestep = True, (2) no -100 in tags.
        # Use attention_mask as the CRF mask (first token [CLS] is always 1).
        # Replace -100 labels with O-tag (0) — CRF mask will handle ignoring
        # special tokens; the O-tag assignment is just to keep tags in-range.
        crf_mask = attention_mask.bool()  # (B, T)
        crf_labels = ner_labels.clone()
        crf_labels[ner_labels == -100] = 0  # O tag
        ner_loss = -model.crf(ner_logits, crf_labels, mask=crf_mask, reduction="mean")
    else:
        ner_loss = F.cross_entropy(
            ner_logits.view(-1, ner_logits.size(-1)),
            ner_labels.view(-1),
            ignore_index=-100,
        )

    # RE loss — every ordered pair of gold spans, NO_REL for unannotated pairs
    re_losses = []
    for b_idx in range(len(gold_entities_list)):
        ents = gold_entities_list[b_idx]
        rels = gold_relations_list[b_idx]
        if len(ents) < 2:
            continue

        rel_lookup = {(h, t): rid for (h, t, rid) in rels}
        spans = [(s, e) for (s, e, _) in ents]
        pairs = []
        targets = []
        for h in spans:
            for t in spans:
                if h == t:
                    continue
                pairs.append((h, t))
                targets.append(rel_lookup.get((h, t), NO_REL_ID))

        if not pairs:
            continue

        targets_t = torch.tensor(targets, device=device, dtype=torch.long)
        re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
        re_losses.append(F.cross_entropy(re_logits, targets_t))

    if re_losses:
        re_loss = torch.stack(re_losses).mean()
    else:
        re_loss = ner_loss.new_tensor(0.0)

    total = ner_loss + re_weight * re_loss
    return total, ner_loss.detach(), re_loss.detach(), ner_logits
