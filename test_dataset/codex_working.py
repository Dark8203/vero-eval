"""
Semantic chunking utilities.

What this module does:
- Provides a drop-in replacement for LangChain's
  `RecursiveCharacterTextSplitter.split_documents(...)`, but with semantic
  grouping of sentences instead of purely size-based splitting.
- Returns chunked `Document` objects while preserving original metadata and
  adding `start_index`/`end_index` (character offsets) and `token_count` for
  provenance and downstream evaluation.

Why this exists (why you need it):
- Fixed-size chunks can break context in the middle of a thought, which harms
  retrieval quality. Semantic chunking keeps related sentences together until
  they drift semantically or hit a size cap, typically improving RAG recall.

Where it fits in your code:
- Replace the two lines where you instantiate
  `RecursiveCharacterTextSplitter(...)` and call `split_documents(docs)` with
  the `semantically_chunk_documents(docs, ...)` function from this module.
  Inputs and outputs mirror the recursive splitter usage (list of `Document`
  in, list of chunked `Document` out), so the rest of your pipeline need not
  change.

High-level approach:
- Split each document into sentences (lightweight regex; you can swap in spaCy/NLTK).
- Encode sentences with a compact SentenceTransformer model.
- Greedily grow a chunk by appending the next sentence if it is similar
  (cosine similarity >= threshold) or until reaching `min_tokens`.
- Respect `max_tokens` bound; when exceeded or dissimilar, flush the chunk.
- Optionally carry over a small sentence overlap to the next chunk to boost recall.
"""

import re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    # Small, fast embedding model family suitable for local semantic grouping
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    # Used only for token length accounting; falls back to whitespace tokens if unavailable
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

try:
    # LangChain core Document (compatible with .model_dump() in recent versions)
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    # Fallback shim if import path changes; provides the minimal shape we need
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def model_dump(self):  # minimal compatibility for downstream usage
            return {"page_content": self.page_content, "metadata": self.metadata}


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter using regex heuristics.

    Why this exists: Chunking at sentence boundaries reduces semantic
    discontinuities and yields more coherent chunks than cutting by raw
    characters.

    Where it fits: First step of semantic chunking; upstream loaders provide a
    single `page_content` string, which we segment here before embedding.

    For production quality, consider spaCy or NLTK's `sent_tokenize`.
    This implementation aims to be dependency-light while performing well on
    common prose.
    """
    text = text.strip()
    if not text:
        return []
    # Split on end punctuation followed by whitespace, after normalizing newlines
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\r", " ").replace("\n", " ").strip())
    # Merge tiny fragments (e.g., caused by abbreviations) back into the previous sentence
    merged: List[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        else:
            if len(p) < 3:  # very short tail likely not a true sentence
                buf = f"{buf} {p}"
            else:
                merged.append(buf)
                buf = p
    if buf:
        merged.append(buf)
    # Return trimmed sentences, dropping any empty strings
    return [s.strip() for s in merged if s.strip()]


def _sentence_spans(text: str, sentences: List[str]) -> List[Optional[tuple]]:
    """Compute (start, end) character spans of each sentence.

    Why this exists: We add `start_index`/`end_index` to chunk metadata for
    provenance and to mimic `add_start_index=True` behavior from the recursive
    splitter. Evaluators can trace answers back to source offsets.

    Where it fits: Used after sentence splitting and before forming chunks so
    we can aggregate per-sentence offsets to per-chunk offsets.

    We move a cursor forward to find sentences in order, which is robust to
    duplicate substrings earlier in the text. Returns None for sentences we
    fail to re-locate (rare, but possible if the source has altered whitespace).
    """
    spans = []
    cursor = 0
    for s in sentences:
        idx = text.find(s, cursor)
        if idx == -1:
            spans.append(None)
        else:
            spans.append((idx, idx + len(s)))
            cursor = idx + len(s)
    return spans


def _load_models(model_name: str):
    """Load embedding model and tokenizer.

    Why this exists: We need sentence embeddings to measure semantic cohesion
    (via cosine similarity) and a tokenizer to enforce min/max token bounds.

    Where it fits: Called once by `semantically_chunk_documents` before
    processing all documents to avoid repeated model loads.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for semantic chunking")
    emb_model = SentenceTransformer(model_name)
    tok = None
    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            tok = None
    return emb_model, tok


def _count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens using the provided tokenizer; fall back to word count.

    Why this exists: RAG retrievers often have budgeted context size; keeping
    chunks within `max_tokens` improves retrieval quality and reduces model
    truncation.

    Where it fits: Used during greedy chunk growth to decide when to flush the
    current chunk and start a new one.

    We avoid adding special tokens to approximate content length constraints.
    """
    if tokenizer is None or not text:
        return max(1, int(len(text.split())))
    return len(tokenizer.encode(text, add_special_tokens=False))


def _semantic_chunk_text(
    text: str,
    emb_model,
    tokenizer=None,
    min_tokens: int = 80,
    max_tokens: int = 350,
    similarity_threshold: float = 0.6,
    overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """Greedy semantic chunking within token bounds.

    Why this exists: This is the core algorithm that groups adjacent sentences
    into semantically consistent chunks sized for retrieval.

    Where it fits: Called per document by `semantically_chunk_documents`; its
    output is then wrapped back into LangChain `Document` objects for downstream
    use.

    - Keep adding the next sentence if it is similar to the running chunk
      centroid OR the current chunk hasn't reached `min_tokens`.
    - Flush when adding would exceed `max_tokens` or when similarity drops
      below `similarity_threshold` after `min_tokens` is satisfied.
    - Maintain an optional sentence overlap to improve retrieval recall.
    Returns a list of dictionaries that carry chunk text, positions and size.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Normalize embeddings so cosine similarity becomes a fast dot product
    sent_embs = emb_model.encode(sentences, normalize_embeddings=True)
    sent_spans = _sentence_spans(text, sentences)

    chunks: List[Dict[str, Any]] = []
    cur_idxs: List[int] = []
    cur_vec = None  # running centroid vector of the current chunk
    cur_tokens = 0  # running token count of the current chunk

    def flush_chunk():
        """Commit the current sentence window as a chunk and prepare overlap."""
        nonlocal cur_idxs, cur_vec, cur_tokens
        if not cur_idxs:
            return
        # Materialize the chunk text from sentence indices
        chunk_sents = [sentences[i] for i in cur_idxs]
        chunk_text = " ".join(chunk_sents).strip()
        # Derive approximate character span within the source text
        starts = [sent_spans[i][0] for i in cur_idxs if sent_spans[i] is not None]
        ends = [sent_spans[i][1] for i in cur_idxs if sent_spans[i] is not None]
        start_char = min(starts) if starts else None
        end_char = max(ends) if ends else None
        chunks.append(
            {
                "text": chunk_text,
                "sent_indices": cur_idxs.copy(),
                "start_char": start_char,
                "end_char": end_char,
                "token_count": cur_tokens,
            }
        )
        # Optionally carry over the last few sentences to the next chunk
        if overlap_sentences > 0:
            overlap = cur_idxs[-overlap_sentences:]
            cur_idxs = overlap.copy()
            if cur_idxs:
                embs = np.vstack([sent_embs[i] for i in cur_idxs])
                vec = embs.mean(axis=0)
                cur_vec = vec / (np.linalg.norm(vec) + 1e-9)
                cur_tokens = _count_tokens(" ".join(sentences[i] for i in cur_idxs), tokenizer)
            else:
                cur_vec, cur_tokens = None, 0
        else:
            cur_idxs, cur_vec, cur_tokens = [], None, 0

    for i, emb in enumerate(sent_embs):
        s = sentences[i]
        s_tokens = _count_tokens(s, tokenizer)

        # Initialize a new chunk if we don't have one yet
        if not cur_idxs:
            cur_idxs = [i]
            cur_vec = emb
            cur_tokens = s_tokens
            continue

        predicted_tokens = cur_tokens + s_tokens
        sim = float(np.dot(cur_vec, emb))  # cosine similarity (embeddings are normalized)
        force_fill = cur_tokens < min_tokens
        can_add = (sim >= similarity_threshold or force_fill) and (predicted_tokens <= max_tokens)

        if can_add:
            # Accept the sentence and update centroid + token count
            cur_idxs.append(i)
            embs = np.vstack([sent_embs[j] for j in cur_idxs])
            vec = embs.mean(axis=0)
            cur_vec = vec / (np.linalg.norm(vec) + 1e-9)
            cur_tokens = predicted_tokens
        else:
            # Commit current chunk and start a fresh one with this sentence
            flush_chunk()
            cur_idxs = [i]
            cur_vec = emb
            cur_tokens = s_tokens

    # Commit any remaining sentences as the final chunk
    flush_chunk()
    return chunks


def semantically_chunk_documents(
    docs: List[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_tokens: int = 80,
    max_tokens: int = 350,
    similarity_threshold: float = 0.6,
    overlap_sentences: int = 1,
) -> List[Document]:
    """Replace RecursiveCharacterTextSplitter with semantic chunking.

    Why this exists: Provides a plug-and-play API that mirrors the recursive
    splitter entry point, so you can swap implementations with minimal code
    changes while gaining semantic chunking benefits.

    Where it fits: Use directly where you currently call
    `text_splitter.split_documents(docs)`. It takes the same `docs` input and
    returns a list of chunked `Document`s.

    Parameters mirror typical splitter knobs for easy substitution.
    Returns a list of LangChain `Document` chunks that preserve the original
    metadata and add `start_index`, `end_index`, and `token_count`.
    """
    emb_model, tokenizer = _load_models(model_name)
    out: List[Document] = []
    for d in docs:
        text = getattr(d, "page_content", "")
        meta = dict(getattr(d, "metadata", {}) or {})
        pieces = _semantic_chunk_text(
            text=text,
            emb_model=emb_model,
            tokenizer=tokenizer,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            similarity_threshold=similarity_threshold,
            overlap_sentences=overlap_sentences,
        )
        for p in pieces:
            md = meta.copy()
            # Align with add_start_index behavior from RecursiveCharacterTextSplitter
            if p.get("start_char") is not None:
                md["start_index"] = int(p["start_char"])  # start offset within source doc
            if p.get("end_char") is not None:
                md["end_index"] = int(p["end_char"])  # end offset (convenience)
            md["token_count"] = int(p.get("token_count", 0))
            out.append(Document(page_content=p["text"], metadata=md))
    return out


# === DataFrame and clustering utilities ===

try:  # Optional; enables density-based clustering without choosing k
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None  # type: ignore

try:
    # Fallback clustering if HDBSCAN is unavailable
    from sklearn.cluster import AgglomerativeClustering
except Exception:  # pragma: no cover
    AgglomerativeClustering = None  # type: ignore


def chunks_to_df(
    chunks: List[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame with chunk text, lengths, and embeddings.

    Why this exists: You often want to audit, visualize, or cluster chunks to
    verify semantic cohesion and tune chunking parameters. A tabular view with
    embeddings is a convenient starting point for such analysis.

    Where it fits: Use immediately after chunking (e.g., the output of
    `semantically_chunk_documents`). You can pass this DataFrame to
    `cluster_chunks_df` to group similar-context chunks.

    Columns produced:
    - chunk_id: derived from metadata where possible, else an index.
    - text: chunk content.
    - char_len: character length of the chunk.
    - token_len: token length (approximate if tokenizer not available).
    - embedding: list[float] embedding vector for the chunk.
    - metadata: original metadata dict, preserved for traceability.
    """
    emb_model, tokenizer = _load_models(model_name)

    texts: List[str] = [getattr(c, "page_content", "") for c in chunks]
    metas: List[Dict[str, Any]] = [dict(getattr(c, "metadata", {}) or {}) for c in chunks]

    # Prepare chunk IDs from metadata if available for better provenance
    chunk_ids: List[str] = []
    for i, md in enumerate(metas):
        # Prefer explicit IDs, else derive from offsets if present, else index
        if "chunk_id" in md:
            chunk_ids.append(str(md["chunk_id"]))
        elif "source" in md and "start_index" in md:
            chunk_ids.append(f"{md.get('source')}:{md.get('start_index')}")
        elif "start_index" in md:
            chunk_ids.append(f"offset:{md.get('start_index')}")
        else:
            chunk_ids.append(f"idx:{i}")

    # Compute token lengths (uses model tokenizer if available)
    token_lens = [_count_tokens(t, tokenizer) for t in texts]
    char_lens = [len(t) for t in texts]

    # Compute embeddings in a batch for efficiency
    embs = emb_model.encode(texts, normalize_embeddings=normalize_embeddings)
    # Ensure Python-native lists for DataFrame serialization
    emb_lists = [np.asarray(e, dtype=float).tolist() for e in embs]

    df = pd.DataFrame(
        {
            "chunk_id": chunk_ids,
            "text": texts,
            "char_len": char_lens,
            "token_len": token_lens,
            "embedding": emb_lists,
            "metadata": metas,
        }
    )
    return df


def cluster_chunks_df(
    df: pd.DataFrame,
    method: str = "hdbscan",
    min_cluster_size: int = 2,
    metric: str = "cosine",
    # For Agglomerative fallback
    distance_threshold: Optional[float] = 0.6,
    n_clusters: Optional[int] = None,
    # Length diversity constraint
    require_length_diversity: bool = True,
    min_length_diff: int = 10,
) -> pd.DataFrame:
    """Cluster chunks by semantic similarity and return df with `cluster_id`.

    Why this exists: To discover groups of semantically similar chunks so you
    can sample evaluation questions that require synthesis across multiple
    places in the corpus, or to detect redundancy.

    Where it fits: Use right after `chunks_to_df`. The result can be filtered
    to clusters (>=2 items) and then consumed by your QA generation scripts.

    Behavior:
    - Default uses HDBSCAN (if installed) for density-based clustering with
      variable cluster counts and `min_cluster_size` control.
    - Falls back to Agglomerative Clustering if HDBSCAN is unavailable. You
      may set `distance_threshold` or `n_clusters` for the fallback.
    - Filters clusters to ensure each has at least 2 items and, if enabled,
      at least two distinct lengths differing by `min_length_diff` tokens.
    """
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column.")

    X = np.vstack(df["embedding"].apply(np.asarray).to_list())

    labels: Optional[np.ndarray] = None
    if method.lower() == "hdbscan" and hdbscan is not None:
        # HDBSCAN expects a distance metric; metric="cosine" is common for embeddings
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X)
    else:
        if AgglomerativeClustering is None:
            raise ImportError("Neither hdbscan nor sklearn AgglomerativeClustering is available.")
        # If distance_threshold is set, sklearn will determine the number of clusters
        # Handle sklearn API differences across versions:
        # - Newer versions use `metric` and have removed `affinity`.
        # - Older versions expect `affinity` and may not accept `metric`.
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage="average",
                distance_threshold=distance_threshold if n_clusters is None else None,
            )
        except TypeError:
            # Fall back to legacy signature that uses `affinity`
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=metric,
                linkage="average",
                distance_threshold=distance_threshold if n_clusters is None else None,
            )
        labels = clusterer.fit_predict(X)

    df = df.copy()
    df["cluster_id"] = labels

    # Drop noise cluster (-1 in HDBSCAN) and singletons
    valid = []
    for cid, grp in df.groupby("cluster_id"):
        if cid == -1:  # noise in HDBSCAN
            continue
        if len(grp) < max(2, min_cluster_size):
            continue
        if require_length_diversity:
            # ensure at least two distinct token lengths with minimum separation
            uniq = np.unique(grp["token_len"].values)
            if len(uniq) < 2:
                continue
            # Check if any pair differs by at least min_length_diff
            ok = False
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    if abs(int(uniq[i]) - int(uniq[j])) >= min_length_diff:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue
        valid.append(grp)

    if not valid:
        # Return empty with the same columns if no clusters meet criteria
        return df.iloc[0:0].copy()

    return pd.concat(valid, axis=0).reset_index(drop=True)
