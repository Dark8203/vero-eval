"""
Semantic chunking utilities.

What this module provides:
- A drop-in replacement for LangChain's `RecursiveCharacterTextSplitter.split_documents(...)`.
- Produces chunked `Document` objects that preserve original metadata and add `start_index`, `end_index`, and `token_count` for provenance.

Why use semantic chunking:
- Fixed-size chunks can split thoughts and reduce retrieval quality.
- Semantic chunking keeps related sentences together until the topic drifts or a size limit is reached, which usually improves RAG recall.

How to use it:
- Replace your `RecursiveCharacterTextSplitter(...)` + `split_documents(docs)` calls with `semantically_chunk_documents(docs, ...)`.
- Inputs and outputs match the recursive splitter (list of `Document` in, list of `Document` out), so downstream code need not change.

High-level approach:
- Split text into sentences (lightweight regex; swap in spaCy/NLTK if desired).
- Encode sentences with a compact SentenceTransformer model.
- Greedily grow a chunk by adding the next sentence if it is semantically similar (cosine similarity above a threshold) or until `min_tokens` is reached.
- Flush a chunk when it becomes dissimilar or would exceed `max_tokens`. Optionally keep a small sentence overlap to boost recall.
"""

import re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    # Small, fast embedding model family suitable for local semantic grouping
    from sentence_transformers import SentenceTransformer
except Exception as e:  
    SentenceTransformer = None

try:
    # Used only for token length accounting; falls back to whitespace tokens if unavailable
    from transformers import AutoTokenizer
except Exception: 
    AutoTokenizer = None

try:
    # LangChain core Document (compatible with .model_dump() in recent versions)
    from langchain_core.documents import Document
except Exception: 
    class Document: 
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def model_dump(self):  # minimal compatibility for downstream usage
            return {"page_content": self.page_content, "metadata": self.metadata}


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter using simple, easy-to-read regex rules.

    What this function provides:
    - A lightweight, dependency-minimal splitter that returns sentence-like
      fragments from a single text string.

    Why use this splitter:
    - Keeps chunking dependency-light and fast.
    - Preserves sentence boundaries so downstream chunking groups natural units
      of thought.

    How to use it:
    - Call with a single document `page_content` string prior to embedding or
      chunk construction. Swap in spaCy/NLTK for higher accuracy if required.

    High-level approach:
    - Normalize whitespace, split on terminal punctuation followed by space,
      and merge very short tails (likely abbrev./noise) back into the
      previous fragment.
    - Returns a list of trimmed, non-empty sentence strings.
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

    What this function provides:
    - A list of (start, end) character offsets for each sentence discovered.

    Why use spans:
    - Enables provenance by attaching `start_index`/`end_index` to chunks so
      consumers can trace answers back to exact offsets in the source.

    How it works:
    - Advances a cursor and searches for each sentence sequentially, returning
      None for any sentence that cannot be re-located (rare, e.g., whitespace
      normalization differences).

    High-level approach:
    - Robust to repeated substrings earlier in the text because the search
      starts at the previous sentence end.
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
    """Load embedding and tokenizer components, documented like module top.

    What this function provides:
    - Instantiates a SentenceTransformer embedding model and, when available,
      an AutoTokenizer for token counting.

    Why load once:
    - Model loading can be expensive; doing it once per batch of documents
      avoids repeated overhead.

    How to use it:
    - Called internally by `semantically_chunk_documents` before processing all
      documents. Raises a clear ImportError if sentence-transformers is absent.

    High-level approach:
    - Try to create both model and tokenizer; gracefully degrade tokenizer to
      None if it cannot be instantiated for the chosen model.
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

    What this function provides:
    - An approximate token count for a text using a tokenizer when available,
      or a fallback word-based estimate.

    Why token counting:
    - Enforces min/max token constraints for chunks to improve retrieval and
      avoid excessive model truncation.

    How it behaves:
    - If tokenizer is None, returns max(1, word_count). Otherwise uses the
      tokenizer.encode(..., add_special_tokens=False) length.

    High-level approach:
    - Provides a pragmatic, dependency-tolerant token length estimation used by
      the greedy chunk growth algorithm.
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

    What this function provides:
    - Groups adjacent sentences into semantically coherent chunks while
      respecting token-based size bounds and optional sentence overlap.

    Why use this algorithm:
    - Produces chunks that keep related sentences together, improving retrieval
      relevance versus fixed-size chopping.

    How to use it:
    - Called per-document by `semantically_chunk_documents`. Parameters mirror
      common splitter knobs (min/max tokens, similarity threshold, overlap).

    High-level approach:
    - Split text into sentences, compute normalized sentence embeddings, then
      greedily grow a chunk by adding the next sentence if similarity to the
      running centroid is above threshold or the chunk hasn't reached
      `min_tokens`.
    - Flush the chunk when adding would exceed `max_tokens` or similarity drops
      (and `min_tokens` is satisfied). Optionally carry an overlap of the last
      sentences into the next chunk to boost recall.
    - Returns a list of dicts containing chunk text, sentence indices, char
      span, and token count for each chunk.
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

    What this function provides:
    - A drop-in replacement for RecursiveCharacterTextSplitter.split_documents
      that returns chunked Document objects with provenance metadata.

    Why use this function:
    - Keeps the same inputs/outputs as the recursive splitter so downstream
      code remains unchanged while gaining semantic chunking.

    How to use it:
    - Call with a list of Documents. The function loads models once, chunks
      each document semantically, and returns new Document objects preserving
      original metadata plus `start_index`, `end_index`, and `token_count`.

    High-level approach:
    - Load embedding and tokenizer, run the per-document semantic chunker, and
      wrap chunk dicts back into Document instances with augmented metadata.
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
                md["end_index"] = int(p["end_char"])  # end offset
            md["token_count"] = int(p.get("token_count", 0))
            out.append(Document(page_content=p["text"], metadata=md))
    return out


# === DataFrame and clustering utilities ===

try:  # Optional; enables density-based clustering without choosing k
    import hdbscan
except Exception:
    hdbscan = None

try:
    # Fallback clustering if HDBSCAN is unavailable
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None


def chunks_to_df(
    chunks: List[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame with chunk text, lengths, and embeddings.

    What this function provides:
    - A pandas DataFrame containing chunk text, lengths, embeddings, and
      preserved metadata for downstream analysis or clustering.

    Why use this:
    - Useful for auditing chunk quality, visualizing embeddings, and feeding
      clustering or evaluation pipelines.

    How to use it:
    - Pass the output of `semantically_chunk_documents`. The function computes
      token/char lengths, batch-encodes chunk embeddings, and returns a DataFrame
      with serializable embedding lists.

    High-level approach:
    - Load models, compute token/char lengths, batch-encode texts (with optional
      normalization), and assemble a DataFrame including chunk IDs derived from
    metadata when possible.
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

    What this function provides:
    - Groups semantically similar chunks and annotates the DataFrame with a
      `cluster_id` column, filtering out noise and undersized clusters.

    Why use clustering:
    - Helps discover redundancy, related content, and clusters suitable for
      multi-source evaluation question generation.

    How to use it:
    - Supply a DataFrame produced by `chunks_to_df` that contains an
      'embedding' column. Choose `method='hdbscan'` when available for density
      clustering, or fall back to AgglomerativeClustering.

    High-level approach:
    - Stack embeddings into X, run the selected clustering algorithm, attach
      labels, filter noise/singletons, and optionally enforce token-length
      diversity within clusters before returning the concatenated results.
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
