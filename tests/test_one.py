import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from vero.test_dataset_generator import chunking_utilities
from vero.evaluator import evaluator as evaluator_module
from vero.metrics.g_eval import g_eval as geval_module


def test_split_sentences_basic():
    text = "Hello world! This is a test. Is it working? Yes."
    sents = chunking_utilities._split_sentences(text)
    assert isinstance(sents, list)
    # Ensure sentences are split on terminal punctuation
    assert "Hello world!" in sents[0]
    assert any("Is it working?" in s for s in sents)
test_split_sentences_basic()

def test_sentence_spans_roundtrip():
    text = "A short sentence. Another one!"
    sents = chunking_utilities._split_sentences(text)
    spans = chunking_utilities._sentence_spans(text, sents)
    # spans should correspond to substrings at offsets
    for sent, span in zip(sents, spans):
        assert span is not None
        start, end = span
        assert text[start:end] == sent


def test_count_tokens_no_tokenizer():
    text = "one two three four"
    # When tokenizer None, falls back to word count but at least 1
    cnt = chunking_utilities._count_tokens(text, tokenizer=None)
    assert cnt == 4


def test_count_tokens_with_fake_tokenizer():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            # simulate subword tokens length equal to number of characters modulo for variety
            return list(range(max(1, len(text.split()))))
    tok = FakeTokenizer()
    cnt = chunking_utilities._count_tokens("a b c d e", tokenizer=tok)
    assert cnt == 5


def test_semantic_chunk_text_simple():
    # Fake embedding model that returns normalized vectors where adjacent sentences are similar
    class FakeEmb:
        def encode(self, sentences, normalize_embeddings=True):
            embs = []
            for i, _ in enumerate(sentences):
                # alternate two nearby vectors so similarity between neighbors is high
                if i % 2 == 0:
                    v = np.array([1.0, 0.0])
                else:
                    v = np.array([0.99, 0.1])
                if normalize_embeddings:
                    v = v / (np.linalg.norm(v) + 1e-9)
                embs.append(v)
            return np.vstack(embs)

    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

    text = "First sentence. Second sentence. Third sentence."
    chunks = chunking_utilities._semantic_chunk_text(
        text=text,
        emb_model=FakeEmb(),
        tokenizer=FakeTok(),
        min_tokens=1,
        max_tokens=1000,
        similarity_threshold=0.0,  # make it permissive
        overlap_sentences=0,
    )
    assert isinstance(chunks, list)
    # With permissive threshold and small limits, expect at least one chunk containing text
    assert any("First" in c["text"] for c in chunks)


def test_semantically_chunk_documents_with_patched_loader():
    # Patch _load_models to return a fake model and tokenizer so semantically_chunk_documents can run
    class FakeEmb:
        def encode(self, sentences, normalize_embeddings=True):
            # each sentence -> small deterministic vector
            arr = []
            for i, _ in enumerate(sentences):
                v = np.array([1.0, 0.0])
                if normalize_embeddings:
                    v = v / (np.linalg.norm(v) + 1e-9)
                arr.append(v)
            return np.vstack(arr)

    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return list(range(max(1, len(text.split()))))

    # Create a minimal Document-like object compatible with the module
    Doc = chunking_utilities.Document

    docs = [Doc(page_content="Alpha one. Alpha two.", metadata={"source": "src1"})]

    with patch.object(chunking_utilities, "_load_models", return_value=(FakeEmb(), FakeTok())):
        out = chunking_utilities.semantically_chunk_documents(docs, min_tokens=1, max_tokens=1000, similarity_threshold=0.0, overlap_sentences=0)
    assert isinstance(out, list)
    assert all(hasattr(d, "page_content") for d in out)
    # Check metadata includes token_count and start/end indices when available
    for d in out:
        md = getattr(d, "metadata", {})
        assert "token_count" in md


def test_chunks_to_df_with_patched_load_models():
    class FakeEmb:
        def encode(self, texts, normalize_embeddings=True):
            # return one-d vector per text for simplicity
            return np.array([[float(len(t))] for t in texts])

    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [0] * max(1, len(text.split()))

    # Build minimal Documents list
    Doc = chunking_utilities.Document
    docs = [
        Doc(page_content="aaa bbb", metadata={"chunk_id": "c1"}),
        Doc(page_content="ccc ddd eee", metadata={"chunk_id": "c2"}),
    ]

    with patch.object(chunking_utilities, "_load_models", return_value=(FakeEmb(), FakeTok())):
        df = chunking_utilities.chunks_to_df(docs, normalize_embeddings=False)
    assert isinstance(df, pd.DataFrame)
    assert "embedding" in df.columns
    # Embeddings should be lists
    assert isinstance(df.loc[0, "embedding"], list)


def test_cluster_chunks_df_agglomerative_fallback():
    # Create DataFrame with simple embeddings
    df = pd.DataFrame({
        "embedding": [[0.0], [0.1], [1.0], [1.1]],
        "token_len": [10, 12, 50, 55],
        "text": ["a", "b", "c", "d"],
        "metadata": [{}, {}, {}, {}],
        "chunk_id": ["i0", "i1", "i2", "i3"],
    })

    # Provide a fake AgglomerativeClustering replacement in the module
    class FakeAgglomerative:
        def __init__(self, n_clusters=None, metric=None, linkage=None, distance_threshold=None):
            self.n_clusters = n_clusters

        def fit_predict(self, X):
            n = X.shape[0]
            # label first half 0, second half 1
            return np.array([0] * (n // 2) + [1] * (n - n // 2))

    with patch.object(chunking_utilities, "hdbscan", None):
        with patch.object(chunking_utilities, "AgglomerativeClustering", FakeAgglomerative):
            out = chunking_utilities.cluster_chunks_df(df, method="agglomerative", min_cluster_size=2, require_length_diversity=False)
    # Expect a non-empty DataFrame (clusters of size >= 2 should be returned)
    assert isinstance(out, pd.DataFrame)
    assert "cluster_id" in out.columns


def test_evaluator_raises_on_missing_paths():
    ev = evaluator_module.Evaluator()
    with pytest.raises(ValueError):
        ev.evaluate_generation(ground_truth_path=None, data_path=None)
    with pytest.raises(ValueError):
        ev.evaluate_reranker(ground_truth_path=None, retriever_data_path=None)
    with pytest.raises(ValueError):
        ev.evaluate_retrieval(data_path=None, retriever_data_path=None)


def test_gevalscore_init_and_context_exit():
    # Patch OpenAI class in the g_eval module so no real network call is made
    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            # Provide minimal chat.completions.create used in evaluate
            class Chat:
                class Completions:
                    def create(self_inner, *args, **kwargs):
                        # Provide a fake response object with expected attributes
                        class Choice:
                            class Message:
                                def __init__(self):
                                    self.content = "3"
                                def strip(self): return "3"
                            def __init__(self):
                                self.message = self.Message()
                                # provide a logprobs like structure used in code path if needed
                                self.logprobs = type("LP", (), {"content": [type("C", (), {"top_logprobs": []})()]})
                                self.choices = []
                        class Resp:
                            def __init__(self):
                                self.choices = [Choice()]
                        return Resp()
                def __init__(self):
                    self.completions = self.Completions()
            self.chat = Chat()

    # Patch imported OpenAI symbol in module
    with patch.object(geval_module, "OpenAI", FakeOpenAI):
        g = geval_module.GEvalScore(api_key="dummy")
        assert hasattr(g, "client")
        # Context manager should leave client cleared after exit
        with g as ctx:
            assert ctx is g
            assert g.client is not None
        # After exit, client should be None (cleanup in __exit__)
        assert g.client is None

