"""Part D, tasks 1-3 -- chunking, TF-IDF retrieval and the grounded policy assistant.

The refund agent (agent_tools.search_policy) reuses `retrieve` from here, as the brief requires.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import common
import mock_llm

CHUNK_WORDS, OVERLAP_WORDS = 40, 10

POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied context. "
    "If the answer is not in the context, say that you could not find the information."
)


def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Windows of `size` words, each starting `size - overlap` words after the previous one.

    The first chunk covers words 0..size-1; stop after the first chunk whose end reaches or passes the last word.
    """
    assert 0 <= overlap < size
    step, chunks, start = size - overlap, [], 0
    while True:
        chunks.append((start, words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += step


def load_chunks(size=CHUNK_WORDS, overlap=OVERLAP_WORDS, policies_dir=None):
    """Read the five policy files and split each into chunks, remembering the source file name."""
    out = []
    for path in sorted((policies_dir or common.POLICIES).glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        for i, (start, piece) in enumerate(chunk_words(words, size, overlap)):
            out.append({"source": path.name, "chunk_id": i, "start_word": start, "end_word": start + len(piece) - 1,
                        "text": " ".join(piece)})
    return out


class PolicyIndex:
    """TF-IDF index over the chunk texts (vectoriser fitted on the chunks, queries transformed with the same vocabulary)."""

    def __init__(self, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
        self.size, self.overlap = size, overlap
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        """Top-k chunks by cosine similarity: a list of {source, score, text}. Ties keep the original chunk order."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4), "text": self.chunks[i]["text"]} for i in order]


_default_index = None


def default_index():
    global _default_index
    if _default_index is None:
        _default_index = PolicyIndex()
    return _default_index


def retrieve(query, k=3, index=None):
    return (index or default_index()).retrieve(query, k)


def build_user_message(question, hits):
    """CONTEXT: newline, each chunk prefixed with [source file] on its own line, newline QUESTION: newline, question."""
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    return f"CONTEXT:\n{context}\nQUESTION:\n{question}"


def answer(question, index=None):
    """Retrieve three chunks, ask the (mock) model with exactly two messages, return its text and the sources used."""
    hits = retrieve(question, k=3, index=index)
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, hits)}]
    response = mock_llm.chat(messages)
    return {"answer": response.choices[0].message.content, "sources": [h["source"] for h in hits],
            "tokens": response.usage.total_tokens}
