"""Librarian: retrieval over the skill library.

The Planner asks "given this user goal, which existing skills are relevant?"
and the Librarian returns the top-k skill specs ranked by relevance. We use a
small in-memory BM25 index over the description + examples + tags of each
skill spec.

Why BM25 (not a vector store)?
    - The library is small (tens to a few hundred skills realistically).
    - BM25 needs no embedding model, no extra service, no extra deps.
    - The retrieval contract is fully behind :class:`SkillIndex`, so we can
      slot in a vector store later without touching the Planner.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from daedalus.core.registry import RegisteredSkill, Registry, get_registry
from daedalus.core.spec import SkillSpec

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

# Tiny English stopword list. We strip these to keep BM25 honest on natural-
# language queries like "take a picture of the screen", where short common
# words otherwise dominate via the IDF term.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "is", "are", "be", "as", "into", "from", "this",
        "that", "these", "those", "it", "its", "do", "does", "i", "we", "you",
        "your", "my", "any", "some", "if", "then", "than", "so", "such",
    }
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def _spec_document(spec: SkillSpec) -> str:
    """Flatten a spec into a single searchable document.

    Includes the skill id (also split on underscores so ``view_screen`` matches
    "screen" queries), description, tags, side effects, and the textual bodies
    of any examples. We deliberately omit code so nothing in ``skill.py`` can
    change retrieval results.
    """
    id_words = spec.id.replace("_", " ")
    parts: list[str] = [spec.id, id_words, spec.description]
    parts.extend(spec.tags)
    # Side-effect names like "screen_input" -> also "screen input"
    for s in spec.side_effects:
        parts.append(s)
        parts.append(s.replace("_", " "))
    parts.extend(spec.preconditions)
    parts.extend(spec.postconditions)
    for ex in spec.examples:
        if ex.note:
            parts.append(ex.note)
        parts.extend(str(k) for k in ex.inputs)
    return " \n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Index protocol + BM25 implementation
# ---------------------------------------------------------------------------


@dataclass
class IndexHit:
    skill_id: str
    score: float


class SkillIndex(Protocol):
    """The Librarian's retrieval contract. Backed by BM25 today; trivially
    swappable for a vector store later (chromadb/lancedb/pgvector)."""

    def add(self, skill_id: str, document: str) -> None: ...
    def query(self, text: str, k: int = 5) -> list[IndexHit]: ...
    def __len__(self) -> int: ...


class BM25Index:
    """Pure-Python BM25 (Robertson). Suitable for libraries up to ~10k skills.

    Parameters
    ----------
    k1, b:
        Standard BM25 hyperparameters. Defaults match Lucene.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: dict[str, list[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._n: int = 0

    def add(self, skill_id: str, document: str) -> None:
        tokens = _tokenize(document)
        if skill_id in self._docs:
            # Replace -- subtract old DF first.
            for term in set(self._docs[skill_id]):
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
            self._n -= 1
        self._docs[skill_id] = tokens
        self._doc_len[skill_id] = len(tokens)
        for term in set(tokens):
            self._df[term] += 1
        self._n += 1

    def query(self, text: str, k: int = 5) -> list[IndexHit]:
        if not self._docs:
            return []
        q_tokens = _tokenize(text)
        if not q_tokens:
            return []
        avgdl = sum(self._doc_len.values()) / max(1, self._n)
        scores: dict[str, float] = {}
        for sid, doc_tokens in self._docs.items():
            tf = Counter(doc_tokens)
            doc_len = self._doc_len[sid]
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self._df[term]
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                tf_t = tf[term]
                denom = tf_t + self._k1 * (1 - self._b + self._b * (doc_len / avgdl))
                score += idf * (tf_t * (self._k1 + 1)) / denom
            if score > 0:
                scores[sid] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [IndexHit(skill_id=sid, score=s) for sid, s in ranked[:k]]

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Librarian
# ---------------------------------------------------------------------------


@dataclass
class SkillCard:
    """The compact view of a skill the Planner sees per retrieval hit."""

    id: str
    version: str
    kind: str
    description: str
    side_effects: list[str]
    inputs_schema: dict
    outputs_schema: dict
    examples: list[dict]
    tags: list[str]
    score: float

    @classmethod
    def from_registry(cls, entry: RegisteredSkill, score: float) -> SkillCard:
        spec = entry.cls.SPEC
        return cls(
            id=entry.id,
            version=entry.version.raw,
            kind=spec.kind,
            description=spec.description.strip(),
            side_effects=list(spec.side_effects),
            inputs_schema=entry.cls.input_schema(),
            outputs_schema=entry.cls.output_schema(),
            examples=[ex.model_dump(exclude_none=True) for ex in spec.examples],
            tags=list(spec.tags),
            score=score,
        )


class Librarian:
    """RAG over the skill library. Built on top of a :class:`SkillIndex`."""

    def __init__(
        self,
        registry: Registry | None = None,
        index: SkillIndex | None = None,
    ) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._index: SkillIndex = index if index is not None else BM25Index()
        self._indexed_ids: set[str] = set()

    def reindex(self) -> None:
        """Rebuild the index from the current registry contents."""
        self._index = BM25Index() if isinstance(self._index, BM25Index) else self._index
        self._indexed_ids.clear()
        for entry in self._registry:
            self._index.add(entry.id, _spec_document(entry.cls.SPEC))
            self._indexed_ids.add(entry.id)

    def add(self, entry: RegisteredSkill) -> None:
        self._index.add(entry.id, _spec_document(entry.cls.SPEC))
        self._indexed_ids.add(entry.id)

    def search(self, query: str, *, k: int = 5) -> list[SkillCard]:
        if not self._indexed_ids:
            self.reindex()
        hits = self._index.query(query, k=k)
        cards: list[SkillCard] = []
        for hit in hits:
            try:
                entry = self._registry.get(hit.skill_id)
            except Exception:
                continue
            cards.append(SkillCard.from_registry(entry, score=hit.score))
        return cards

    def all_cards(self) -> list[SkillCard]:
        """Return every registered skill as a SkillCard (no scoring)."""
        out: list[SkillCard] = []
        for entry in self._registry:
            out.append(SkillCard.from_registry(entry, score=0.0))
        return out

    def card_for(self, skill_id: str) -> SkillCard | None:
        try:
            entry = self._registry.get(skill_id)
        except Exception:
            return None
        return SkillCard.from_registry(entry, score=0.0)

    def __len__(self) -> int:
        return len(self._registry)


def make_librarian(skills: Iterable[RegisteredSkill] | None = None) -> Librarian:
    """Convenience: build a Librarian indexed over the global registry (or a
    given set of registered skills)."""
    lib = Librarian()
    if skills is not None:
        for entry in skills:
            lib.add(entry)
    else:
        lib.reindex()
    return lib
