"""Skill library: on-disk store, indexer, and retrieval (Librarian).

The library lives in ``skills/<skill_id>/`` (one folder per skill). Each folder
contains:

- ``spec.yaml`` — :class:`daedalus.core.spec.SkillSpec` metadata
- ``skill.py`` — module that defines and registers the implementation class
- ``tests/`` — JSON fixtures the registry can replay against the MockBackend
"""

from daedalus.library.librarian import (
    BM25Index,
    Librarian,
    SkillCard,
    SkillIndex,
    make_librarian,
)
from daedalus.library.loader import LoaderError, load_core_skills, load_library, load_skill

__all__ = [
    "BM25Index",
    "Librarian",
    "LoaderError",
    "SkillCard",
    "SkillIndex",
    "load_core_skills",
    "load_library",
    "load_skill",
    "make_librarian",
]
