"""Librarian: BM25 retrieval over the on-disk skill library."""

from __future__ import annotations

from daedalus.library import BM25Index, Librarian


def test_bm25_returns_relevant_skill_first():
    idx = BM25Index()
    idx.add("click_mouse", "click the mouse at a screen pixel")
    idx.add("type_text", "type a literal text string at the current focus")
    idx.add("wait", "sleep for milliseconds")
    hits = idx.query("click on the screen", k=3)
    assert hits
    assert hits[0].skill_id == "click_mouse"


def test_bm25_empty_query_returns_nothing():
    idx = BM25Index()
    idx.add("foo", "bar")
    assert idx.query("", k=5) == []
    assert idx.query("###", k=5) == []


def test_librarian_search_against_real_library():
    lib = Librarian()
    lib.reindex()
    cards = lib.search("type some text into the editor", k=3)
    assert cards
    assert cards[0].id == "type_text"
    # Schemas come back populated.
    assert "properties" in cards[0].inputs_schema
    assert "properties" in cards[0].outputs_schema


def test_librarian_card_for_known_skill():
    lib = Librarian()
    card = lib.card_for("click_mouse")
    assert card is not None
    assert card.id == "click_mouse"
    assert "screen_input" in card.side_effects


def test_librarian_search_with_synonyms():
    lib = Librarian()
    lib.reindex()
    cards = lib.search("press the keyboard shortcut control c to copy", k=5)
    ids = [c.id for c in cards]
    assert "type_shortcut" in ids


def test_librarian_search_for_screenshot():
    lib = Librarian()
    lib.reindex()
    cards = lib.search("capture a view of the current screen as png", k=3)
    assert cards
    assert cards[0].id == "view_screen"


def test_all_cards_returns_every_registered_skill():
    lib = Librarian()
    cards = lib.all_cards()
    ids = {c.id for c in cards}
    assert {"click_mouse", "type_text", "type_shortcut", "view_screen", "wait"} <= ids
