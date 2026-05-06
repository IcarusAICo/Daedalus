"""Tests for RunStore and AgentMemory."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------


class TestRunStore:
    @pytest.fixture
    def store(self, tmp_path: Path):
        from daedalus.core.store import RunStore

        return RunStore(tmp_path / "test.db", "test-task-1")

    def test_create_table_and_append(self, store):
        store.create_table("spots", {"x": "int", "y": "int", "label": "str"})
        rid = store.append("spots", {"x": 100, "y": 200, "label": "btn"})
        assert rid > 0
        assert store.count("spots") == 1

    def test_idempotent_create(self, store):
        store.create_table("t", {"a": "int"})
        store.create_table("t", {"a": "int"})
        assert store.count("t") == 0

    def test_all_rows(self, store):
        store.create_table("pts", {"x": "int", "y": "int"})
        store.append("pts", {"x": 1, "y": 2})
        store.append("pts", {"x": 3, "y": 4})
        rows = store.all_rows("pts")
        assert len(rows) == 2
        assert rows[0]["x"] == 1
        assert rows[1]["y"] == 4

    def test_query_with_where(self, store):
        store.create_table("items", {"kind": "str", "val": "int"})
        store.append("items", {"kind": "a", "val": 10})
        store.append("items", {"kind": "b", "val": 20})
        store.append("items", {"kind": "a", "val": 30})
        results = store.query("items", where={"kind": "a"})
        assert len(results) == 2
        assert all(r["kind"] == "a" for r in results)

    def test_json_column(self, store):
        store.create_table("data", {"payload": "json"})
        store.append("data", {"payload": {"nested": [1, 2, 3]}})
        rows = store.all_rows("data")
        assert rows[0]["payload"] == {"nested": [1, 2, 3]}

    def test_bool_column(self, store):
        store.create_table("flags", {"active": "bool"})
        store.append("flags", {"active": True})
        store.append("flags", {"active": False})
        rows = store.all_rows("flags")
        assert rows[0]["active"] is True
        assert rows[1]["active"] is False

    def test_table_names(self, store):
        store.create_table("alpha", {"x": "int"})
        store.create_table("beta", {"y": "int"})
        names = store.table_names()
        assert "alpha" in names
        assert "beta" in names

    def test_snapshot(self, store):
        store.create_table("a", {"v": "int"})
        store.append("a", {"v": 42})
        snap = store.snapshot()
        assert "a" in snap
        assert len(snap["a"]) == 1
        assert snap["a"][0]["v"] == 42

    def test_missing_table_raises(self, store):
        with pytest.raises(KeyError, match="does not exist"):
            store.all_rows("nonexistent")

    def test_separate_task_ids(self, tmp_path):
        from daedalus.core.store import RunStore

        s1 = RunStore(tmp_path / "shared.db", "task-1")
        s2 = RunStore(tmp_path / "shared.db", "task-2")
        s1.create_table("pts", {"x": "int"})
        s2.create_table("pts", {"x": "int"})
        s1.append("pts", {"x": 1})
        s2.append("pts", {"x": 2})
        s2.append("pts", {"x": 3})
        assert s1.count("pts") == 1
        assert s2.count("pts") == 2

    def test_concurrent_appends(self, store):
        store.create_table("concurrent", {"n": "int"})
        errors: list[Exception] = []

        def writer(start: int):
            try:
                for i in range(50):
                    store.append("concurrent", {"n": start + i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert store.count("concurrent") == 200


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


class TestAgentMemory:
    @pytest.fixture
    def mem(self, tmp_path: Path):
        from daedalus.memory import AgentMemory

        return AgentMemory(tmp_path / "memory.db")

    def test_add_and_recall(self, mem):
        mem.add_fact("strategy", "Use grid scanning for puzzle layouts", "task-1")
        facts = mem.recall("grid scanning")
        assert len(facts) >= 1
        assert any("grid scanning" in f.content for f in facts)

    def test_category_filter(self, mem):
        mem.add_fact("app_behavior", "The submit button is at the bottom", "task-1")
        mem.add_fact("failure_mode", "Clicking too fast causes errors", "task-1")
        facts = mem.recall("button", category="app_behavior")
        assert all(f.category == "app_behavior" for f in facts)

    def test_unknown_category_defaults_to_general(self, mem):
        fid = mem.add_fact("unknown_category", "test content", "task-1")
        facts = mem.all_facts(category="general")
        assert any(f.id == fid for f in facts)

    def test_skill_outcomes(self, mem):
        mem.add_skill_outcome("click_element", "task-1", success=True)
        mem.add_skill_outcome("click_element", "task-1", success=True)
        mem.add_skill_outcome("click_element", "task-2", success=False, notes="timeout")
        stats = mem.get_skill_stats("click_element")
        assert stats["successes"] == 2
        assert stats["failures"] == 1

    def test_all_facts(self, mem):
        for i in range(5):
            mem.add_fact("general", f"fact {i}", "task-1")
        facts = mem.all_facts()
        assert len(facts) == 5

    def test_recall_empty_query(self, mem):
        mem.add_fact("general", "some fact", "task-1")
        facts = mem.recall("")
        assert len(facts) >= 1

    def test_tags(self, mem):
        mem.add_fact("strategy", "Try scrolling first", "task-1", tags=["scroll", "navigation"])
        facts = mem.all_facts()
        assert facts[0].tags == ["scroll", "navigation"]


# ---------------------------------------------------------------------------
# DSL: $store: references
# ---------------------------------------------------------------------------


class TestStoreRefs:
    def test_is_store_ref(self):
        from daedalus.executor.dsl import is_store_ref

        assert is_store_ref("$store:edges")
        assert is_store_ref("$store:edges.count")
        assert not is_store_ref("$ref:step1.x")
        assert not is_store_ref("just a string")

    def test_resolve_store_ref_all_rows(self, tmp_path):
        from daedalus.core.store import RunStore
        from daedalus.executor.dsl import resolve_store_ref

        store = RunStore(tmp_path / "test.db", "t1")
        store.create_table("pts", {"x": "int", "y": "int"})
        store.append("pts", {"x": 1, "y": 2})
        store.append("pts", {"x": 3, "y": 4})
        result = resolve_store_ref("$store:pts", store)
        assert len(result) == 2
        assert result[0]["x"] == 1

    def test_resolve_store_ref_count(self, tmp_path):
        from daedalus.core.store import RunStore
        from daedalus.executor.dsl import resolve_store_ref

        store = RunStore(tmp_path / "test.db", "t1")
        store.create_table("pts", {"x": "int"})
        store.append("pts", {"x": 1})
        store.append("pts", {"x": 2})
        result = resolve_store_ref("$store:pts.count", store)
        assert result == 2

    def test_resolve_store_ref_no_store(self):
        from daedalus.executor.dsl import resolve_store_ref

        with pytest.raises(Exception, match="no RunStore"):
            resolve_store_ref("$store:test", None)

    def test_has_dynamic_refs(self):
        from daedalus.executor.dsl import _has_dynamic_refs

        assert _has_dynamic_refs({"x": "$ref:step.x"})
        assert _has_dynamic_refs({"x": "$store:pts"})
        assert _has_dynamic_refs({"nested": {"y": "$store:pts.count"}})
        assert not _has_dynamic_refs({"x": 42, "y": "hello"})

    def test_resolve_inputs_with_store(self, tmp_path):
        from daedalus.core.store import RunStore
        from daedalus.executor.dsl import resolve_inputs

        store = RunStore(tmp_path / "test.db", "t1")
        store.create_table("data", {"val": "int"})
        store.append("data", {"val": 10})
        store.append("data", {"val": 20})

        resolved = resolve_inputs(
            {"rows": "$store:data", "n": "$store:data.count", "static": 42},
            saved_outputs={},
            store=store,
        )
        assert len(resolved["rows"]) == 2
        assert resolved["n"] == 2
        assert resolved["static"] == 42


# ---------------------------------------------------------------------------
# Planner: StrategyResult
# ---------------------------------------------------------------------------


class TestStrategyResult:
    def test_needs_new_skills(self):
        from daedalus.planner.planner import MissingSkillSpec, StrategyResult

        empty = StrategyResult()
        assert not empty.needs_new_skills

        with_skills = StrategyResult(
            composite_skills=[
                MissingSkillSpec(
                    proposed_id="test_skill",
                    description="A test skill",
                    rationale="needed for testing",
                    inputs_hint={},
                    outputs_hint={},
                )
            ]
        )
        assert with_skills.needs_new_skills
