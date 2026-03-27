"""Tests for backend.services.roots_registry."""

from __future__ import annotations

import json
from collections import OrderedDict

import pytest

from backend.services.roots_registry import (
    add_root,
    load_roots_state,
    remove_root,
    save_roots_state,
    stable_root_id,
)


@pytest.fixture
def tmp_roots(tmp_path):
    (tmp_path / "parent").mkdir()
    (tmp_path / "parent" / "child").mkdir()
    return tmp_path


def test_stable_root_id_uniquifies_when_id_taken_by_other_path(tmp_roots):
    p = str(tmp_roots / "parent")
    roots: dict[str, str] = {}
    first = stable_root_id(p, roots)
    roots[first] = "/some/other/path"
    second = stable_root_id(p, roots)
    assert second != first
    assert second.startswith(first[:6])


def test_add_root_returns_false_when_covered(tmp_roots):
    roots = OrderedDict()
    parent = str(tmp_roots / "parent")
    child = str(tmp_roots / "parent" / "child")
    added, rid_p = add_root(parent, roots)
    assert added is True
    nested, rid_c = add_root(child, roots)
    assert nested is False
    assert rid_c == rid_p


def test_add_root_drops_descendant_when_parent_added(tmp_roots):
    roots = OrderedDict()
    child = str(tmp_roots / "parent" / "child")
    parent = str(tmp_roots / "parent")
    add_root(child, roots)
    assert len(roots) == 1
    add_root(parent, roots)
    assert len(roots) == 1
    assert next(iter(roots.values())) == parent


def test_load_save_roundtrip(tmp_roots, tmp_path):
    roots = OrderedDict()
    parent = str(tmp_roots / "parent")
    add_root(parent, roots)
    path = tmp_path / "roots.json"
    save_roots_state(roots, str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["roots"] == [parent]

    roots2 = OrderedDict()
    load_roots_state(roots2, str(path), default_root=None)
    assert list(roots2.values()) == [parent]


def test_remove_root(tmp_roots):
    roots = OrderedDict()
    parent = str(tmp_roots / "parent")
    _, rid = add_root(parent, roots)
    assert remove_root(rid, roots) is True
    assert remove_root(rid, roots) is False
