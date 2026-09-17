"""Tests for backend.services.roots_registry."""

from __future__ import annotations

import json
from collections import OrderedDict

import pytest

from backend.services.roots_registry import (
    ROLE_EXTERNAL,
    ROLE_LIBRARY,
    add_root,
    library_root_ids,
    load_roots_state,
    remove_root,
    root_role,
    save_roots_state,
    set_root_role,
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


def test_root_role_defaults_to_library(tmp_roots):
    roots = OrderedDict()
    _, rid = add_root(str(tmp_roots / "parent"), roots)
    assert root_role(rid, {}) == ROLE_LIBRARY


def test_external_role_roundtrips_by_path(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "ext").mkdir()
    roots = OrderedDict()
    lib = str(tmp_path / "lib")
    ext = str(tmp_path / "ext")
    roles: dict[str, str] = {}
    _, rid_lib = add_root(lib, roots)
    _, rid_ext = add_root(ext, roots)
    set_root_role(rid_ext, ROLE_EXTERNAL, roles)
    path = tmp_path / "roots.json"

    save_roots_state(roots, str(path), root_roles=roles)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["external_roots"] == [ext]

    roots2 = OrderedDict()
    roles2: dict[str, str] = {}
    load_roots_state(
        roots2, str(path), default_root=None, root_roles=roles2
    )
    by_path = {p: rid for rid, p in roots2.items()}
    assert roles2[by_path[ext]] == ROLE_EXTERNAL
    assert roles2[by_path[lib]] == ROLE_LIBRARY


def test_load_without_external_key_defaults_to_library(tmp_roots, tmp_path):
    roots = OrderedDict()
    parent = str(tmp_roots / "parent")
    add_root(parent, roots)
    path = tmp_path / "roots.json"
    save_roots_state(roots, str(path))  # no roles passed: legacy shape

    roots2 = OrderedDict()
    roles: dict[str, str] = {}
    load_roots_state(roots2, str(path), default_root=None, root_roles=roles)
    assert all(role == ROLE_LIBRARY for role in roles.values())


def test_library_root_ids_excludes_external(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "ext").mkdir()
    roots = OrderedDict()
    _, rid_lib = add_root(str(tmp_path / "lib"), roots)
    _, rid_ext = add_root(str(tmp_path / "ext"), roots)
    roles = {rid_ext: ROLE_EXTERNAL}
    assert library_root_ids(roots, roles) == [rid_lib]
