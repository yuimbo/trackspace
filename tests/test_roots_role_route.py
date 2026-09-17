"""Route tests for root roles: My library vs External.

External roots are analysed and laid out like any other, but the folder tree tags
them so the frontend can section them and keep them out of the default view.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from backend.routes.api_folders import create_api_folders_blueprint
from backend.services.request_validation import ApiValidationError
from backend.services.roots_registry import ROLE_EXTERNAL, ROLE_LIBRARY


def _app(bp) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(
            Path(__file__).resolve().parents[1] / "backend" / "templates"
        ),
    )

    @app.errorhandler(ApiValidationError)
    def _handle(exc: ApiValidationError):
        return {"ok": False, "error": str(exc)}, 400

    app.register_blueprint(bp)
    app.testing = True
    return app


def _client(*, roots: dict[str, str], root_roles: dict[str, str]):
    roles = dict(root_roles)

    def _folder_tree(root_abs: str, root_id: str, rel_prefix: str = "") -> dict:
        return {
            "name": root_id,
            "path": root_id,
            "children": [],
            "root_id": root_id,
        }

    bp = create_api_folders_blueprint(
        roots=roots,
        root_roles=roles,
        folder_tree=_folder_tree,
        resolve_virtual_path=lambda rel: "/music",
        virtual_from_abs=lambda p: p,
        add_root=lambda p: (True, "newroot"),
        remove_root=lambda rid: rid in roots,
        set_root_role=lambda rid, role: roles.__setitem__(rid, role),
        schedule_root_watch=lambda rid: None,
        unschedule_root_watch=lambda rid: None,
        save_roots_state=lambda: None,
        track_cache=None,
        dir_color_fn=lambda p: "#000",
    )
    return _app(bp).test_client(), roles


def test_folders_payload_tags_each_root_with_its_role():
    client, _ = _client(
        roots={"lib": "/music/lib", "ext": "/music/ext"},
        root_roles={"ext": ROLE_EXTERNAL},
    )
    res = client.get("/api/folders")
    assert res.status_code == 200
    roles = {r["root_id"]: r["role"] for r in res.get_json()["roots"]}
    assert roles == {"lib": ROLE_LIBRARY, "ext": ROLE_EXTERNAL}


def test_roots_role_rejects_unknown_role():
    client, _ = _client(roots={"lib": "/music/lib"}, root_roles={})
    res = client.post("/api/roots/role", json={"root_id": "lib", "role": "bogus"})
    assert res.status_code == 400


def test_roots_role_rejects_unknown_root():
    client, _ = _client(roots={"lib": "/music/lib"}, root_roles={})
    res = client.post(
        "/api/roots/role", json={"root_id": "nope", "role": ROLE_EXTERNAL}
    )
    assert res.status_code == 200
    assert res.get_json()["ok"] is False


def test_roots_role_sets_external():
    client, roles = _client(roots={"lib": "/music/lib"}, root_roles={})
    res = client.post(
        "/api/roots/role", json={"root_id": "lib", "role": ROLE_EXTERNAL}
    )
    assert res.status_code == 200
    assert res.get_json() == {"ok": True, "root_id": "lib", "role": ROLE_EXTERNAL}
    assert roles["lib"] == ROLE_EXTERNAL


def test_partial_renders_external_section():
    client, _ = _client(
        roots={"lib": "/music/lib", "ext": "/music/ext"},
        root_roles={"ext": ROLE_EXTERNAL},
    )
    res = client.get("/partials/folder-tree")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'data-root-role="library"' in html
    assert 'data-root-role="external"' in html
    assert "folder-section-label" in html
    assert "folder-role-btn" in html
