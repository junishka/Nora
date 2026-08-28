"""Bundle-safety and stability of the cache-busted index.

``_materialize_cache_busted_index`` must never write inside
``web_dir``: in a packaged macOS .app that directory sits under
``Contents/Resources/`` — sealed by codesign — and writability is no
discriminator (a drag-installed, user-owned .app is writable while
still being sealed). The bust file always goes to a temp directory,
with a ``<base href>`` pointing relative refs back at ``web_dir``.

The temp-dir routing also fixes a build-id feedback loop: when the
bust file lived next to the assets, its own mtime fed the next
launch's hash, rolling the build id — and busting every cache — on
every start even with no asset change.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nora.ui import _materialize_cache_busted_index


def _write_web_dir(root: Path) -> Path:
    web = root / "web"
    web.mkdir()
    (web / "index.html").write_text(
        "<html>\n<head>\n<title>nora</title>\n"
        '<link rel="stylesheet" href="style.css" />\n'
        "</head>\n<body>\n"
        '<script src="app.js"></script>\n'
        "</body>\n</html>",
        encoding="utf-8",
    )
    (web / "app.js").write_text("// app", encoding="utf-8")
    (web / "style.css").write_text("body {}", encoding="utf-8")
    return web


def test_bust_file_never_lands_in_web_dir_even_when_writable(
    tmp_path: Path,
) -> None:
    """The writable-directory case IS the signed-bundle case (a
    user-owned .app is writable), so even a writable web_dir must
    stay untouched — the bust file routes to the temp dir and the
    HTML gains a <base href> back to web_dir."""
    web = _write_web_dir(tmp_path)
    assert os.access(web, os.W_OK)  # the pre-fix discriminator
    before = sorted(p.name for p in web.iterdir())

    out = _materialize_cache_busted_index(web, web / "index.html")

    assert out != web / "index.html", "busting silently disabled"
    assert out.parent == Path(tempfile.gettempdir()) / "nora-cache-bust"
    assert out.name.startswith(".index.bust-")
    # web_dir (the sealed bundle surface) is byte-for-byte untouched.
    assert sorted(p.name for p in web.iterdir()) == before

    html = out.read_text(encoding="utf-8")
    assert f'<base href="{web.resolve().as_uri()}/"' in html
    assert 'app.js?v=' in html
    assert 'style.css?v=' in html


def test_build_id_stable_across_launches(tmp_path: Path) -> None:
    """Two launches with unchanged assets must produce the same build
    id. Pre-fix, the first launch's bust file fed the second
    launch's mtime hash, so the id rolled every start."""
    web = _write_web_dir(tmp_path)
    first = _materialize_cache_busted_index(web, web / "index.html")
    second = _materialize_cache_busted_index(web, web / "index.html")
    assert first.name == second.name


def test_asset_change_rolls_build_id(tmp_path: Path) -> None:
    web = _write_web_dir(tmp_path)
    first = _materialize_cache_busted_index(web, web / "index.html")
    # Bump app.js's mtime well past filesystem timestamp granularity.
    st = (web / "app.js").stat()
    os.utime(web / "app.js", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    second = _materialize_cache_busted_index(web, web / "index.html")
    assert first.name != second.name


def test_leftover_bundle_bust_file_is_ignored_and_removed(
    tmp_path: Path,
) -> None:
    """A bust file that an old build wrote INTO web_dir (the bundle)
    must neither feed the hash nor survive the launch: excluding it
    keeps the build id equal to a clean launch's, and deleting it
    restores the bundle seal that its presence broke."""
    web = _write_web_dir(tmp_path)
    clean = _materialize_cache_busted_index(web, web / "index.html")

    leftover = web / ".index.bust-deadbeef0000.html"
    leftover.write_text("<html></html>", encoding="utf-8")

    after = _materialize_cache_busted_index(web, web / "index.html")
    assert after.name == clean.name, (
        "a leftover bust file inside web_dir changed the build id"
    )
    assert not leftover.exists(), (
        "leftover bust file inside the bundle was not cleaned up"
    )
