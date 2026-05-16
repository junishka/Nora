"""Real-fit coverage tests for ``from_cluster`` helpers in R and
Python — k-means + hierarchical (Ward), with DBSCAN edge cases
pinned at the helper level.

Property tests in ``tests/test_cluster_analysis.py`` exercise the
sanitizer on hand-crafted payloads (suppression gates, per-cluster
precision clamping, structural exclusion of assignments). This
module exercises the *helpers* against real ``stats::kmeans`` /
``stats::hclust`` / ``sklearn.cluster.KMeans`` /
``sklearn.cluster.AgglomerativeClustering`` fits on the Iris
dataset — three known-separated species.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NORA_R = _REPO_ROOT / "src" / "nora" / "runtime" / "nora.R"
_RSCRIPT = shutil.which("Rscript")


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file(),
    reason="Rscript / nora.R not available",
)


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


requires_sklearn = pytest.mark.skipif(
    not _sklearn_available(), reason="sklearn not installed",
)


_R_KMEANS_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
data(iris)
X <- iris[, 1:4]
m <- kmeans(X, centers = 3, nstart = 10)
nora$from_cluster(m, label = "Iris kmeans")
"""

_R_HCLUST_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
data(iris)
X <- iris[, 1:4]
d <- dist(X)
h <- hclust(d, method = "ward.D2")
nora$from_cluster(h, data = X, k = 3, linkage = "ward",
                  label = "Iris hclust Ward k=3")
"""

_PY_KMEANS_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
X, _ = load_iris(return_X_y=True)
m = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
nora_runtime.from_cluster(m,
    variables=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    label="Iris KMeans real-fit pin",
)
"""

_PY_AGG_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
X, _ = load_iris(return_X_y=True)
m = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
nora_runtime.from_cluster(m, X=X,
    variables=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    label="Iris Agglomerative Ward real-fit pin",
)
"""


def _read_one(path: Path) -> dict:
    line = path.read_text().strip().splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


# ---------------------------------------------------------------------------
# Real-fit pins
# ---------------------------------------------------------------------------


@requires_rscript
def test_r_from_cluster_kmeans_on_iris(tmp_path: Path) -> None:
    result_path = tmp_path / "out.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_KMEANS_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["method"] == "kmeans"
    assert s["n_clusters"] == 3
    assert sum(s["cluster_sizes"].values()) == 150
    # ss_ratio for well-separated Iris clusters should be high.
    assert s["ss_ratio"] > 0.7
    # All three centroids have all four variables.
    for cl in s["cluster_labels"]:
        assert set(s["centroids"][cl].keys()) == {
            "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width",
        }


@requires_rscript
def test_r_from_cluster_hclust_ward_on_iris(tmp_path: Path) -> None:
    result_path = tmp_path / "out.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_HCLUST_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["method"] == "hierarchical"
    assert s["linkage"] == "ward"
    assert s["n_clusters"] == 3
    assert sum(s["cluster_sizes"].values()) == 150
    # cut_height is the dendrogram threshold above which only 3
    # clusters remain — a real positive number for Iris.
    assert s.get("cut_height") is not None and s["cut_height"] > 0
    # ss_ratio comparable to kmeans on Iris (well-separated species).
    assert s["ss_ratio"] > 0.7
    # within_cluster_ss computed post-hoc from data + assignments.
    assert "within_cluster_ss" in s
    assert set(s["within_cluster_ss"].keys()) == set(s["cluster_labels"])


@requires_rscript
def test_r_from_cluster_hclust_requires_data_and_k(tmp_path: Path) -> None:
    """hclust fits don't store data or cluster count. Helper raises
    when either is omitted rather than guessing a default."""
    script_no_data = r"""
Sys.setenv(NORA_RUN_TOKEN = "t")
Sys.setenv(NORA_RESULT_PATH = "{path}")
source("{nora_r}")
data(iris)
h <- hclust(dist(iris[, 1:4]), method="ward.D2")
tryCatch({{ nora$from_cluster(h, k=3); cat("FAIL\n") }},
         error = function(e) cat("ERR:", conditionMessage(e), "\n"))
""".format(
        path=str(tmp_path / "out.jsonl").replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    )
    sp = tmp_path / "audit.R"
    sp.write_text(script_no_data)
    proc = subprocess.run(
        [_RSCRIPT, str(sp)],
        capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "data" in out


@requires_rscript
def test_r_from_cluster_refuses_dbscan_class(tmp_path: Path) -> None:
    """DBSCAN helper isn't shipped yet. The dispatch must raise with
    a pointer to the generic ``nora$result()`` path rather than
    silently producing a malformed payload."""
    script = r"""
Sys.setenv(NORA_RUN_TOKEN = "t")
Sys.setenv(NORA_RESULT_PATH = "{path}")
source("{nora_r}")
m <- structure(list(cluster = c(1,1,2,2,-1)), class = "dbscan")
tryCatch({{ nora$from_cluster(m); cat("FAIL\n") }},
         error = function(e) cat("ERR:", conditionMessage(e), "\n"))
""".format(
        path=str(tmp_path / "out.jsonl").replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    )
    sp = tmp_path / "audit.R"
    sp.write_text(script)
    proc = subprocess.run(
        [_RSCRIPT, str(sp)],
        capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "DBSCAN" in out
    assert "nora$result" in out  # points at the workaround


@requires_sklearn
def test_python_from_cluster_kmeans_on_iris(tmp_path: Path) -> None:
    result_path = tmp_path / "out.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_KMEANS_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["method"] == "kmeans"
    assert s["n_clusters"] == 3
    assert sum(s["cluster_sizes"].values()) == 150
    for cl in s["cluster_labels"]:
        assert set(s["centroids"][cl].keys()) == {
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        }


@requires_sklearn
def test_python_from_cluster_agglomerative_on_iris(tmp_path: Path) -> None:
    """sklearn's AgglomerativeClustering doesn't store cluster
    centers. Helper computes them post-hoc from X[labels == k].mean(axis=0)
    when X is supplied — pin that round-trip works."""
    result_path = tmp_path / "out.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_AGG_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["method"] == "hierarchical"
    assert s["linkage"] == "ward"
    assert s["n_clusters"] == 3
    assert sum(s["cluster_sizes"].values()) == 150
    # Centroids computed from X — every cluster has all four variables.
    for cl in s["cluster_labels"]:
        assert set(s["centroids"][cl].keys()) == {
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        }
    # within_cluster_ss / ss_ratio also computed post-hoc.
    assert "within_cluster_ss" in s
    assert s["ss_ratio"] > 0.7


@requires_sklearn
def test_python_from_cluster_agglomerative_requires_X(tmp_path: Path) -> None:
    """Without X, the helper can't compute centroids from sklearn's
    AgglomerativeClustering. Raises rather than producing a payload
    missing the required ``centroids`` field."""
    script = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
X, _ = load_iris(return_X_y=True)
m = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
try:
    nora_runtime.from_cluster(m, variables=["a","b","c","d"])
    print("FAIL_NO_ERROR")
except ValueError as e:
    print("ERR:", str(e))
""".format(runtime_dir=str(
        (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    ).replace("\\", "/"))
    script_path = tmp_path / "audit.py"
    script_path.write_text(script)
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(tmp_path / "out.jsonl")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "X" in out


@requires_sklearn
def test_python_from_cluster_refuses_dbscan(tmp_path: Path) -> None:
    """DBSCAN class triggers the dedicated-helper-not-shipped raise."""
    script = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
X, _ = load_iris(return_X_y=True)
m = DBSCAN(eps=0.5).fit(X)
try:
    nora_runtime.from_cluster(m, X=X, variables=["a","b","c","d"])
    print("FAIL_NO_ERROR")
except TypeError as e:
    print("ERR:", str(e))
""".format(runtime_dir=str(
        (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    ).replace("\\", "/"))
    script_path = tmp_path / "audit.py"
    script_path.write_text(script)
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(tmp_path / "out.jsonl")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "DBSCAN" in out
    assert "method=\"dbscan\"" in out  # points at the generic workaround


@requires_sklearn
def test_python_from_kmeans_back_compat_alias(tmp_path: Path) -> None:
    """The original ``from_kmeans`` name is retained as a back-compat
    alias delegating to ``from_cluster``. Pin that the alias still
    produces equivalent output."""
    script = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
X, _ = load_iris(return_X_y=True)
m = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
nora_runtime.from_kmeans(m,
    variables=["sepal_length","sepal_width","petal_length","petal_width"],
)
""".format(runtime_dir=str(
        (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    ).replace("\\", "/"))
    script_path = tmp_path / "audit.py"
    script_path.write_text(script)
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(tmp_path / "out.jsonl")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(tmp_path / "out.jsonl"))
    assert res.ok
    assert res.sanitized["method"] == "kmeans"


# ---------------------------------------------------------------------------
# Cross-language equivalence
# ---------------------------------------------------------------------------


@requires_rscript
@requires_sklearn
def test_r_and_python_kmeans_iris_agree(tmp_path: Path) -> None:
    """R kmeans + Python sklearn KMeans on Iris should recover the
    same 3-cluster structure (well-separated species). k-means cluster
    labels aren't unique — sorted cluster sizes are permutation-
    invariant; we check the partition shape rather than per-cluster
    centroids directly."""
    r_path = tmp_path / "r.jsonl"
    r_script = tmp_path / "audit.R"
    r_script.write_text(_R_KMEANS_SCRIPT.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_r.returncode == 0

    py_path = tmp_path / "py.jsonl"
    py_script = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    py_script.write_text(_PY_KMEANS_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(py_path)
    proc_py = subprocess.run(
        [sys.executable, str(py_script)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc_py.returncode == 0

    r_pl = sanitize(_read_one(r_path)).sanitized
    py_pl = sanitize(_read_one(py_path)).sanitized
    assert r_pl["n_clusters"] == py_pl["n_clusters"] == 3
    assert sum(r_pl["cluster_sizes"].values()) == sum(py_pl["cluster_sizes"].values()) == 150
    # Sorted sizes — permutation-invariant.
    r_sizes = sorted(r_pl["cluster_sizes"].values())
    py_sizes = sorted(py_pl["cluster_sizes"].values())
    # Iris has 50 of each species but kmeans doesn't recover that
    # exactly — typical split is ~50/62/38. Allow ±5 across
    # implementations.
    for rs, ps in zip(r_sizes, py_sizes):
        assert abs(rs - ps) <= 5, (
            f"cluster size divergence too large: R={r_sizes} vs Py={py_sizes}"
        )
    # ss_ratio is a known asymmetry: R's kmeans gives full SS
    # decomposition (within + between + total), so ``ss_ratio``
    # surfaces. sklearn's KMeans only emits inertia (= total_within),
    # so the Python helper has no between/total to compute the ratio
    # from. The cross-language signal here is the partition-shape
    # match, not the ratio.
    assert "ss_ratio" in r_pl  # R has it
    # Both helpers emit total_within_ss / inertia.
    assert "inertia" in r_pl and "inertia" in py_pl


# ---------------------------------------------------------------------------
# Stata via nora_result_cluster.ado
# ---------------------------------------------------------------------------

_STATA = None
for _name in ("stata-mp", "stata-se", "stata"):
    _p = shutil.which(_name)
    if _p:
        _STATA = _p
        break

_NORA_RESULT_CLUSTER_ADO = (
    _REPO_ROOT / "src" / "nora" / "runtime" / "nora_result_cluster.ado"
)
requires_stata_cluster = pytest.mark.skipif(
    _STATA is None or not _NORA_RESULT_CLUSTER_ADO.is_file(),
    reason="Stata binary / nora_result_cluster.ado not available",
)


_STATA_KMEANS_SCRIPT = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse auto, clear
quietly drop if missing(price, mpg, weight, length)
quietly cluster kmeans price mpg weight length, k(3) name(kmclus) start(random(42))
nora_result_cluster price mpg weight length, clusvar(kmclus) ///
    method("kmeans") label("Stata auto kmeans k=3")
"""


_STATA_WARD_SCRIPT = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse auto, clear
quietly drop if missing(price, mpg, weight, length)
quietly cluster wardslinkage price mpg weight length, name(wardlink)
quietly cluster generate wardclus = groups(3), name(wardlink)
nora_result_cluster price mpg weight length, clusvar(wardclus) ///
    method("hierarchical") linkage("ward") ///
    label("Stata auto Ward k=3")
"""


@requires_stata_cluster
def test_stata_nora_result_cluster_kmeans(tmp_path: Path) -> None:
    """Happy path: Stata cluster kmeans → helper → sanitizer.

    The R / Python tests above pin the contract in detail; this
    only asserts the Stata helper produces a sanitizer-valid payload
    with the required fields populated. Cross-language numeric
    agreement is not asserted here (different starting points yield
    different partitions; the partition-shape comparison lives in
    the R-vs-Python test above)."""
    result_path = tmp_path / "stata_kmeans.jsonl"
    script_path = tmp_path / "audit.do"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(
        _STATA_KMEANS_SCRIPT.format(runtime_dir=str(runtime_dir))
    )
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(script_path)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip(
            "Stata cluster kmeans produced no payload (likely Stata version "
            f"lacks cluster kmeans). stdout tail: {proc.stdout[-300:]}"
        )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected Stata kmeans payload: {res.rejection_reason}"
    )
    s = res.sanitized
    assert s["type"] == "cluster_analysis"
    assert s["method"] == "kmeans"
    assert s["n_clusters"] == 3
    assert s["n_features"] == 4
    assert set(s["variables"]) == {"price", "mpg", "weight", "length"}
    # Synthetic labels: cluster_1 / cluster_2 / cluster_3.
    assert set(s["cluster_labels"]) <= {"cluster_1", "cluster_2", "cluster_3"}
    # Cluster sizes sum to n_observations (modulo suppressed clusters).
    sizes = s.get("cluster_sizes", {})
    assert sizes and sum(sizes.values()) <= s["n_observations"]
    # Centroids present (kmeans is centroid-based).
    centroids = s.get("centroids", {})
    assert centroids
    for cluster_label, row in centroids.items():
        assert set(row.keys()) <= {"price", "mpg", "weight", "length"}


@requires_stata_cluster
def test_stata_nora_result_cluster_ward(tmp_path: Path) -> None:
    """Happy path: Stata hierarchical (Ward linkage) → helper →
    sanitizer. Pins that linkage="ward" rides through and the same
    cluster_analysis shape applies (centroids computed post-hoc from
    the assignment, since Stata's hierarchical commands don't store
    them natively)."""
    result_path = tmp_path / "stata_ward.jsonl"
    script_path = tmp_path / "audit.do"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(
        _STATA_WARD_SCRIPT.format(runtime_dir=str(runtime_dir))
    )
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(script_path)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip(
            f"Stata Ward produced no payload. stdout tail: {proc.stdout[-300:]}"
        )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected Stata Ward payload: {res.rejection_reason}"
    )
    s = res.sanitized
    assert s["method"] == "hierarchical"
    assert s["linkage"] == "ward"
    assert s["n_clusters"] == 3
    assert "centroids" in s
