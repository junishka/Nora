"""Tests for ``_resolve_sdc_and_source_n``'s policy lookup.

The policy file keys datasets by basename — see ``policy.py:134-137``
("datasets keys are dataset filenames (not full paths)") — and
``get_schema`` honours that by passing ``path.name`` into the lookup.
``submit_script`` goes through ``_resolve_sdc_and_source_n``, which
previously passed the raw ``source_dataset`` string straight through.
That meant a researcher's ``non_disclosive_variables`` opt-in keyed
``data.csv`` was silently ignored if the model passed ``./data.csv``
or ``sub/data.csv`` as ``source_dataset`` — the policy lookup missed,
the SDC config degraded to the conservative default, and the
variable's min/max got suppressed against the researcher's stated
intent.

These tests pin the normalization so the regression can't recur.
"""

from __future__ import annotations

from pathlib import Path

from nora.config import set_cwd
from nora.policy import DatasetPolicy, NoraPolicy, save_policy
from nora.tools import _resolve_sdc_and_source_n


def _setup_policy(tmp_path: Path, basename: str, opt_in: tuple[str, ...]) -> None:
    """Write a policy.json with one dataset entry keyed by basename."""
    save_policy(tmp_path, NoraPolicy(
        datasets={basename: DatasetPolicy(
            non_disclosive_variables=opt_in,
            set_at="2026-04-21T00:00:00+00:00",
        )},
    ))


def test_policy_picked_up_for_bare_basename(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    (tmp_path / "data.csv").write_text("x,y\n1,2\n")
    _setup_policy(tmp_path, "data.csv", ("age",))

    sdc_cfg, _source_n, _audit_s = _resolve_sdc_and_source_n(tmp_path, "data.csv")
    assert sdc_cfg.non_disclosive_variables == frozenset({"age"})


def test_policy_picked_up_for_dot_slash_prefix(tmp_path: Path) -> None:
    """``./data.csv`` is a normal relative-path form a model might emit
    (R / Python conventions both accept it). Before the fix, the
    policy lookup missed because the raw string ``./data.csv`` was
    used as the lookup key instead of the basename."""
    set_cwd(tmp_path)
    (tmp_path / "data.csv").write_text("x,y\n1,2\n")
    _setup_policy(tmp_path, "data.csv", ("age",))

    sdc_cfg, _source_n, _audit_s = _resolve_sdc_and_source_n(tmp_path, "./data.csv")
    assert sdc_cfg.non_disclosive_variables == frozenset({"age"})


def test_policy_picked_up_for_subdirectory_path(tmp_path: Path) -> None:
    """The model often passes a relative path through a subdirectory
    (``data/cohort.csv``). Policy keys are still basenames per the
    policy.py contract, so the lookup must use ``cohort.csv``."""
    set_cwd(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "data.csv").write_text("x,y\n1,2\n")
    _setup_policy(tmp_path, "data.csv", ("age",))

    sdc_cfg, _source_n, _audit_s = _resolve_sdc_and_source_n(
        tmp_path, "sub/data.csv",
    )
    assert sdc_cfg.non_disclosive_variables == frozenset({"age"})


def test_no_policy_entry_returns_default_config(tmp_path: Path) -> None:
    """A dataset without an explicit policy entry must still resolve
    cleanly to the default SDC config (no non-disclosive opt-ins) —
    don't regress the no-policy-yet case while fixing the basename
    mismatch."""
    set_cwd(tmp_path)
    (tmp_path / "data.csv").write_text("x,y\n1,2\n")
    # No save_policy call: policy file absent.

    sdc_cfg, _source_n, _audit_s = _resolve_sdc_and_source_n(tmp_path, "data.csv")
    assert sdc_cfg.non_disclosive_variables == frozenset()


def test_no_source_dataset_returns_default_config(tmp_path: Path) -> None:
    """Defensive: missing source_dataset short-circuits before any
    path normalization runs."""
    set_cwd(tmp_path)
    sdc_cfg, source_n, _audit_s = _resolve_sdc_and_source_n(tmp_path, None)
    assert sdc_cfg.non_disclosive_variables == frozenset()
    assert source_n is None


def test_path_escape_falls_back_safely(tmp_path: Path) -> None:
    """A path-escape attempt (``../something``) shouldn't raise out
    of policy resolution; the lookup just misses and the default
    config returns. The escape is the script author's bug to fix
    elsewhere — this function's job is to not blow up sanitization."""
    set_cwd(tmp_path)
    sdc_cfg, _source_n, _audit_s = _resolve_sdc_and_source_n(
        tmp_path, "../escaped.csv",
    )
    assert sdc_cfg.non_disclosive_variables == frozenset()
