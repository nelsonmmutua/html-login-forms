"""Tests for retrain.py."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SIG_MALICIOUS = "(body(div(form(label(input))(label(input))(button))(iframe)(style)))"
SIG_NO_FORM   = "(body(div(header(nav(ul(li(a))(li(a)))))(main(p)(p))))"


def _make_unique_sigs(base: str, n: int) -> list[str]:
    return [base.replace("body", f"body{i}", 1) for i in range(n)]


EXISTING = pd.DataFrame({
    "html_signature": _make_unique_sigs(SIG_MALICIOUS, 30) + _make_unique_sigs(SIG_NO_FORM, 30),
    "label": ["LOGIN_FORM_MALICIOUS"] * 30 + ["NO_FORM"] * 30,
})

NEW_DATA = pd.DataFrame({
    "html_signature": [
        "(body(div(form(input)(input)(button)(iframe)(style))))",
        "(body(nav(a)(a)(a)))",
    ],
    "label": ["LOGIN_FORM", "NO_FORM"],
})


def test_new_data_merges_and_deduplicates(tmp_path):
    existing_lf = tmp_path / "train_lf.csv"
    existing_nf = tmp_path / "train_nf.csv"
    new_path    = tmp_path / "new_data.csv"

    EXISTING[EXISTING["label"] == "LOGIN_FORM_MALICIOUS"].to_csv(existing_lf, sep=";", index=False)
    EXISTING[EXISTING["label"] == "NO_FORM"].to_csv(existing_nf)
    NEW_DATA.to_csv(new_path, index=False)

    import htmlloginforms.config as config
    with patch.object(config, "TRAIN_LOGIN_FORM", existing_lf), \
         patch.object(config, "TRAIN_NO_FORM",    existing_nf):
        from htmlloginforms.retrain import load_and_merge
        merged = load_and_merge(new_path)

    assert len(merged) >= len(EXISTING)
    assert merged.duplicated(subset=["html_signature"]).sum() == 0


def test_binary_label_assigned(tmp_path):
    existing_lf = tmp_path / "train_lf.csv"
    existing_nf = tmp_path / "train_nf.csv"
    new_path    = tmp_path / "new_data.csv"

    EXISTING[EXISTING["label"] == "LOGIN_FORM_MALICIOUS"].to_csv(existing_lf, sep=";", index=False)
    EXISTING[EXISTING["label"] == "NO_FORM"].to_csv(existing_nf)
    NEW_DATA.to_csv(new_path, index=False)

    import htmlloginforms.config as config
    with patch.object(config, "TRAIN_LOGIN_FORM", existing_lf), \
         patch.object(config, "TRAIN_NO_FORM",    existing_nf):
        from htmlloginforms.retrain import load_and_merge
        merged = load_and_merge(new_path)

    assert "binary_label" in merged.columns
    assert set(merged["binary_label"]).issubset({0, 1})
