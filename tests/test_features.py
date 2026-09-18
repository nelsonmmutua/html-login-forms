"""Tests for features.py."""
import pandas as pd

from features import count_all_tags, count_tag, has_tag, structural_features

SIG_MALICIOUS = "(body(div(form(label(input))(label(input))(button))(iframe)(style)))"
SIG_NO_FORM   = "(body(div(header(nav(ul(li(a))(li(a)))))(main(p)(p))))"


def test_has_tag_present():
    assert has_tag(SIG_MALICIOUS, "form") == 1


def test_has_tag_absent():
    assert has_tag(SIG_NO_FORM, "form") == 0


def test_count_tag():
    assert count_tag(SIG_MALICIOUS, "input") == 2


def test_count_all_tags():
    assert count_all_tags("(body(div(p)))") == 3


def test_structural_features_columns():
    from config import STRUCTURAL_COLS
    df  = pd.DataFrame({"html_signature": [SIG_MALICIOUS, SIG_NO_FORM]})
    out = structural_features(df)
    for col in STRUCTURAL_COLS:
        assert col in out.columns, f"Missing column: {col}"


def test_structural_features_malicious_has_form():
    df  = pd.DataFrame({"html_signature": [SIG_MALICIOUS]})
    out = structural_features(df)
    assert out["has_form"].iloc[0] == 1


def test_structural_features_no_form_missing_form():
    df  = pd.DataFrame({"html_signature": [SIG_NO_FORM]})
    out = structural_features(df)
    assert out["has_form"].iloc[0] == 0


def test_structural_features_row_count():
    df  = pd.DataFrame({"html_signature": [SIG_MALICIOUS, SIG_NO_FORM]})
    out = structural_features(df)
    assert len(out) == 2


def test_form_density_range():
    df  = pd.DataFrame({"html_signature": [SIG_MALICIOUS, SIG_NO_FORM]})
    out = structural_features(df)
    assert out["form_density"].between(0, 1).all()
