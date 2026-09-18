"""Tests for config.py."""
from config import (
    CLASSES,
    CREDENTIAL_GROUP,
    MALICIOUS_LABEL,
    RF_PARAMS,
    STRUCTURAL_COLS,
    XGB_PARAMS,
)


def test_structural_cols_count():
    assert len(STRUCTURAL_COLS) == 14


def test_credential_group_members():
    assert set(CREDENTIAL_GROUP) == {"form", "input", "label", "iframe", "style"}


def test_has_flags_in_structural_cols():
    has_flags = [c for c in STRUCTURAL_COLS if c.startswith("has_")]
    assert len(has_flags) == len(CREDENTIAL_GROUP) + 1


def test_classes():
    assert CLASSES == ["NO_FORM", "LOGIN_FORM_MALICIOUS"]


def test_malicious_label_in_classes():
    assert MALICIOUS_LABEL in CLASSES


def test_xgb_params_keys():
    required = {"n_estimators", "max_depth", "learning_rate", "random_state"}
    assert required.issubset(XGB_PARAMS)


def test_rf_params_keys():
    required = {"n_estimators", "class_weight", "random_state"}
    assert required.issubset(RF_PARAMS)
