"""Structural feature helpers for login-form detection."""
import re

import pandas as pd

from htmlloginforms.config import CREDENTIAL_GROUP


def has_tag(sig: str, tag: str) -> int:
    return int(bool(re.search(rf"\({tag}\b", sig)))


def count_tag(sig: str, tag: str) -> int:
    return len(re.findall(rf"\({tag}\b", sig))


def count_all_tags(sig: str) -> int:
    return len(re.findall(r"\(\w+", sig))


def structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add STRUCTURAL_COLS to a copy of *frame* and return it."""
    out = frame.copy()
    for tag in CREDENTIAL_GROUP:
        out[f"has_{tag}"] = out["html_signature"].apply(lambda s, t=tag: has_tag(s, t))
    for tag in ["input", "label", "form", "button"]:
        out[f"{tag}_count"] = out["html_signature"].apply(lambda s, t=tag: count_tag(s, t))
    out["signature_length"] = out["html_signature"].str.len()
    out["total_tag_count"]  = out["html_signature"].apply(count_all_tags)
    out["form_density"] = (
        (out["form_count"] + out["input_count"] + out["label_count"])
        / out["total_tag_count"].replace(0, 1)
    )
    out["input_per_form"] = out["input_count"] / out["form_count"].replace(0, 1)
    cred_cols = [f"has_{t}" for t in CREDENTIAL_GROUP]
    out["has_credential_pattern"] = out[cred_cols].all(axis=1).astype(int)
    return out
