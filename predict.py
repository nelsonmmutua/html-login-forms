"""Run inference on one or more html_signatures using a trained structural model."""
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from config import STRUCTURAL_COLS, STRUCTURAL_XGB_MODEL
from features import structural_features

# Minimal pattern: a valid signature starts with (body or (html and contains nested parens
_VALID_SIG_CHARS = set("abcdefghijklmnopqrstuvwxyz()")


def _validate_signatures(html_signatures: list[str]) -> None:
    if not html_signatures:
        raise ValueError("No signatures provided.")
    for i, sig in enumerate(html_signatures):
        if not sig or not sig.strip():
            raise ValueError(f"Signature at index {i} is empty.")
        if not sig.strip().startswith("("):
            raise ValueError(
                f"Signature at index {i} does not look like a valid html_signature "
                f"(expected a string starting with '('). Got: {sig[:60]!r}"
            )
        if sig.count("(") != sig.count(")"):
            raise ValueError(
                f"Signature at index {i} has mismatched parentheses "
                f"({sig.count('(')} opening, {sig.count(')')} closing)."
            )
        if not _VALID_SIG_CHARS.issuperset(sig.replace(" ", "").lower()):
            raise ValueError(
                f"Signature at index {i} contains unexpected characters. "
                "Expected only tag names and parentheses."
            )


def predict(html_signatures: list[str], model_path: Path) -> pd.DataFrame:
    _validate_signatures(html_signatures)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    df = pd.DataFrame({"html_signature": html_signatures})
    df = structural_features(df)
    X = df[STRUCTURAL_COLS].values
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df["prediction"] = ["LOGIN_FORM_MALICIOUS" if p == 1 else "NO_FORM" for p in preds]
    df["malicious_probability"] = probs
    return df[["html_signature", "prediction", "malicious_probability"]]


def predict_csv(input_path: Path, model_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    """Read html_signature column from a CSV, run predictions, optionally write results CSV."""
    df = pd.read_csv(input_path)
    if "html_signature" not in df.columns:
        raise ValueError(f"Input CSV must have an 'html_signature' column. Found: {list(df.columns)}")
    results = predict(df["html_signature"].tolist(), model_path)
    if "url" in df.columns:
        results.insert(0, "url", df["url"].values)
    if output_path:
        results.to_csv(output_path, index=False)
        log.info("Results saved → %s (%d rows)", output_path, len(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict login-form maliciousness from html_signature.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-csv", type=Path, metavar="CSV",
                       help="CSV file with an 'html_signature' column.")
    group.add_argument("signatures", nargs="*", default=[],
                       help="One or more html_signature strings (inline).")
    parser.add_argument("--model",      type=Path, default=STRUCTURAL_XGB_MODEL)
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save results to this CSV (only with --input-csv).")
    args = parser.parse_args()
    try:
        if args.input_csv:
            results = predict_csv(args.input_csv, args.model, args.output_csv)
            if not args.output_csv:
                log.info("\n%s", results.to_string(index=False))
        else:
            log.info("\n%s", predict(args.signatures, args.model).to_string(index=False))
    except (ValueError, FileNotFoundError) as e:
        log.error("Error: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
