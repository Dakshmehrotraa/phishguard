"""Explanations for individual predictions.

URL model: SHAP TreeExplainer when shap is installed and the model is a single
tree model; otherwise occlusion (replace one feature with the typical value of a
legitimate URL and measure how much the phishing probability drops).

Text model: token occlusion (remove one word and re-score), which works for any
classifier and gives human-readable reasons such as "'verify' +1.8".

Occlusion impacts are in log-odds (positive = pushes toward phishing). SHAP impacts
are in the model's own output units (probability for Random Forest, log-odds for XGBoost).
"""
from __future__ import annotations

import re

import numpy as np

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

FEATURE_LABELS = {
    "url_length": "URL length", "host_length": "Domain length", "path_length": "Path length",
    "query_length": "Query-string length", "num_dots": "Number of dots", "num_hyphens": "Number of hyphens",
    "num_underscores": "Number of underscores", "num_slashes": "Number of slashes", "num_at": "'@' symbols",
    "num_question": "'?' symbols", "num_equals": "'=' symbols", "num_ampersand": "'&' symbols",
    "num_percent": "Percent-encoded characters", "num_tilde": "'~' symbols", "num_digits": "Digits in URL",
    "digit_ratio": "Share of digits", "letter_ratio": "Share of letters", "special_char_ratio": "Share of special characters",
    "num_subdomains": "Number of subdomains", "host_has_hyphen": "Hyphen in domain", "has_ip": "IP address instead of domain",
    "uses_https": "Uses HTTPS", "has_port": "Explicit port number", "tld_length": "TLD length",
    "suspicious_tld": "Frequently abused TLD", "is_shortener": "URL shortener", "num_suspicious_words": "Phishing keywords (login, verify...)",
    "double_slash_in_path": "'//' redirect in path", "has_punycode": "Punycode (look-alike characters)",
    "path_depth": "Path depth", "url_entropy": "Character randomness (entropy)", "longest_token_length": "Longest token length",
    "host_digit_count": "Digits in domain",
}


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _logit(model, X) -> np.ndarray:
    """Log-odds of phishing. Occlusion is measured here rather than on raw probability,
    because confident predictions (p close to 1) barely move in probability space."""
    if hasattr(model, "decision_function") and type(model).__name__ == "LogisticRegression":
        return model.decision_function(X)
    try:  # exact log-probabilities avoid rounding p to 1.0 (e.g. Naive Bayes)
        lp = np.maximum(model.predict_log_proba(X), np.log(1e-12))
        return lp[:, 1] - lp[:, 0]
    except (AttributeError, NotImplementedError):
        p = np.clip(_proba(model, X), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


def _tree_shap(model, x_row):
    import shap
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(x_row)
    if isinstance(vals, list):  # older shap: one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:  # newer shap: (rows, features, classes)
        vals = vals[..., 1]
    return vals[0]


def explain_tabular(bundle: dict, x_row: np.ndarray, top_k: int = 6) -> list[dict]:
    model, names = bundle["model"], bundle["feature_names"]
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    contrib, method = None, "occlusion"
    if HAS_SHAP and type(model).__name__ in {"RandomForestClassifier", "XGBClassifier"}:
        try:
            contrib, method = _tree_shap(model, x_row), "SHAP"
        except Exception:
            contrib = None
    if contrib is None:
        reference = np.asarray(bundle["reference"], dtype=float)
        variants = np.repeat(x_row, len(names), axis=0)
        variants[np.arange(len(names)), np.arange(len(names))] = reference
        contrib = _logit(model, x_row)[0] - _logit(model, variants)
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [
        {"feature": names[i], "label": FEATURE_LABELS.get(names[i], names[i]),
         "value": round(float(x_row[0, i]), 3), "impact": round(float(contrib[i]), 4), "method": method}
        for i in order if abs(contrib[i]) > 1e-3
    ]


def explain_text(bundle: dict, text: str, top_k: int = 6) -> list[dict]:
    vec, model = bundle["vectorizer"], bundle["model"]
    words = re.findall(r"[\w'£$€₹]+", text)
    uniq = list(dict.fromkeys(w.lower() for w in words))[:60]
    if not uniq:
        return []
    variants = [re.sub(rf"(?<![\w'£$€₹]){re.escape(w)}(?![\w'£$€₹])", " ", text, flags=re.I) for w in uniq]
    X = vec.transform([text] + variants)
    if "HistGradientBoosting" in bundle.get("model_name", ""):
        X = X.toarray()
    scores = _logit(model, X)
    contrib = scores[0] - scores[1:]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [{"token": uniq[i], "impact": round(float(contrib[i]), 4), "method": "token occlusion"}
            for i in order if abs(contrib[i]) > 1e-3]


def explain_domain(bundle: dict, domain: str, top_k: int = 6) -> list[dict]:
    """URL model: (1) occlude each numeric domain feature (set it to the typical value of a
    legitimate domain), (2) remove each part of the domain name ('emerald-exch-0utlook' ->
    'emerald', 'exch', '0utlook') and re-score. Impacts are log-odds."""
    vec, model, names = bundle["vectorizer"], bundle["model"], bundle["feature_names"]
    num = vec.numeric([domain])
    base = _logit(model, vec.transform([domain]))[0]
    variants = np.repeat(num, len(names), axis=0)
    variants[np.arange(len(names)), np.arange(len(names))] = np.asarray(bundle["reference"], dtype=float)
    num_contrib = base - _logit(model, vec.transform([domain] * len(names), numeric=variants))
    out = [{"feature": names[i], "label": FEATURE_LABELS.get(names[i], names[i]),
            "value": round(float(num[0, i]), 3), "impact": round(float(num_contrib[i]), 4), "method": "occlusion"}
           for i in range(len(names))]
    name_part = domain.rsplit(".", 1)[0]
    parts = list(dict.fromkeys(p for p in re.split(r"[.\-_]", name_part) if len(p) >= 2))
    if len(parts) > 1:
        reduced = [re.sub(rf"(^|[.\-_]){re.escape(p)}(?=$|[.\-_])", r"\1", domain).strip(".-_") or domain for p in parts]
        part_contrib = base - _logit(model, vec.transform(reduced))
        out += [{"feature": f"part:{p}", "label": f"Domain part '{p}'", "value": "",
                 "impact": round(float(c), 4), "method": "occlusion"} for p, c in zip(parts, part_contrib)]
    out.sort(key=lambda r: -abs(r["impact"]))
    return [r for r in out[:top_k] if abs(r["impact"]) > 1e-3]
