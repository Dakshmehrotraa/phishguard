"""Real-time scanning: loads the trained models, scores input, explains the result
and fuses URL + text evidence into one verdict."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from .config import PHISHING_THRESHOLD, SUSPICIOUS_THRESHOLD, TEXT_MODEL_PATH, URL_MODEL_PATH
from .explain import explain_domain, explain_text
from . import rules
from .allowlist import is_trusted
from .url_features import find_urls, registered_domain


def verdict(p: float) -> str:
    if p >= PHISHING_THRESHOLD:
        return "phishing"
    if p >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "safe"


@lru_cache(maxsize=1)
def _load(path):
    if not path.exists():
        return None
    return joblib.load(path)


def load_models(reload: bool = False):
    if reload:
        _load.cache_clear()
    return _load(URL_MODEL_PATH), _load(TEXT_MODEL_PATH)


def scan_url(url: str, explain: bool = True) -> dict:
    url_bundle, _ = load_models()
    if url_bundle is None:
        raise RuntimeError("URL model not trained yet. Run: python -m phishguard.train url")
    # 1) trusted-domain allowlist, 2) ML model on unknown registered domains,
    # 3) transparent rules on the full URL. The URL's risk is the higher of model and rules.
    domain = registered_domain(url)
    trusted = is_trusted(domain)
    if trusted:
        p_model = 0.0
    else:
        p_model = float(url_bundle["model"].predict_proba(url_bundle["vectorizer"].transform([domain]))[0, 1])
    rule_result = rules.evaluate(url)
    p = max(p_model, rule_result["score"])
    out = {"type": "url", "input": url, "probability": round(p, 4), "verdict": verdict(p),
           "model": "allowlist (trusted domain)" if trusted else url_bundle["model_name"], "domain": domain,
           "trusted_domain": trusted, "model_probability": round(p_model, 4),
           "rule_score": rule_result["score"], "rules_fired": rule_result["fired"]}
    if explain:
        out["reasons"] = [] if trusted else explain_domain(url_bundle, domain)
    return out


def scan_text(text: str, explain: bool = True) -> dict:
    url_bundle, text_bundle = load_models()
    if text_bundle is None:
        raise RuntimeError("Text model not trained yet. Run: python -m phishguard.train text")
    X = text_bundle["vectorizer"].transform([text])
    if "HistGradientBoosting" in text_bundle["model_name"]:
        X = X.toarray()
    p = float(text_bundle["model"].predict_proba(X)[0, 1])
    out = {"type": "text", "input": text, "probability": round(p, 4), "verdict": verdict(p),
           "model": text_bundle["model_name"]}
    if explain:
        out["reasons"] = explain_text(text_bundle, text)
    # Links inside the message are scanned with the URL model as well
    out["links"] = [scan_url(u, explain=explain) for u in find_urls(text)[:5]] if url_bundle else []
    return out


def scan(url: str | None = None, text: str | None = None, explain: bool = True) -> dict:
    """Decision fusion: the overall risk is the highest risk among the message and every link
    (a message is dangerous if any part of it is). Each component's score is kept so the
    user can see which part triggered the alert."""
    if not url and not text:
        raise ValueError("Provide a url, a text, or both.")
    parts = []
    if text:
        t = scan_text(text, explain)
        parts.append(t)
        parts.extend(t.pop("links", []))
    if url:
        parts.append(scan_url(url, explain))
    top = max(parts, key=lambda r: r["probability"])
    return {"probability": top["probability"], "verdict": verdict(top["probability"]),
            "triggered_by": f"{top['type']}: {top['input'][:80]}", "components": parts}
