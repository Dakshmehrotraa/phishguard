import json

import numpy as np
import scipy.sparse as sp

from phishguard.balance import _builtin_smote
from phishguard.text_features import clean_text


def test_builtin_smote_balances_dense_and_sparse():
    rng = np.random.default_rng(0)
    X = rng.random((40, 4))
    y = np.array([0] * 32 + [1] * 8)
    Xr, yr = _builtin_smote(X, y)
    assert (yr == 0).sum() == (yr == 1).sum() == 32
    Xs, ys = _builtin_smote(sp.csr_matrix(X), y)
    assert sp.issparse(Xs) and Xs.shape[0] == 64
    # synthetic points lie inside the minority range
    assert Xr[40:].min() >= X[y == 1].min() - 1e-9 and Xr[40:].max() <= X[y == 1].max() + 1e-9


def test_clean_text_placeholders():
    t = clean_text("Win Rs 5000! Call +91 98765 43210 or visit http://x.tk now")
    assert "urltoken" in t and "phonetoken" in t and "moneytoken" in t


def test_training_writes_models_and_reports(trained):
    tmp = trained["tmp"]
    assert (tmp / "url.joblib").exists() and (tmp / "text.joblib").exists()
    report = json.loads((tmp / "url_metrics.json").read_text())
    assert report["selected_model"] == trained["url"]["model_name"]
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(report["final_model_test_metrics"])
    assert (tmp / "url_comparison.csv").exists()


def test_scan_url_returns_verdict_and_reasons(trained):
    from phishguard.predictor import scan_url
    r = scan_url("http://paypal-login-verify.tk/secure/login.php?id=88231")
    assert r["verdict"] in {"phishing", "suspicious", "safe"}
    assert 0 <= r["probability"] <= 1
    assert r["reasons"] and {"feature", "impact"} <= set(r["reasons"][0])


def test_scan_text_scans_embedded_links_and_fuses(trained):
    from phishguard.predictor import scan
    r = scan(text="URGENT your account is blocked, verify KYC now at http://sbi-kyc-update.tk/login")
    types = [c["type"] for c in r["components"]]
    assert types[0] == "text" and "url" in types
    assert r["probability"] == max(c["probability"] for c in r["components"])


def test_obvious_cases_on_sample_models(trained):
    from phishguard.predictor import scan_text
    assert scan_text("Congratulations you won a lucky draw, claim your prize now bit.ly/win")["probability"] > 0.5
    assert scan_text("are we still meeting for lunch at 1")["probability"] < 0.5


def test_feedback_roundtrip(trained):
    from phishguard import store
    store.add_feedback("url", "http://new-scam.example/verify", True)
    fb = store.load_feedback("url")
    assert fb.iloc[-1]["label"] == 1
