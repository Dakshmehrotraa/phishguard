"""Train, compare and save the detection models.

    python -m phishguard.train all                # url + text + benchmark
    python -m phishguard.train url --max-rows 60000
    python -m phishguard.train text --data data/sms/SMSSpamCollection data/emails.csv
    python -m phishguard.train url --include-feedback   # retraining loop

Protocol (same for every model, so the comparison is fair):
  1. Stratified 80/20 train/test split. The test set is never touched until the end.
  2. The training part is split again 80/20 into fit/validation.
  3. SMOTE is applied to the fit split only; each candidate is trained and scored.
  4. The candidate with the best validation F1 is retrained on the full training
     split (SMOTE again, training data only) and evaluated once on the test set.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from . import __version__
from .balance import smote
from .config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE, TEXT_MODEL_PATH, URL_MODEL_PATH
from .data import load_text_data, load_uci_websites, load_url_csv
from .models import tabular_candidates, text_candidates, url_candidates
from .store import load_feedback
from .text_features import build_vectorizer
from .url_features import HOST_FEATURE_NAMES, DomainVectorizer, registered_domain


# ---------------------------------------------------------------- helpers
def metrics(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4) if len(set(y_true)) > 1 else None,
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _to_dense_if_needed(name, X):
    return X.toarray() if "HistGradientBoosting" in name and hasattr(X, "toarray") else X


def compare_and_select(raw_fit, y_fit, raw_val, y_val, raw_test, y_test, candidates, make_features):
    """make_features(train_raw) -> transform function fitted on train_raw only."""
    transform = make_features(raw_fit)
    Xf, Xv, Xt = transform(raw_fit), transform(raw_val), transform(raw_test)
    Xf_bal, yf_bal, smote_method = smote(Xf, y_fit, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidates.items():
        t0 = time.time()
        m = clone(model).fit(_to_dense_if_needed(name, Xf_bal), yf_bal)
        secs = time.time() - t0
        val = metrics(y_val, m.predict_proba(_to_dense_if_needed(name, Xv))[:, 1])
        tst = metrics(y_test, m.predict_proba(_to_dense_if_needed(name, Xt))[:, 1])
        rows.append({"model": name, "val_f1": val["f1"], **{k: v for k, v in tst.items() if k != "confusion_matrix"},
                     "train_seconds": round(secs, 1)})
        print(f"  {name:<45} val F1={val['f1']:.4f}  test F1={tst['f1']:.4f}  ({secs:.1f}s)")
    table = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    return table, table.loc[0, "model"], smote_method


def save_report(task: str, table: pd.DataFrame, final: dict, extra: dict):
    table.to_csv(REPORTS_DIR / f"{task}_comparison.csv", index=False)
    payload = {"task": task, "trained_at": datetime.now().isoformat(timespec="seconds"),
               "version": __version__, **extra, "final_model_test_metrics": final,
               "comparison": table.to_dict(orient="records")}
    (REPORTS_DIR / f"{task}_metrics.json").write_text(json.dumps(payload, indent=2))
    try:
        _plot(task, table, final)
    except Exception as e:  # plotting is optional
        print(f"  (chart skipped: {e})")


def _plot(task, table, final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    t = table.set_index("model")[metric_cols].astype(float)
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(t) + 2))
    t.plot.barh(ax=ax, width=0.8, color=["#1f4e79", "#2e86c1", "#48c9b0", "#f5b041", "#cb4335"])
    lo = max(0.0, np.nanmin(t.values) - 0.05)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("Score on held-out test set")
    ax.set_title(f"{task.upper()} model comparison")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{task}_comparison.png", dpi=160)
    plt.close(fig)

    cm = final["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:,}", ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{task.upper()} - final model")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{task}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def _splits(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)
    return X_tr, y_tr, X_fit, y_fit, X_val, y_val, X_te, y_te


def _subsample(df, max_rows):
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df["label"], random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- tasks
def train_url(data=None, max_rows=None, include_feedback=False):
    print("\n=== URL model ===")
    df = load_url_csv(data)
    # The model sees the registered domain only (see url_features.py): one row per domain, so
    # the same site never sits in both the training and the test split. If a domain appears
    # with both labels (e.g. a legitimate site whose free subdomains were abused), the
    # legitimate label is kept, because the domain itself is a real site.
    df["host"] = df["url"].map(registered_domain)
    df = (df[df["host"] != ""].sort_values("label", kind="stable")
          .drop_duplicates("host").reset_index(drop=True))
    df = _subsample(df, max_rows)
    print(f"  domains={len(df):,}  phishing={int(df.label.sum()):,}  legitimate={int((1 - df.label).sum()):,}")
    X = df["host"].to_numpy(dtype=object)
    y = df["label"].to_numpy()
    X_tr, y_tr, X_fit, y_fit, X_val, y_val, X_te, y_te = _splits(X, y)
    n_fb = 0
    if include_feedback:
        fb = load_feedback("url")
        if len(fb):
            n_fb = len(fb)
            Xfb = fb["url"].map(registered_domain).to_numpy(dtype=object)
            X_tr, y_tr = np.concatenate([X_tr, Xfb]), np.concatenate([y_tr, fb["label"].to_numpy()])
            X_fit, y_fit = np.concatenate([X_fit, Xfb]), np.concatenate([y_fit, fb["label"].to_numpy()])
        print(f"  feedback rows added to training: {n_fb}")

    def make_features(train_domains):
        return DomainVectorizer().fit(train_domains).transform

    cands = url_candidates()
    table, best, smote_method = compare_and_select(X_fit, y_fit, X_val, y_val, X_te, y_te, cands, make_features)
    vec = DomainVectorizer().fit(X_tr)
    X_bal, y_bal, _ = smote(vec.transform(X_tr), y_tr, random_state=RANDOM_STATE)
    model = clone(cands[best]).fit(X_bal, y_bal)
    final = metrics(y_te, model.predict_proba(vec.transform(X_te))[:, 1])
    print(f"  selected: {best}  ->  test F1={final['f1']}  accuracy={final['accuracy']}")

    bundle = {"kind": "url", "vectorizer": vec, "model": model, "model_name": best,
              "feature_names": HOST_FEATURE_NAMES,
              "reference": np.median(vec.numeric(X_tr[y_tr == 0]), axis=0),
              "metrics": final, "version": __version__}
    joblib.dump(bundle, URL_MODEL_PATH)
    save_report("url", table, final, {"dataset": str(data or "PhiUSIIL (UCI #967)"), "rows": len(df),
                                      "features": len(HOST_FEATURE_NAMES), "input": "registered domain only (char 2-5 grams + domain features)", "balancing": smote_method,
                                      "selected_model": best, "feedback_rows": n_fb})
    return bundle


def train_text(data=None, include_feedback=False):
    print("\n=== Text (SMS / email) model ===")
    df = load_text_data(data)
    print(f"  rows={len(df):,}  scam/spam={int(df.label.sum()):,}  legitimate={int((1 - df.label).sum()):,}")
    texts, y = df["text"].to_numpy(dtype=object), df["label"].to_numpy()
    T_tr, y_tr, T_fit, y_fit, T_val, y_val, T_te, y_te = _splits(texts, y)
    n_fb = 0
    if include_feedback:
        fb = load_feedback("text")
        if len(fb):
            n_fb = len(fb)
            T_tr, y_tr = np.concatenate([T_tr, fb["text"].to_numpy(dtype=object)]), np.concatenate([y_tr, fb["label"]])
            T_fit, y_fit = np.concatenate([T_fit, fb["text"].to_numpy(dtype=object)]), np.concatenate([y_fit, fb["label"]])
        print(f"  feedback rows added to training: {n_fb}")

    def make_features(train_raw):
        vec = build_vectorizer().fit(train_raw)
        return vec.transform

    cands = text_candidates()
    table, best, smote_method = compare_and_select(T_fit, y_fit, T_val, y_val, T_te, y_te, cands, make_features)
    vec = build_vectorizer().fit(T_tr)
    X_bal, y_bal, _ = smote(vec.transform(T_tr), y_tr, random_state=RANDOM_STATE)
    model = clone(cands[best]).fit(_to_dense_if_needed(best, X_bal), y_bal)
    final = metrics(y_te, model.predict_proba(_to_dense_if_needed(best, vec.transform(T_te)))[:, 1])
    print(f"  selected: {best}  ->  test F1={final['f1']}  accuracy={final['accuracy']}")

    bundle = {"kind": "text", "vectorizer": vec, "model": model, "model_name": best,
              "metrics": final, "version": __version__}
    joblib.dump(bundle, TEXT_MODEL_PATH)
    save_report("text", table, final, {"dataset": ", ".join(map(str, data)) if data else "UCI SMS Spam Collection",
                                       "rows": len(df), "balancing": smote_method, "selected_model": best,
                                       "feedback_rows": n_fb})
    return bundle


def train_benchmark(data=None):
    """Benchmark on UCI Phishing Websites (30 pre-computed features) for comparison with literature."""
    print("\n=== Benchmark: UCI Phishing Websites (30 features) ===")
    X_df, y = load_uci_websites(data)
    print(f"  rows={len(X_df):,}  phishing={int(y.sum()):,}  legitimate={int((1 - y).sum()):,}")
    X = X_df.to_numpy(dtype=float)
    X_tr, y_tr, X_fit, y_fit, X_val, y_val, X_te, y_te = _splits(X, y)
    cands = tabular_candidates()
    table, best, smote_method = compare_and_select(X_fit, y_fit, X_val, y_val, X_te, y_te, cands,
                                                   lambda _t: (lambda X_: X_))
    X_bal, y_bal, _ = smote(X_tr, y_tr, random_state=RANDOM_STATE)
    model = clone(cands[best]).fit(X_bal, y_bal)
    final = metrics(y_te, model.predict_proba(X_te)[:, 1])
    print(f"  selected: {best}  ->  test F1={final['f1']}  accuracy={final['accuracy']}")
    save_report("benchmark", table, final, {"dataset": "UCI Phishing Websites (#327)", "rows": len(X_df),
                                            "features": X.shape[1], "balancing": smote_method,
                                            "selected_model": best})


def write_summary():
    """Collect every *_metrics.json into reports/summary.md (ready to copy into the results slide)."""
    lines = ["# PhishGuard - results summary", ""]
    for task in ("url", "text", "benchmark"):
        p = REPORTS_DIR / f"{task}_metrics.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        f = r["final_model_test_metrics"]
        lines += [f"## {task.upper()}  -  {r['dataset']}", "",
                  f"Rows: {r['rows']:,} | Balancing: {r['balancing']} | Selected (best validation F1): **{r['selected_model']}**", "",
                  "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |", "|---|---|---|---|---|---|---|"]
        for row in r["comparison"]:
            lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                         f"{row['f1']:.4f} | {row['roc_auc']:.4f} | {row['false_positive_rate']:.4f} |")
        lines += ["", f"Final model on test set: accuracy {f['accuracy']}, precision {f['precision']}, "
                  f"recall {f['recall']}, F1 {f['f1']}, ROC-AUC {f['roc_auc']}, confusion matrix {f['confusion_matrix']}", ""]
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nSummary written to {REPORTS_DIR / 'summary.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train PhishGuard models")
    ap.add_argument("task", choices=["url", "text", "benchmark", "all"])
    ap.add_argument("--data", nargs="+", help="override dataset path(s)")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample the URL dataset for faster runs")
    ap.add_argument("--include-feedback", action="store_true", help="add user feedback from the database to training")
    a = ap.parse_args(argv)
    one = a.data[0] if a.data else None
    if a.task in ("url", "all"):
        train_url(one if a.task == "url" else None, a.max_rows, a.include_feedback)
    if a.task in ("text", "all"):
        train_text(a.data if a.task == "text" else None, a.include_feedback)
    if a.task in ("benchmark", "all"):
        try:
            train_benchmark(one if a.task == "benchmark" else None)
        except FileNotFoundError as e:
            if a.task == "benchmark":
                raise
            print(f"\n(benchmark skipped: {e})")
    write_summary()


if __name__ == "__main__":
    main()
