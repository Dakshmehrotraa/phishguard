"""Dataset loaders. Every loader returns labels as 1 = phishing/scam, 0 = legitimate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

LABEL_MAP = {
    "phishing": 1, "phish": 1, "malicious": 1, "bad": 1, "spam": 1, "smish": 1,
    "smishing": 1, "scam": 1, "fraud": 1, "1": 1,
    "legitimate": 0, "legit": 0, "benign": 0, "good": 0, "ham": 0, "safe": 0, "0": 0,
}


def _map_labels(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(LABEL_MAP)
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Unrecognised label values: {bad}. Expected e.g. phishing/legitimate, spam/ham or 1/0.")
    return mapped.astype(int)


def _find(pattern: str, base: Path = DATA_DIR) -> Path | None:
    hits = sorted(p for p in base.rglob(pattern) if not p.name.startswith("."))  # skip e.g. ".old.arff"
    return hits[0] if hits else None


# ---------------------------------------------------------------- SMS / text
def load_sms(path: str | Path | None = None) -> pd.DataFrame:
    """UCI SMS Spam Collection: tab-separated 'ham|spam<TAB>message', no header."""
    path = Path(path) if path else _find("SMSSpamCollection")
    if path is None or not path.exists():
        raise FileNotFoundError("UCI SMS Spam Collection not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3, encoding="utf-8")
    df["label"] = _map_labels(df["label"])
    return df.dropna(subset=["text"]).reset_index(drop=True)


def load_text_csv(path: str | Path) -> pd.DataFrame:
    """Any CSV with a text column (text/message/body/email) and a label column."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[c] for c in ("text", "message", "body", "email", "content", "sms") if c in cols), None)
    label_col = next((cols[c] for c in ("label", "class", "type", "category", "target") if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError(f"{path}: need a text column and a label column, found {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": _map_labels(df[label_col])})
    return out.dropna().reset_index(drop=True)


def load_text_data(paths: list[str] | None = None) -> pd.DataFrame:
    if not paths:
        return load_sms()
    frames = [load_sms(p) if Path(p).name == "SMSSpamCollection" or p.endswith(".tsv") else load_text_csv(p)
              for p in paths]
    return pd.concat(frames, ignore_index=True).drop_duplicates("text").reset_index(drop=True)


# ---------------------------------------------------------------- URLs
def load_url_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Raw-URL dataset. Default: PhiUSIIL (UCI #967). Any CSV with url + label columns also works.

    PhiUSIIL encodes label 1 = legitimate and 0 = phishing, so it is flipped here.
    """
    path = Path(path) if path else _find("PhiUSIIL*.csv")
    if path is None or not path.exists():
        raise FileNotFoundError("URL dataset not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("url")
    label_col = next((cols[c] for c in ("label", "status", "type", "class", "result") if c in cols), None)
    if url_col is None or label_col is None:
        raise ValueError(f"{path}: need 'url' and 'label' columns, found {list(df.columns)[:10]}...")

    if "phiusiil" in path.name.lower():
        y = 1 - df[label_col].astype(int)  # flip: 1 = phishing
        n_phish = int(y.sum())
        if n_phish > len(y) / 2:
            print(f"[warn] PhiUSIIL: {n_phish} of {len(y)} rows marked phishing; expected the minority "
                  "(100,945 of 235,795). Check the label mapping.")
    else:
        y = _map_labels(df[label_col])
    out = pd.DataFrame({"url": df[url_col].astype(str), "label": y.astype(int)})
    return out.drop_duplicates("url").reset_index(drop=True)


# ---------------------------------------------------------------- UCI Phishing Websites (benchmark)
def load_uci_websites(path: str | Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """UCI Phishing Websites (Mohammad & McCluskey): 11,055 rows x 30 features in ARFF format.

    'Result' = -1 is phishing, 1 is legitimate. These features are pre-computed
    (some need page content / WHOIS), so this dataset is used for the benchmark
    comparison only, not for live URL scanning.
    """
    path = Path(path) if path else _find("*.arff")
    if path is None or not path.exists():
        raise FileNotFoundError("UCI Phishing Websites .arff not found. Run: python scripts/download_data.py")
    names, rows, in_data = [], [], False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            names.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([int(v) for v in line.split(",")])
    df = pd.DataFrame(rows, columns=names)
    target = names[-1]
    y = (df[target] == -1).astype(int).to_numpy()
    return df.drop(columns=[target]), y
