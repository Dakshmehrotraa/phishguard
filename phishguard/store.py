"""SQLite log of scans and user feedback. Feedback rows feed the retraining loop
(the project's answer to concept drift: new attacker patterns get labelled and
folded back into training)."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, input TEXT,
    probability REAL, verdict TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, input TEXT, label INTEGER
);
"""


def _conn(db_path=None):
    con = sqlite3.connect(db_path or DB_PATH)
    con.executescript(SCHEMA)
    return con


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_scan(kind: str, text: str, probability: float, verdict: str, db_path=None) -> int:
    with closing(_conn(db_path)) as con, con:
        cur = con.execute("INSERT INTO scans (ts, kind, input, probability, verdict) VALUES (?,?,?,?,?)",
                          (_now(), kind, text, probability, verdict))
        return cur.lastrowid


def add_feedback(kind: str, text: str, is_phishing: bool, db_path=None) -> int:
    if kind not in ("url", "text"):
        raise ValueError("kind must be 'url' or 'text'")
    with closing(_conn(db_path)) as con, con:
        cur = con.execute("INSERT INTO feedback (ts, kind, input, label) VALUES (?,?,?,?)",
                          (_now(), kind, text, int(is_phishing)))
        return cur.lastrowid


def recent_scans(limit: int = 50, db_path=None) -> pd.DataFrame:
    with closing(_conn(db_path)) as con:
        return pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC LIMIT ?", con, params=(limit,))


def load_feedback(kind: str, db_path=None) -> pd.DataFrame:
    """Return feedback as columns (url|text, label), latest label wins per input."""
    with closing(_conn(db_path)) as con:
        df = pd.read_sql_query("SELECT input, label FROM feedback WHERE kind=? ORDER BY id", con, params=(kind,))
    df = df.drop_duplicates("input", keep="last")
    return df.rename(columns={"input": "url" if kind == "url" else "text"})
