"""Retraining loop (concept-drift handling).

Folds the user feedback stored by the app/API into the training data and retrains
both models, then asks a running API to reload them.

    python scripts/retrain.py
    python scripts/retrain.py --max-rows 80000
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishguard import store  # noqa: E402
from phishguard.train import train_text, train_url, write_summary  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    a = ap.parse_args()

    print(f"Feedback available: {len(store.load_feedback('url'))} URLs, {len(store.load_feedback('text'))} messages")
    train_url(max_rows=a.max_rows, include_feedback=True)
    train_text(include_feedback=True)
    write_summary()
    try:
        req = urllib.request.Request(f"{a.api}/reload", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            print("API reloaded:", json.loads(r.read()))
    except Exception:
        print("API not running (or unreachable) - it will load the new models on next start.")
