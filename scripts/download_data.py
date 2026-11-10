"""Download the public datasets used by PhishGuard into ./data.

    python scripts/download_data.py            # all three
    python scripts/download_data.py sms url    # pick some

If a download fails (network / firewall), open the page link printed below in a
browser, download the zip manually and unzip it into the folder shown.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "sms": {
        "name": "UCI SMS Spam Collection (5,574 messages)",
        "zip": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "page": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "folder": "sms",
        "expect": "SMSSpamCollection",
    },
    "url": {
        "name": "PhiUSIIL Phishing URL Dataset (235,795 URLs)",
        "zip": "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        "page": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
        "folder": "phiusiil",
        "expect": "PhiUSIIL*.csv",
    },
    "websites": {
        "name": "UCI Phishing Websites (11,055 rows, 30 features) - benchmark",
        "zip": "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip",
        "page": "https://archive.ics.uci.edu/dataset/327/phishing+websites",
        "folder": "uci_websites",
        "expect": "*.arff",
    },
}


def _extract(blob: bytes, dest: Path):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):  # zip-slip guard
                continue
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(member)
            if member.lower().endswith(".zip"):  # some UCI archives are nested
                _extract(data, dest)
            else:
                target.write_bytes(data)


def fetch(key: str) -> bool:
    d = DATASETS[key]
    dest = DATA / d["folder"]
    if list(dest.rglob(d["expect"])):
        print(f"[skip] {d['name']} already in {dest}")
        return True
    print(f"[get ] {d['name']}")
    try:
        req = urllib.request.Request(d["zip"], headers={"User-Agent": "PhishGuard/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
        dest.mkdir(parents=True, exist_ok=True)
        _extract(blob, dest)
    except Exception as e:
        print(f"[fail] {e}\n       Download manually from {d['page']} and unzip into {dest}")
        return False
    found = list(dest.rglob(d["expect"]))
    print(f"[ ok ] {found[0] if found else dest}")
    return bool(found)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(DATASETS)
    ok = all([fetch(k) for k in keys])
    sys.exit(0 if ok else 1)
