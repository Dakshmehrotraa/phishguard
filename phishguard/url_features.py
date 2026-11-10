"""Lexical and host-based URL feature extraction.

All features are computed from the URL string alone (no network lookups),
so extraction is fast enough for real-time scanning and gives the same
result at training time and at prediction time.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "bank", "banking", "confirm", "password",
    "webscr", "ebayisapi", "wallet", "suspend", "unlock", "billing", "invoice",
    "free", "bonus", "lucky", "prize", "gift", "reward", "claim", "kyc", "otp",
)

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "s.id", "t.ly", "v.gd", "lnkd.in", "trib.al",
}

# TLDs that are frequently abused in phishing campaigns (cheap / free registration)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "work", "click",
    "link", "country", "kim", "loan", "men", "date", "racing", "review", "win",
    "party", "stream", "download", "accountant", "science", "cricket", "icu",
    "buzz", "rest", "fit", "monster", "cyou", "sbs", "cfd",
}

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_at",
    "num_question", "num_equals", "num_ampersand", "num_percent", "num_tilde",
    "num_digits", "digit_ratio", "letter_ratio", "special_char_ratio",
    "num_subdomains", "host_has_hyphen", "has_ip", "uses_https", "has_port",
    "tld_length", "suspicious_tld", "is_shortener", "num_suspicious_words",
    "double_slash_in_path", "has_punycode", "path_depth", "url_entropy",
    "longest_token_length", "host_digit_count",
]


def _normalise(url: str) -> tuple[str, str]:
    """Return (original_scheme, url_with_scheme) so urlsplit parses the host."""
    url = url.strip()
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", url)
    if m:
        return m.group(1).lower(), url
    return "", "http://" + url


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        # hex / decimal encoded IPs such as 0x7f000001 or 3232235777
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host, re.I))


def extract_url_features(url: str) -> dict[str, float]:
    scheme, full = _normalise(str(url))
    try:
        parts = urlsplit(full)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # malformed port etc.
        parts = urlsplit("http://invalid")
        host, port = "", None
    path, query = parts.path, parts.query
    raw = str(url).strip()
    lower = raw.lower()

    labels = [p for p in host.split(".") if p]
    tld = labels[-1] if labels else ""
    is_ip = _is_ip(host)
    # Subdomains = labels beyond "domain.tld"; ignore a leading "www"
    sub_labels = labels[:-2] if len(labels) > 2 else []
    if sub_labels and sub_labels[0] == "www":
        sub_labels = sub_labels[1:]

    n = max(len(raw), 1)
    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    tokens = [t for t in re.split(r"[/.?=&_\-:@~%+#]", lower) if t]

    feats = {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_at": raw.count("@"),
        "num_question": raw.count("?"),
        "num_equals": raw.count("="),
        "num_ampersand": raw.count("&"),
        "num_percent": raw.count("%"),
        "num_tilde": raw.count("~"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "special_char_ratio": (n - digits - letters) / n,
        "num_subdomains": 0 if is_ip else len(sub_labels),
        "host_has_hyphen": int("-" in host),
        "has_ip": int(is_ip),
        "uses_https": int(scheme == "https"),
        "has_port": int(port is not None),
        "tld_length": 0 if is_ip else len(tld),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(host.removeprefix("www.") in SHORTENERS),
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "path_depth": len([p for p in path.split("/") if p]),
        "url_entropy": _entropy(lower),
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "host_digit_count": sum(ch.isdigit() for ch in host),
    }
    return {k: float(feats[k]) for k in FEATURE_NAMES}


def featurize_urls(urls) -> "list[list[float]]":
    return [[f[k] for k in FEATURE_NAMES] for f in map(extract_url_features, urls)]


URL_REGEX = re.compile(
    r"(?:(?:https?://|www\.)[^\s<>\"']+|\b[a-z0-9.-]+\.(?:com|net|org|in|info|ly|co|io|xyz|top|tk|ml|ga|cf|gq|me|link|click)(?:/[^\s<>\"']*)?)",
    re.I,
)


def find_urls(text: str) -> list[str]:
    """Pull URLs out of free text (SMS / email body)."""
    return [u.rstrip(".,);:!?") for u in URL_REGEX.findall(text or "")]


# ---------------------------------------------------------------- domain-level model input
# Public URL datasets collect legitimate URLs differently from phishing URLs. In PhiUSIIL
# every legitimate URL is a bare "https://www.<domain>" homepage, so scheme, "www." and path
# features would teach the model "has a path / no https = phishing", and since legitimate
# URLs never have subdomains either, "any subdomain = phishing". The ML model therefore sees
# only the registered domain (e.g. google.com, sbi.co.in); subdomain, path and scheme tricks
# are handled by the transparent rules in phishguard/rules.py instead.
PATH_OR_SCHEME_FEATURES = {
    "url_length", "path_length", "query_length", "num_slashes", "num_at", "num_question",
    "num_equals", "num_ampersand", "num_percent", "num_tilde", "uses_https", "has_port",
    "double_slash_in_path", "path_depth", "num_subdomains", "is_shortener",
}
HOST_FEATURE_NAMES = [n for n in FEATURE_NAMES if n not in PATH_OR_SCHEME_FEATURES]


def host_of(url: str) -> str:
    _, full = _normalise(str(url))
    try:
        host = (urlsplit(full).hostname or "").lower()
    except ValueError:
        host = ""
    return host.removeprefix("www.")


# Common two-part public suffixes (a small stand-in for the full Public Suffix List)
TWO_PART_SUFFIXES = {
    "co.in", "ac.in", "gov.in", "org.in", "net.in", "nic.in", "edu.in", "res.in", "gen.in", "firm.in",
    "co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "or.jp", "com.br", "com.cn", "com.mx", "com.tr", "com.sg", "com.my",
    "co.nz", "co.za", "co.kr", "com.pk", "com.bd", "com.np", "com.lk", "com.ar", "com.co",
    "com.hk", "com.tw", "co.id", "or.id", "web.id", "my.id", "biz.id", "com.ng", "com.eg", "com.sa",
    "com.vn", "com.ph", "co.th", "in.th", "com.ua", "co.il", "com.ru",
}


def split_host(host: str) -> tuple[str, str]:
    """Return (subdomain, registered_domain), e.g. 'mail.google.com' -> ('mail', 'google.com')."""
    labels = [p for p in host.split(".") if p]
    if len(labels) < 2 or _is_ip(host):
        return "", host
    n = 3 if ".".join(labels[-2:]) in TWO_PART_SUFFIXES and len(labels) >= 3 else 2
    return ".".join(labels[:-n]), ".".join(labels[-n:])


def registered_domain(url: str) -> str:
    return split_host(host_of(url))[1]


def featurize_hosts(urls) -> "list[list[float]]":
    """Features of the registered domain only. In the training data, legitimate URLs never
    have subdomains, so subdomain signals cannot be learned fairly; rules.py covers them."""
    rows = []
    for u in urls:
        f = extract_url_features(registered_domain(u))
        rows.append([f[k] for k in HOST_FEATURE_NAMES])
    return rows


class DomainVectorizer:
    """Registered domain -> sparse features: character 2-5 grams (catch look-alikes such as
    '0utlook', 'paypa1', 'secure-update') plus the scaled numeric domain features."""

    def __init__(self, max_features: int = 50000):
        self.max_features = max_features

    def fit(self, domains):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MaxAbsScaler
        domains = list(domains)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     sublinear_tf=True, max_features=self.max_features).fit(domains)
        self.scaler_ = MaxAbsScaler().fit(self.numeric(domains))
        return self

    @staticmethod
    def numeric(domains):
        import numpy as np
        return np.asarray(featurize_hosts(domains), dtype=float)

    def transform(self, domains, numeric=None):
        import scipy.sparse as sp
        domains = list(domains)
        num = self.numeric(domains) if numeric is None else numeric
        return sp.hstack([self.char_.transform(domains), sp.csr_matrix(self.scaler_.transform(num))]).tocsr()
