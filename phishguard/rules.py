"""Transparent heuristic rules for URL tricks the domain-level ML model cannot see
(paths, redirects, '@' tricks, shorteners). Each rule has a fixed risk weight; the rule
score is 1 - prod(1 - w) over the rules that fire. These are the "Heuristic Rules"
branch of the architecture: simple, explainable, and not learned from biased data."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from .allowlist import is_trusted
from .url_features import SHORTENERS, SUSPICIOUS_WORDS, _is_ip, _normalise, split_host

RULES = [
    # (id, weight, description)
    ("at_symbol", 0.60, "'@' in the URL: the browser ignores everything before it"),
    ("ip_host", 0.60, "Uses a raw IP address instead of a domain name"),
    ("punycode", 0.50, "Punycode domain (can imitate a brand with look-alike characters)"),
    ("redirect_in_path", 0.40, "Contains '//' or another URL inside the path (redirect trick)"),
    ("keywords_in_path", 0.40, "Several phishing keywords in the path (login, verify, account...)"),
    ("shortener", 0.40, "URL shortener hides the real destination"),
    ("brand_in_subdomain", 0.55, "Well-known brand name in a subdomain of an unrelated domain"),
    ("keywords_in_subdomain", 0.45, "Phishing keywords in the subdomain (login, secure, verify...)"),
    ("free_hosting", 0.35, "Site on a free hosting / app platform (common for throwaway phishing pages)"),
    ("brand_on_free_hosting", 0.60, "Well-known brand name on a free hosting / app platform page"),
    ("many_subdomains", 0.35, "Unusually many subdomain levels"),
    ("http_with_credentials_words", 0.30, "Plain http (no encryption) on a login/verify page"),
]  # 12 rules
_WEIGHT = {r[0]: r[1] for r in RULES}
_DESC = {r[0]: r[2] for r in RULES}

BRANDS = ("paypal", "apple", "google", "microsoft", "outlook", "office365", "amazon", "netflix",
          "facebook", "instagram", "whatsapp", "sbi", "hdfc", "icici", "axis", "paytm", "phonepe",
          "flipkart", "irctc", "dhl", "fedex", "indiapost", "chase", "wellsfargo", "bankofamerica")


FREE_HOSTS = {"firebaseapp.com", "web.app", "repl.co", "replit.app", "000webhostapp.com", "netlify.app",
              "vercel.app", "glitch.me", "weebly.com", "wixsite.com", "blogspot.com", "github.io",
              "pages.dev", "workers.dev", "herokuapp.com", "azurewebsites.net", "appspot.com",
              "ngrok.io", "ngrok-free.app", "square.site", "webflow.io", "framer.app", "godaddysites.com",
              "sites.google.com", "ipfs.io", "dweb.link", "r2.dev", "surge.sh", "onrender.com"}


def _keywords(s: str) -> set:
    """Distinct phishing keywords in s; 'bank' is not counted again when 'banking' matched."""
    found = {w for w in SUSPICIOUS_WORDS if w in s}
    return {w for w in found if not any(w != o and w in o for o in found)}


def _on_platform(host: str) -> bool:
    """True for user-content pages on a free hosting / app platform (e.g. x.firebaseapp.com,
    sites.google.com/view/x), where anyone can publish under the platform's domain."""
    return host == "sites.google.com" or any(host.endswith("." + h) for h in FREE_HOSTS)


def evaluate(url: str) -> dict:
    scheme, full = _normalise(str(url))
    try:
        parts = urlsplit(full)
        host = (parts.hostname or "").lower()
    except ValueError:
        parts, host = urlsplit("http://invalid"), ""
    path_q = (parts.path + "?" + parts.query).lower()
    subdomains, registered = split_host(host.removeprefix("www."))
    platform = _on_platform(host)
    # A trusted organisation's own site (sbi.co.in/web/personal-banking, netbanking.hdfcbank.com)
    # naturally uses words like "bank" or "login"; keyword rules only apply to everyone else.
    own_trusted_site = is_trusted(registered) and not platform
    kw_in_path = 0 if own_trusted_site else len(_keywords(path_q))
    kw_in_sub = 0 if own_trusted_site else len(_keywords(subdomains))

    fired = []
    if "@" in parts.netloc:
        fired.append("at_symbol")
    if _is_ip(host):
        fired.append("ip_host")
    if "xn--" in host:
        fired.append("punycode")
    if "//" in parts.path or re.search(r"[?&=/](https?%3a|https?:)", full[8:], re.I):
        fired.append("redirect_in_path")
    if kw_in_path >= 2:
        fired.append("keywords_in_path")
    if host.removeprefix("www.") in SHORTENERS:
        fired.append("shortener")
    if subdomains and any(b in subdomains for b in BRANDS) and not any(b in registered for b in BRANDS):
        fired.append("brand_in_subdomain")
    if kw_in_sub >= 1:
        fired.append("keywords_in_subdomain")
    if platform:
        fired.append("free_hosting")
        if any(b in subdomains or b in path_q for b in BRANDS):
            fired.append("brand_on_free_hosting")
    if subdomains.count(".") >= 2:
        fired.append("many_subdomains")
    if scheme == "http" and kw_in_path >= 1:
        fired.append("http_with_credentials_words")

    score = 1.0
    for r in fired:
        score *= 1 - _WEIGHT[r]
    return {"score": round(1 - score, 4),
            "fired": [{"rule": r, "label": _DESC[r], "weight": _WEIGHT[r]} for r in fired]}
