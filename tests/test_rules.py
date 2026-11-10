from phishguard.rules import evaluate
from phishguard.url_features import HOST_FEATURE_NAMES, featurize_hosts, host_of


def test_host_of_strips_scheme_www_and_path():
    assert host_of("https://www.Example.com/a/b?x=1") == "example.com"
    assert host_of("example.com") == "example.com"


def test_host_features_ignore_path_scheme_and_www():
    a = featurize_hosts(["https://www.github.com"])[0]
    b = featurize_hosts(["http://github.com/some/long/path?with=query"])[0]
    assert a == b and len(a) == len(HOST_FEATURE_NAMES)


def test_rules_quiet_on_normal_links():
    for u in ["https://github.com/explore", "https://en.wikipedia.org/wiki/Phishing",
              "https://www.amazon.in/dp/B0CX23"]:
        assert evaluate(u)["score"] == 0, u


def test_rules_fire_on_tricks():
    assert evaluate("http://192.168.1.4/login")["score"] >= 0.6
    assert evaluate("http://user@evil.com/")["score"] >= 0.6
    r = evaluate("http://paypal.com.secure-login.xyz/verify/account")
    assert {f["rule"] for f in r["fired"]} >= {"brand_in_subdomain", "keywords_in_path"}
    assert evaluate("https://bit.ly/abc")["fired"][0]["rule"] == "shortener"


def test_allowlist_trusts_known_domains_but_rules_still_run(trained):
    from phishguard.predictor import scan_url
    r = scan_url("https://mail.google.com/mail/u/0/")
    assert r["trusted_domain"] and r["verdict"] == "safe"
    r = scan_url("https://sites.google.com/view/sbi-kyc-verify-login")
    assert r["trusted_domain"] and r["rule_score"] > 0
