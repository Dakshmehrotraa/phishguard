from phishguard.url_features import FEATURE_NAMES, extract_url_features, find_urls


def test_all_features_present():
    f = extract_url_features("https://www.example.com/a/b?x=1")
    assert list(f) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in f.values())


def test_ip_and_https():
    f = extract_url_features("http://192.168.10.5/login/verify.php")
    assert f["has_ip"] == 1 and f["uses_https"] == 0 and f["num_subdomains"] == 0
    assert extract_url_features("https://google.com")["uses_https"] == 1


def test_suspicious_signals():
    f = extract_url_features("http://secure.login.paypal-verify.tk/~acc//redirect")
    assert f["suspicious_tld"] == 1
    assert f["num_suspicious_words"] >= 3
    assert f["host_has_hyphen"] == 1
    assert f["double_slash_in_path"] == 1
    assert f["num_subdomains"] == 2


def test_shortener_punycode_and_missing_scheme():
    assert extract_url_features("bit.ly/abc")["is_shortener"] == 1
    assert extract_url_features("http://xn--pypal-4ve.com")["has_punycode"] == 1
    assert extract_url_features("example.com/path")["host_length"] == len("example.com")


def test_malformed_url_does_not_crash():
    extract_url_features("http://bad:port:here/")
    extract_url_features("")


def test_find_urls_in_text():
    urls = find_urls("Verify at http://sbi-kyc.tk/login. Or visit bit.ly/x1 now, or www.abc.com!")
    assert "http://sbi-kyc.tk/login" in urls
    assert any(u.startswith("bit.ly") for u in urls)
    assert "www.abc.com" in urls
