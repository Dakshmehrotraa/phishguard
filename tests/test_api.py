import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture()
def client(trained):
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["url_model"] and r["text_model"]


def test_scan_endpoints(client):
    r = client.post("/scan", json={"text": "Claim your prize at http://win-free.tk/claim"})
    assert r.status_code == 200 and r.json()["verdict"] in {"phishing", "suspicious", "safe"}
    assert client.post("/scan/url", json={"url": "https://github.com"}).status_code == 200
    assert client.post("/scan/text", json={"text": "see you at 5"}).status_code == 200
    assert client.post("/scan", json={}).status_code == 422


def test_feedback_and_history(client):
    assert client.post("/feedback", json={"kind": "text", "input": "x", "is_phishing": False}).json()["stored"]
    assert isinstance(client.get("/history").json(), list)
