"""PhishGuard REST API.

    uvicorn api.main:app --reload
    Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phishguard import __version__, predictor, store

app = FastAPI(
    title="PhishGuard API",
    version=__version__,
    description="AI-based scam and phishing detection for URLs, SMS and email text.",
)
# Allow calls from the Streamlit UI, a browser extension or any local front end
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ScanRequest(BaseModel):
    url: Optional[str] = Field(None, examples=["http://sbi-kyc-update.tk/login.php"])
    text: Optional[str] = Field(None, examples=["URGENT: your account is blocked. Verify KYC at http://sbi-kyc-update.tk"])
    explain: bool = True


class UrlRequest(BaseModel):
    url: str
    explain: bool = True


class TextRequest(BaseModel):
    text: str
    explain: bool = True


class FeedbackRequest(BaseModel):
    kind: Literal["url", "text"]
    input: str
    is_phishing: bool


def _run(fn, *args, log_kind=None, log_input=None, **kwargs):
    try:
        result = fn(*args, **kwargs)
    except RuntimeError as e:  # model not trained
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if log_kind:
        store.log_scan(log_kind, log_input, result["probability"], result["verdict"])
    return result


@app.get("/health")
def health():
    url_b, text_b = predictor.load_models()
    info = lambda b: None if b is None else {"model": b["model_name"], "test_metrics": {  # noqa: E731
        k: v for k, v in b["metrics"].items() if k != "confusion_matrix"}}
    return {"status": "ok", "version": __version__, "url_model": info(url_b), "text_model": info(text_b)}


@app.post("/scan")
def scan(req: ScanRequest):
    """Scan a URL, a message, or both. Links inside the message are scanned too."""
    label = " | ".join(x for x in (req.url, req.text) if x)
    return _run(predictor.scan, req.url, req.text, req.explain, log_kind="combined", log_input=label)


@app.post("/scan/url")
def scan_url(req: UrlRequest):
    return _run(predictor.scan_url, req.url, req.explain, log_kind="url", log_input=req.url)


@app.post("/scan/text")
def scan_text(req: TextRequest):
    return _run(predictor.scan_text, req.text, req.explain, log_kind="text", log_input=req.text)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Report a wrong verdict. Feedback is used by `python scripts/retrain.py`."""
    fid = store.add_feedback(req.kind, req.input, req.is_phishing)
    return {"stored": True, "id": fid}


@app.get("/history")
def history(limit: int = 50):
    return store.recent_scans(limit).to_dict(orient="records")


@app.post("/reload")
def reload_models():
    """Load freshly retrained models without restarting the server."""
    url_b, text_b = predictor.load_models(reload=True)
    return {"url_model": url_b and url_b["model_name"], "text_model": text_b and text_b["model_name"]}
