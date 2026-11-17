"""PhishGuard demo dashboard.

    streamlit run app/streamlit_app.py

Test links: add ?url=<link> or ?msg=<message text> (URL-encoded) to the app's address
and the scan runs as soon as the page opens.
"""
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from phishguard import predictor, store  # noqa: E402
from phishguard.config import REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="PhishGuard · scam & phishing check", page_icon="◆", layout="wide")

EXAMPLES = {
    "Choose an example…": "",
    "SMS · fake KYC alert": "URGENT: Your SBI account will be blocked today. Complete KYC now at http://sbi-kyc-update.tk/login",
    "SMS · prize scam": "Congratulations! You have won Rs 50,000 in our lucky draw. Claim now: bit.ly/claim-prize-now",
    "SMS · normal message": "Hey, are we still meeting at the library at 4? I'll bring the notes.",
    "Email · fake delivery": "Dear customer, your parcel could not be delivered. Pay the redelivery fee of Rs 25 at http://indiapost-redelivery.top/pay within 24 hours.",
}


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def load_report(task):
    p = REPORTS_DIR / f"{task}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def feedback_buttons(kind, text, key):
    st.write("")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Report as phishing", key=f"fp_{key}"):
        store.add_feedback(kind, text, True)
        st.toast("Thanks. It will be used in the next retraining run.")
    if c2.button("Report as safe", key=f"fs_{key}"):
        store.add_feedback(kind, text, False)
        st.toast("Thanks. It will be used in the next retraining run.")


def show_message_result(result, text, key):
    single = len(result["components"]) == 1
    why = None if single else result["triggered_by"].split(": ", 1)[-1]
    html(ui.verdict_card(result["verdict"], result["probability"], why))
    for comp in result["components"]:
        html(ui.component_card(comp))
    feedback_buttons("text", text, key)


def show_url_result(result, url, key):
    html(ui.verdict_card(result["verdict"], result["probability"]))
    html(ui.component_card(result))
    feedback_buttons("url", url, key)


def scan_from_link():
    """Shareable test links: ?url=<link> or ?msg=<message text> scans on page load."""
    link_url, link_msg = st.query_params.get("url"), st.query_params.get("msg")
    if not (link_url or link_msg):
        return
    logged = st.session_state.setdefault("logged_links", set())
    html('<div class="pg-linkscan">Scan from a shared link</div>')
    if link_msg:
        html(f'<div class="pg-card"><div class="pg-kicker">Message</div>'
             f'<div style="margin-top:.3rem">{ui.escape(link_msg)}</div></div>')
        result = predictor.scan(text=link_msg)
        if ("msg", link_msg) not in logged:
            store.log_scan("text", link_msg, result["probability"], result["verdict"])
            logged.add(("msg", link_msg))
        show_message_result(result, link_msg, "link_msg")
    else:
        result = predictor.scan_url(link_url)
        if ("url", link_url) not in logged:
            store.log_scan("url", link_url, result["probability"], result["verdict"])
            logged.add(("url", link_url))
        show_url_result(result, link_url, "link_url")
    if st.button("Clear and check something else"):
        st.query_params.clear()
        st.rerun()
    st.write("")


url_b, text_b = predictor.load_models()
html(ui.CSS)
html(ui.top_bar(url_b is not None and text_b is not None))

text_r, bench_r = load_report("text"), load_report("benchmark")
stats = [
    (f"{text_r['final_model_test_metrics']['accuracy'] * 100:.1f}%" if text_r else "–", "SMS accuracy on held-out data"),
    (f"{bench_r['final_model_test_metrics']['accuracy'] * 100:.1f}%" if bench_r else "–", "Benchmark website accuracy"),
    ("12", "transparent rules for link tricks"),
    ("< 1 s", "per check, with reasons"),
]
html(ui.hero(stats))

if url_b is None or text_b is None:
    st.error("Models are not trained yet. Run `python scripts/download_data.py` and then "
             "`python -m phishguard.train all`, and reload this page.")
    st.stop()

scan_from_link()

tab_msg, tab_url, tab_perf, tab_hist = st.tabs(["Check a message", "Check a link", "How it performs", "Recent checks"])

with tab_msg:
    choice = st.selectbox("Start from an example, or paste your own below", list(EXAMPLES))
    text = st.text_area("SMS or email text", value=EXAMPLES[choice], height=130,
                        placeholder="Paste the message exactly as you received it…")
    if st.button("Check message", type="primary") and text.strip():
        result = predictor.scan(text=text)
        store.log_scan("text", text, result["probability"], result["verdict"])
        show_message_result(result, text, "msg")

with tab_url:
    url = st.text_input("Link", placeholder="http://secure-paypal-login.xyz/verify")
    if st.button("Check link", type="primary") and url.strip():
        result = predictor.scan_url(url)
        store.log_scan("url", url, result["probability"], result["verdict"])
        show_url_result(result, url, "url")

with tab_perf:
    html('<div class="pg-sub" style="margin-top:.8rem">Every number below comes from data the models never saw '
         'during training (a stratified 20% test split).</div>')
    for task, title in (("text", "Messages: SMS and email"), ("url", "Links: domain model"),
                        ("benchmark", "Benchmark: UCI Phishing Websites")):
        r = load_report(task)
        if r:
            html(ui.perf_section(title, r))

with tab_hist:
    hist = store.recent_scans(40)
    html('<div class="pg-sec">Recent checks</div><div class="pg-sub" style="margin:0 0 .6rem">'
         'Stored on this server for the retraining loop.</div>')
    html(ui.history_rows(hist.to_dict(orient="records") if not hist.empty else []))

html(ui.footer())
