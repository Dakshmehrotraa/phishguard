"""PhishGuard demo dashboard.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from phishguard import predictor, store  # noqa: E402
from phishguard.config import REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")

COLORS = {"phishing": "#c0392b", "suspicious": "#d68910", "safe": "#1e8449"}
ICONS = {"phishing": "🚨", "suspicious": "⚠️", "safe": "✅"}

EXAMPLES = {
    "SMS: fake KYC alert": "URGENT: Your SBI account will be blocked today. Complete KYC now at http://sbi-kyc-update.tk/login",
    "SMS: prize scam": "Congratulations! You have won Rs 50,000 in our lucky draw. Claim now: bit.ly/claim-prize-now",
    "SMS: normal message": "Hey, are we still meeting at the library at 4? I'll bring the notes.",
    "Email: fake delivery": "Dear customer, your parcel could not be delivered. Pay the redelivery fee of Rs 25 at http://indiapost-redelivery.top/pay within 24 hours.",
}


def verdict_banner(result):
    v, p = result["verdict"], result["probability"]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[v]}22;border-left:6px solid {COLORS[v]}'>"
        f"<span style='font-size:1.4rem;font-weight:700;color:{COLORS[v]}'>{ICONS[v]} {v.upper()}</span>"
        f"<span style='float:right;font-size:1.1rem'>Risk score: <b>{p:.1%}</b></span></div>",
        unsafe_allow_html=True,
    )


def show_reasons(component):
    reasons = component.get("reasons") or []
    if component["type"] == "url":
        st.write(f"**Domain model** ({component['domain']}): {component['model_probability']:.1%} · "
                 f"**Rules**: {component['rule_score']:.1%}")
        for r in component.get("rules_fired", []):
            st.markdown(f"- 🚩 {r['label']} (weight {r['weight']:.2f})")
        st.caption("Domain signals the model weighed:")
    if not reasons:
        st.caption("No single feature stood out.")
        return
    if component["type"] == "url":
        df = pd.DataFrame([{"Signal": r["label"], "Value": str(r["value"]), "Impact": r["impact"]} for r in reasons])
    else:
        df = pd.DataFrame([{"Word": r["token"], "Impact": r["impact"]} for r in reasons])
    st.dataframe(df, hide_index=True)
    st.caption(f"Positive impact pushes toward phishing, negative toward safe. Method: {reasons[0]['method']}.")


def feedback_buttons(kind, text, key):
    c1, c2 = st.columns(2)
    if c1.button("Mark as phishing", key=f"fp_{key}"):
        store.add_feedback(kind, text, True)
        st.success("Saved. It will be used in the next retraining run.")
    if c2.button("Mark as safe", key=f"fs_{key}"):
        store.add_feedback(kind, text, False)
        st.success("Saved. It will be used in the next retraining run.")


url_b, text_b = predictor.load_models()
st.title("🛡️ PhishGuard")
st.caption("AI-based scam and phishing detection for URLs, SMS and email")
if url_b is None or text_b is None:
    st.error("Models are not trained yet. Run `python scripts/download_data.py` and then "
             "`python -m phishguard.train all`, and reload this page.")
    st.stop()

tab_msg, tab_url, tab_perf, tab_hist = st.tabs(["📩 Scan message", "🔗 Scan URL", "📊 Model performance", "🕑 History"])

with tab_msg:
    choice = st.selectbox("Load an example (optional)", ["-"] + list(EXAMPLES))
    text = st.text_area("Paste an SMS or email body", value=EXAMPLES.get(choice, ""), height=140)
    if st.button("Scan message", type="primary") and text.strip():
        result = predictor.scan(text=text)
        store.log_scan("text", text, result["probability"], result["verdict"])
        verdict_banner(result)
        st.write(f"**Triggered by:** {result['triggered_by']}")
        for i, comp in enumerate(result["components"]):
            title = "Message content" if comp["type"] == "text" else f"Link: {comp['input']}"
            with st.expander(f"{ICONS[comp['verdict']]} {title} - {comp['probability']:.1%} ({comp['model']})", expanded=i == 0):
                show_reasons(comp)
        st.divider()
        st.write("Was this verdict wrong?")
        feedback_buttons("text", text, "msg")

with tab_url:
    url = st.text_input("URL to check", placeholder="http://secure-paypal-login.xyz/verify")
    if st.button("Scan URL", type="primary") and url.strip():
        result = predictor.scan_url(url)
        store.log_scan("url", url, result["probability"], result["verdict"])
        verdict_banner(result)
        st.caption(f"Model: {result['model']}")
        show_reasons(result)
        st.divider()
        st.write("Was this verdict wrong?")
        feedback_buttons("url", url, "url")

with tab_perf:
    for task, title in (("url", "URL model"), ("text", "Message (SMS/email) model"), ("benchmark", "Benchmark: UCI Phishing Websites")):
        p = REPORTS_DIR / f"{task}_metrics.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        f = r["final_model_test_metrics"]
        st.subheader(title)
        st.caption(f"Dataset: {r['dataset']} · {r['rows']:,} rows · Balancing: {r['balancing']} · "
                   f"Selected by validation F1: {r['selected_model']}")
        cols = st.columns(5)
        for c, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            c.metric(k.replace("_", "-").upper() if k == "roc_auc" else k.capitalize(), f"{f[k]:.4f}")
        st.dataframe(pd.DataFrame(r["comparison"]).drop(columns=["val_f1"], errors="ignore"),
                     hide_index=True)
        img_cols = st.columns([3, 2])
        for c, name in zip(img_cols, (f"{task}_comparison.png", f"{task}_confusion_matrix.png")):
            if (REPORTS_DIR / name).exists():
                c.image(str(REPORTS_DIR / name))

with tab_hist:
    hist = store.recent_scans(100)
    if hist.empty:
        st.info("No scans yet.")
    else:
        st.dataframe(hist, hide_index=True)
        st.bar_chart(hist["verdict"].value_counts())
