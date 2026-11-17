"""HTML/CSS building blocks for the PhishGuard dashboard.

Pure functions that return HTML strings, so they can be tested without Streamlit.
Every piece of user input is escaped before it is placed in the page.
"""
from __future__ import annotations

from html import escape

VERDICT_TEXT = {"phishing": "Phishing", "suspicious": "Suspicious", "safe": "Safe"}
VERDICT_NOTE = {
    "phishing": "Treat this as a scam. Don't click, reply or pay.",
    "suspicious": "Something looks off. Verify through an official channel first.",
    "safe": "No scam signals found.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--paper:#F3EFE7;--card:#FFFDF8;--ink:#15171C;--ink2:#3A3E48;--muted:#6E727C;--line:#E3DDD0;
--phish:#D9480F;--phish-bg:#FBE9E0;--sus:#B7791F;--sus-bg:#F8EFD9;--safe:#2F7A4F;--safe-bg:#E3F0E7;}
html,body,[class*="css"],.stApp,.stMarkdown,p,li,label,input,textarea,button{font-family:'IBM Plex Sans',sans-serif!important;}
.stApp{background:var(--paper);}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important;}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1040px;padding-top:2.2rem;padding-bottom:4rem;}
/* top bar */
.pg-top{display:flex;justify-content:space-between;align-items:center;border-bottom:1.5px solid var(--ink);padding-bottom:.7rem;margin-bottom:2.2rem;}
.pg-brand{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;letter-spacing:-.01em;color:var(--ink);display:flex;align-items:center;gap:.55rem;}
.pg-mark{width:22px;height:22px;border-radius:5px;background:var(--ink);display:inline-flex;align-items:center;justify-content:center;}
.pg-mark i{width:8px;height:8px;border-radius:50%;background:var(--phish);display:block;}
.pg-status{font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;display:flex;align-items:center;gap:.45rem;}
.pg-status .dot{width:7px;height:7px;border-radius:50%;background:var(--safe);box-shadow:0 0 0 3px var(--safe-bg);}
/* hero */
.pg-hero{display:grid;grid-template-columns:1.35fr 1fr;gap:2.4rem;align-items:end;margin-bottom:2rem;}
.pg-hero h1{font-family:'Space Grotesk',sans-serif!important;font-weight:700;font-size:3rem;line-height:1.02;letter-spacing:-.035em;color:var(--ink);margin:0 0 .8rem 0;padding:0;}
.pg-hero h1 em{font-style:normal;color:var(--phish);}
.pg-hero p{color:var(--ink2);font-size:1.02rem;line-height:1.55;margin:0;max-width:34rem;}
.pg-stats{display:grid;grid-template-columns:1fr 1fr;border:1.5px solid var(--ink);background:var(--card);}
.pg-stat{padding:.85rem 1rem;border-right:1px solid var(--line);border-bottom:1px solid var(--line);}
.pg-stat:nth-child(2n){border-right:none;}.pg-stat:nth-child(n+3){border-bottom:none;}
.pg-stat b{display:block;font-family:'Space Grotesk',sans-serif;font-size:1.55rem;letter-spacing:-.02em;color:var(--ink);line-height:1.1;}
.pg-stat span{font-size:.74rem;color:var(--muted);line-height:1.3;display:block;margin-top:.15rem;}
@media (max-width:760px){.pg-hero{grid-template-columns:1fr;gap:1.2rem}.pg-hero h1{font-size:2.2rem}}
/* tabs */
[data-baseweb="tab-list"]{gap:1.6rem;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{padding:.55rem 0!important;background:transparent!important;}
button[data-baseweb="tab"] p{font-family:'Space Grotesk',sans-serif!important;font-weight:600;font-size:.98rem;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink);}
[data-baseweb="tab-highlight"]{background:var(--ink)!important;height:2px!important;}
[data-baseweb="tab-border"]{display:none;}
/* inputs and buttons */
.stTextArea textarea,.stTextInput input{background:var(--card)!important;border-radius:0!important;font-size:.98rem!important;}
[data-baseweb="textarea"],[data-baseweb="input"],[data-baseweb="select"]>div{border:1.5px solid var(--ink)!important;border-radius:0!important;background:var(--card)!important;}
.stTextInput input{font-family:'IBM Plex Mono',monospace!important;}
.stButton>button{border-radius:0!important;border:1.5px solid var(--ink)!important;font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;letter-spacing:.01em;padding:.5rem 1.3rem!important;box-shadow:3px 3px 0 var(--ink);transition:transform .08s,box-shadow .08s;}
.stButton>button:hover{transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--ink);}
.stButton>button:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink);}
[data-testid="stBaseButton-primary"]{background:var(--ink)!important;color:var(--paper)!important;}
[data-testid="stBaseButton-secondary"]{background:var(--card)!important;color:var(--ink)!important;}
label p{font-weight:500!important;color:var(--ink2)!important;}
/* verdict card */
.pg-verdict{border:1.5px solid var(--ink);background:var(--card);margin:1.4rem 0 1rem 0;display:grid;grid-template-columns:1fr auto;}
.pg-verdict .v-main{padding:1.2rem 1.4rem;border-left:10px solid var(--c);}
.pg-kicker{font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
.pg-verdict .v-word{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2.3rem;letter-spacing:-.03em;color:var(--c);line-height:1.05;margin:.15rem 0 .35rem;}
.pg-verdict .v-note{color:var(--ink2);font-size:.95rem;}
.pg-verdict .v-why{margin-top:.6rem;font-size:.85rem;color:var(--muted);}
.pg-verdict .v-why code{font-family:'IBM Plex Mono',monospace;background:var(--bg);color:var(--ink);padding:.1rem .35rem;font-size:.8rem;word-break:break-all;}
.pg-verdict .v-score{padding:1.2rem 1.5rem;border-left:1px solid var(--line);background:var(--bg);display:flex;flex-direction:column;justify-content:center;align-items:flex-end;min-width:170px;}
.pg-verdict .v-num{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:2.6rem;color:var(--ink);line-height:1;}
.pg-verdict .v-num small{font-size:1.1rem;color:var(--muted);margin-left:.1rem;}
.pg-meter{grid-column:1/-1;height:8px;background:linear-gradient(90deg,var(--safe-bg) 0 40%,var(--sus-bg) 40% 70%,var(--phish-bg) 70% 100%);position:relative;border-top:1px solid var(--line);}
.pg-meter .fill{position:absolute;left:0;top:0;bottom:0;background:var(--c);}
.pg-meter .tick{position:absolute;top:-3px;bottom:-3px;width:1.5px;background:var(--ink);opacity:.35;}
/* detail cards */
.pg-card{border:1px solid var(--line);background:var(--card);padding:1rem 1.2rem;margin:.75rem 0;}
.pg-card-head{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin-bottom:.6rem;}
.pg-card-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1.02rem;color:var(--ink);}
.pg-card-title code{font-family:'IBM Plex Mono',monospace;font-weight:400;font-size:.82rem;color:var(--ink2);word-break:break-all;}
.pg-pill{font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:.2rem .5rem;color:var(--c);background:var(--bg);white-space:nowrap;}
.pg-sub{font-size:.8rem;color:var(--muted);margin:-.25rem 0 .7rem;}
.pg-chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.2rem 0 .8rem;}
.pg-chip{border:1px solid var(--phish);color:var(--phish);background:var(--phish-bg);font-size:.8rem;padding:.25rem .55rem;}
.pg-chip b{font-family:'IBM Plex Mono',monospace;font-weight:600;margin-left:.35rem;}
.pg-split{display:flex;gap:1.6rem;font-size:.82rem;color:var(--ink2);margin-bottom:.7rem;}
.pg-split b{font-family:'IBM Plex Mono',monospace;color:var(--ink);}
/* signal bars */
.pg-sig{display:grid;grid-template-columns:minmax(120px,1.3fr) 2fr 64px;gap:.8rem;align-items:center;padding:.38rem 0;border-top:1px dashed var(--line);font-size:.86rem;}
.pg-sig .name{color:var(--ink);}
.pg-sig .name small{font-family:'IBM Plex Mono',monospace;color:var(--muted);margin-left:.3rem;}
.pg-sig .track{position:relative;height:10px;background:linear-gradient(90deg,transparent calc(50% - .5px),var(--line) calc(50% - .5px) calc(50% + .5px),transparent calc(50% + .5px));}
.pg-sig .bar{position:absolute;top:1px;bottom:1px;}
.pg-sig .bar.up{left:50%;background:var(--phish);}
.pg-sig .bar.down{right:50%;background:var(--safe);}
.pg-sig .val{font-family:'IBM Plex Mono',monospace;font-size:.8rem;text-align:right;color:var(--ink2);}
.pg-legend{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:var(--muted);letter-spacing:.05em;text-transform:uppercase;margin-top:.5rem;}
/* performance */
.pg-sec{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.35rem;letter-spacing:-.02em;color:var(--ink);margin:1.6rem 0 .2rem;}
.pg-tiles{display:grid;grid-template-columns:repeat(5,1fr);border:1.5px solid var(--ink);background:var(--card);margin:.8rem 0;}
.pg-tile{padding:.75rem .9rem;border-right:1px solid var(--line);}.pg-tile:last-child{border-right:none;}
.pg-tile b{display:block;font-family:'IBM Plex Mono',monospace;font-size:1.25rem;color:var(--ink);}
.pg-tile span{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.pg-perf{display:grid;grid-template-columns:1.6fr 1fr;gap:1.2rem;}
@media (max-width:760px){.pg-perf{grid-template-columns:1fr}.pg-tiles{grid-template-columns:repeat(3,1fr)}}
.pg-hbar{display:grid;grid-template-columns:minmax(150px,1.4fr) 2fr 52px;gap:.7rem;align-items:center;font-size:.84rem;padding:.3rem 0;}
.pg-hbar .t{height:12px;background:#EDE8DD;position:relative;}
.pg-hbar .t i{position:absolute;left:0;top:0;bottom:0;background:var(--ink);}
.pg-hbar.best .t i{background:var(--phish);}
.pg-hbar.best .n{font-weight:600;}
.pg-hbar .v{font-family:'IBM Plex Mono',monospace;font-size:.8rem;text-align:right;}
.pg-cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:3px;font-size:.78rem;}
.pg-cm div{padding:.55rem .5rem;text-align:center;}
.pg-cm .h{color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;}
.pg-cm .ok{background:var(--safe-bg);}.pg-cm .bad{background:var(--phish-bg);}
.pg-cm .ok b,.pg-cm .bad b{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;display:block;color:var(--ink);}
/* history */
.pg-row{display:grid;grid-template-columns:110px 1fr 70px 130px;gap:.9rem;align-items:center;padding:.55rem 0;border-top:1px solid var(--line);font-size:.86rem;}
.pg-row .in{color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pg-row .sc{font-family:'IBM Plex Mono',monospace;text-align:right;}
.pg-row .ts{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.75rem;text-align:right;}
.pg-foot{margin-top:3rem;padding-top:.9rem;border-top:1px solid var(--line);font-size:.78rem;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem;}
.pg-foot a{color:var(--ink2);}
.pg-linkscan{font-family:'IBM Plex Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--phish);margin-top:.5rem;}
</style>
"""

_VARS = {
    "phishing": "--c:var(--phish);--bg:var(--phish-bg);",
    "suspicious": "--c:var(--sus);--bg:var(--sus-bg);",
    "safe": "--c:var(--safe);--bg:var(--safe-bg);",
}


def _pct(p: float) -> str:
    v = round(p * 100, 1)
    return f"{v:.0f}" if v in (0, 100) else f"{v:.1f}"


def top_bar(models_ok: bool) -> str:
    status = "Models loaded" if models_ok else "Models missing"
    return (f'<div class="pg-top"><div class="pg-brand"><span class="pg-mark"><i></i></span>PhishGuard</div>'
            f'<div class="pg-status"><span class="dot"></span>{status}</div></div>')


def hero(stats: list[tuple[str, str]]) -> str:
    tiles = "".join(f'<div class="pg-stat"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for v, k in stats[:4])
    return ('<div class="pg-hero"><div><h1>Is this a <em>scam</em>?</h1>'
            '<p>Paste a text message, an email or a link. PhishGuard reads the words, checks every link '
            'inside, and shows exactly why it reached its verdict.</p></div>'
            f'<div class="pg-stats">{tiles}</div></div>')


def verdict_card(verdict: str, probability: float, why: str | None = None) -> str:
    p = max(0.0, min(1.0, float(probability)))
    why_html = f'<div class="v-why">Triggered by <code>{escape(why)}</code></div>' if why else ""
    return (f'<div class="pg-verdict" style="{_VARS[verdict]}">'
            f'<div class="v-main"><div class="pg-kicker">Verdict</div>'
            f'<div class="v-word">{VERDICT_TEXT[verdict]}</div>'
            f'<div class="v-note">{VERDICT_NOTE[verdict]}</div>{why_html}</div>'
            f'<div class="v-score"><div class="pg-kicker">Risk score</div>'
            f'<div class="v-num">{_pct(p)}<small>%</small></div></div>'
            f'<div class="pg-meter"><div class="fill" style="width:{p * 100:.1f}%"></div>'
            f'<div class="tick" style="left:40%"></div><div class="tick" style="left:70%"></div></div></div>')


def _signal_rows(rows: list[tuple[str, str, float]]) -> str:
    """rows: (name, extra, impact). Bars are scaled to the largest absolute impact."""
    if not rows:
        return '<div class="pg-sub">No single signal stood out.</div>'
    scale = max(abs(r[2]) for r in rows) or 1.0
    out = []
    for name, extra, impact in rows:
        w = min(50.0, abs(impact) / scale * 50.0)
        cls = "up" if impact > 0 else "down"
        extra_html = f"<small>{escape(extra)}</small>" if extra not in ("", None) else ""
        out.append(f'<div class="pg-sig"><div class="name">{escape(name)}{extra_html}</div>'
                   f'<div class="track"><div class="bar {cls}" style="width:{w:.1f}%"></div></div>'
                   f'<div class="val">{impact:+.2f}</div></div>')
    out.append('<div class="pg-legend"><span>&larr; pushes toward safe</span><span>pushes toward phishing &rarr;</span></div>')
    return "".join(out)


def component_card(comp: dict) -> str:
    v = comp["verdict"]
    if comp["type"] == "text":
        title = "Message wording"
        rows = [(f"“{r['token']}”", "", r["impact"]) for r in comp.get("reasons") or []]
        sub = f"Model: {escape(comp['model'])} &middot; word-level reasons"
        extra = ""
    else:
        title = f"Link <code>{escape(comp['input'])}</code>"
        rows = [(r["label"], "" if r["value"] == "" else str(r["value"]), r["impact"])
                for r in comp.get("reasons") or []]
        sub = f"Domain checked: <b>{escape(comp.get('domain', ''))}</b> &middot; {escape(comp['model'])}"
        fired = comp.get("rules_fired") or []
        chips = "".join(f'<span class="pg-chip">{escape(r["label"])}<b>{r["weight"]:.2f}</b></span>' for r in fired)
        extra = (f'<div class="pg-split"><span>Domain model <b>{_pct(comp["model_probability"])}%</b></span>'
                 f'<span>Rules <b>{_pct(comp["rule_score"])}%</b></span></div>'
                 + (f'<div class="pg-kicker" style="margin-bottom:.35rem">Rules that fired</div><div class="pg-chips">{chips}</div>' if chips else ""))
        if comp.get("trusted_domain"):
            extra += '<div class="pg-sub">Trusted domain: the model is skipped, but the rules still check the full link.</div>'
    heading = "Signals"
    if v == "safe":  # explain the decision: show what made it look safe
        rows = [r for r in rows if r[2] < 0]
        heading = "Why it looks safe"
    return (f'<div class="pg-card" style="{_VARS[v]}"><div class="pg-card-head">'
            f'<div class="pg-card-title">{title}</div><span class="pg-pill">{VERDICT_TEXT[v]} &middot; {_pct(comp["probability"])}%</span></div>'
            f'<div class="pg-sub">{sub}</div>{extra}'
            + f'<div class="pg-kicker" style="margin:.2rem 0 .2rem">{heading}</div>'
            + _signal_rows(rows) + "</div>")


def perf_section(title: str, report: dict) -> str:
    f = report["final_model_test_metrics"]
    tiles = "".join(f'<div class="pg-tile"><b>{f[k]:.3f}</b><span>{lab}</span></div>' for k, lab in
                    [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                     ("f1", "F1"), ("roc_auc", "ROC-AUC")])
    comp = sorted(report["comparison"], key=lambda r: -r["f1"])
    bars = "".join(
        f'<div class="pg-hbar{" best" if r["model"] == report["selected_model"] else ""}">'
        f'<div class="n">{escape(r["model"])}</div><div class="t"><i style="width:{r["f1"] * 100:.1f}%"></i></div>'
        f'<div class="v">{r["f1"]:.3f}</div></div>' for r in comp)
    cm = f["confusion_matrix"]
    matrix = ('<div class="pg-cm"><div></div><div class="h">Pred. safe</div><div class="h">Pred. phishing</div>'
              f'<div class="h">Actual safe</div><div class="ok"><b>{cm["tn"]:,}</b>correct</div><div class="bad"><b>{cm["fp"]:,}</b>false alarms</div>'
              f'<div class="h">Actual phishing</div><div class="bad"><b>{cm["fn"]:,}</b>missed</div><div class="ok"><b>{cm["tp"]:,}</b>caught</div></div>')
    return (f'<div class="pg-sec">{escape(title)}</div>'
            f'<div class="pg-sub" style="margin:0">{escape(report["dataset"])} &middot; {report["rows"]:,} rows &middot; '
            f'selected by validation F1: <b>{escape(report["selected_model"])}</b></div>'
            f'<div class="pg-tiles">{tiles}</div>'
            f'<div class="pg-perf"><div class="pg-card"><div class="pg-kicker" style="margin-bottom:.4rem">F1 score of every candidate model (0 to 1)</div>{bars}</div>'
            f'<div class="pg-card"><div class="pg-kicker" style="margin-bottom:.5rem">Confusion matrix, selected model</div>{matrix}</div></div>')


def history_rows(rows: list[dict]) -> str:
    if not rows:
        return '<div class="pg-sub">No scans yet. Results you check appear here.</div>'
    out = []
    for r in rows:
        v = r["verdict"] if r["verdict"] in _VARS else "safe"
        out.append(f'<div class="pg-row" style="{_VARS[v]}"><span class="pg-pill">{VERDICT_TEXT[v]}</span>'
                   f'<div class="in">{escape(str(r["input"]))}</div><div class="sc">{_pct(r["probability"])}%</div>'
                   f'<div class="ts">{escape(str(r["ts"]).replace("T", " ")[:16])}</div></div>')
    return "".join(out)


def footer() -> str:
    return ('<div class="pg-foot"><span>PhishGuard &middot; B.Tech final year project &middot; '
            'checks link text only; it never opens the links you paste.</span>'
            '<span>Scores: Safe &lt; 40% &middot; Suspicious 40&ndash;70% &middot; Phishing &ge; 70%</span></div>')
