"""Trusted-domain allowlist, checked before the ML model.

Production phishing filters (browser safe-browsing, mail gateways) combine reputation
lists with ML: well-known domains are trusted, and the model judges unknown ones. The
domain model cannot learn famous short domains from PhiUSIIL (its legitimate set is
long-tail sites), so without this list it would flag sites such as google.com.

Rules (rules.py) still run on allowlisted domains, because free subdomains on trusted
platforms (sites.google.com, *.firebaseapp.com, *.github.io) are often abused.

Add your own domains, one per line, to data/allowlist.txt.
"""
from __future__ import annotations

from functools import lru_cache

from .config import DATA_DIR

TRUSTED = """
google.com google.co.in youtube.com gmail.com googleapis.com gstatic.com android.com
microsoft.com live.com outlook.com office.com office365.com microsoftonline.com bing.com msn.com
apple.com icloud.com amazon.com amazon.in amazonaws.com primevideo.com
facebook.com instagram.com whatsapp.com whatsapp.net messenger.com meta.com threads.net
twitter.com x.com linkedin.com reddit.com pinterest.com quora.com tumblr.com snapchat.com
telegram.org discord.com zoom.us slack.com dropbox.com box.com notion.so canva.com figma.com
wikipedia.org wikimedia.org wiktionary.org archive.org
github.com gitlab.com bitbucket.org stackoverflow.com stackexchange.com python.org pypi.org
npmjs.com docker.com kaggle.com huggingface.co anthropic.com claude.ai openai.com
netflix.com hotstar.com jiocinema.com spotify.com gaana.com primevideo.com
paypal.com stripe.com razorpay.com visa.com mastercard.com
flipkart.com myntra.com snapdeal.com meesho.com ajio.com nykaa.com bigbasket.com blinkit.com
swiggy.com zomato.com zepto.com uber.com olacabs.com rapido.bike makemytrip.com goibibo.com
irctc.co.in indianrail.gov.in
sbi.co.in onlinesbi.sbi onlinesbi.com sbicard.com hdfcbank.com icicibank.com axisbank.com
kotak.com yesbank.in pnbindia.in bankofbaroda.in canarabank.com unionbankofindia.co.in
idfcfirstbank.com indusind.com rbi.org.in npci.org.in
paytm.com phonepe.com bhimupi.org.in mobikwik.com
gov.in nic.in india.gov.in uidai.gov.in incometax.gov.in gst.gov.in epfindia.gov.in
digilocker.gov.in passportindia.gov.in mygov.in nptel.ac.in swayam.gov.in
indiapost.gov.in airtel.in jio.com vi.in bsnl.co.in
ndtv.com thehindu.com hindustantimes.com indianexpress.com timesofindia.com
indiatimes.com bbc.com bbc.co.uk cnn.com nytimes.com theguardian.com reuters.com
coursera.org udemy.com edx.org khanacademy.org byjus.com unacademy.com geeksforgeeks.org
w3schools.com medium.com substack.com wordpress.com blogger.com
yahoo.com duckduckgo.com proton.me protonmail.com zoho.com zoho.in
adobe.com oracle.com ibm.com intel.com nvidia.com samsung.com dell.com hp.com lenovo.com
""".split()


@lru_cache(maxsize=1)
def trusted_domains() -> frozenset:
    extra = DATA_DIR / "allowlist.txt"
    custom = set()
    if extra.exists():
        custom = {ln.strip().lower().removeprefix("www.") for ln in extra.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")}
    return frozenset(TRUSTED) | custom


def is_trusted(registered_domain: str) -> bool:
    d = registered_domain.lower()
    # gov.in / nic.in style: trust any registered domain under these public-sector suffixes
    return d in trusted_domains() or d.endswith((".gov.in", ".nic.in", ".ac.in", ".edu.in"))
