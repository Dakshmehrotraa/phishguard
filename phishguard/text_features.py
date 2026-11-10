"""Text preprocessing and TF-IDF vectorisation for SMS / email content."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\+?\d[\d\s-]{7,}\d")
_MONEY = re.compile(r"([£$€₹]\s?\d[\d,.]*|\b(rs\.?|inr|usd|gbp|eur)\s?\d[\d,.]*|\d[\d,.]*\s?(rs|inr|usd|gbp|eur)\b)", re.I)
_NUM = re.compile(r"\d+")


def clean_text(text: str) -> str:
    """Replace volatile tokens with placeholders so the model learns patterns, not specific numbers."""
    t = str(text).lower()
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _MONEY.sub(" moneytoken ", t)
    t = _PHONE.sub(" phonetoken ", t)
    t = _NUM.sub(" numtoken ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_vectorizer() -> FeatureUnion:
    """Word 1-2 grams catch phrases ("claim your prize"); character 3-5 grams catch
    obfuscated spellings ("fr33", "acc0unt") that word tokens miss."""
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                 max_features=20000, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, max_features=30000, sublinear_tf=True)),
    ])
