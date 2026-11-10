"""Candidate classifiers.

XGBoost is used when installed; otherwise scikit-learn's
HistGradientBoostingClassifier stands in as the gradient-boosting model
(and is labelled as such in every report, so results are never mislabelled).
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # not installed, or (on macOS) the libomp runtime is missing: brew install libomp
    HAS_XGB = False


def _boosting(sparse: bool = False):
    if HAS_XGB:
        return "XGBoost", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        )
    if sparse:
        return None, None  # HistGradientBoosting cannot take sparse TF-IDF input
    return "HistGradientBoosting (XGBoost not installed)", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.1, random_state=RANDOM_STATE,
    )


def tabular_candidates() -> dict:
    """For numeric feature tables (URL features, UCI Phishing Websites)."""
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)
    boost_name, boost = _boosting()
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": rf,
        boost_name: boost,
    }
    short = "XGB" if HAS_XGB else "HGB"
    models[f"Ensemble (RF + {short}, soft vote)"] = VotingClassifier(
        [("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
         ("boost", _boosting()[1])],
        voting="soft",
    )
    return models


def text_candidates() -> dict:
    """For sparse TF-IDF text features (SMS / email)."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=3000, C=10),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
    }
    boost_name, boost = _boosting(sparse=True)
    if boost is not None:
        models[boost_name] = boost
        models["Ensemble (LR + RF + XGB, soft vote)"] = VotingClassifier(
            [("lr", LogisticRegression(max_iter=3000, C=10)),
             ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
             ("xgb", _boosting(sparse=True)[1])],
            voting="soft",
        )
    else:
        models["Ensemble (LR + RF, soft vote)"] = VotingClassifier(
            [("lr", LogisticRegression(max_iter=3000, C=10)),
             ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))],
            voting="soft",
        )
    return models


def url_candidates() -> dict:
    """For the sparse domain features (~170k domains x ~50k columns). The Random Forest is kept
    small (fewer trees, min leaf size) so the forest fits comfortably in a laptop's memory."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=3000, C=4),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.5,
                                                n_jobs=-1, random_state=RANDOM_STATE),
    }
    boost_name, boost = _boosting(sparse=True)
    members = [("lr", LogisticRegression(max_iter=3000, C=4)), ("nb", MultinomialNB(alpha=0.1))]
    if boost is not None:
        models[boost_name] = boost
        members.append(("xgb", _boosting(sparse=True)[1]))
    label = " + ".join({"lr": "LR", "nb": "NB", "xgb": "XGB"}[k] for k, _ in members)
    models[f"Ensemble ({label}, soft vote)"] = VotingClassifier(members, voting="soft")
    return models
