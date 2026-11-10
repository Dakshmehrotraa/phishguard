# PhishGuard: AI-Based Scam and Phishing Detection System

PhishGuard detects phishing URLs and scam SMS/email messages with machine learning. It returns a risk score, a verdict (safe / suspicious / phishing) and the reasons behind each verdict. A FastAPI backend and a Streamlit dashboard sit on top of the models, and user feedback flows into a retraining loop.

## Architecture

```
 Inputs            URL                      SMS / email text
                    │                              │
 Features   registered domain → char 2-5     cleaning → TF-IDF (word 1-2 grams
            grams + 17 domain features      + character 3-5 grams)
            (TLD, keywords, entropy...)           │   links found in the text ──┐
            + 11 heuristic rules on the           │                             │
            full URL (@, IP, redirects,           │                             │
            shorteners, brand/keywords in         │                             │
            subdomain, free hosting)              │                             │
                    │                              │                             │
 Balancing        SMOTE (training split only)     SMOTE (training split only)   │
                    │                              │                             │
 Models     LR · Random Forest · XGBoost    NB · LR · Random Forest · XGBoost   │
            · RF+XGB soft-vote ensemble     · soft-vote ensemble                 │
            (best validation F1 is kept)    (best validation F1 is kept)         │
            trusted-domain allowlist first;        │                             │
            URL risk = max(model, rules)           │                             │
                    │                              │                             │
 Decision           └──────── fusion: highest risk of message + every link ◄────┘
                              + explanation (SHAP / occlusion)
                                         │
 Serving              FastAPI (/scan, /feedback)  ·  Streamlit dashboard
                                         │
 Retraining           feedback (SQLite) → scripts/retrain.py → reload
```

| Deck element | Where it is in the code |
|---|---|
| URL feature extraction | `phishguard/url_features.py` |
| Heuristic rules branch | `phishguard/rules.py` |
| Trusted-domain allowlist | `phishguard/allowlist.py` (+ your own `data/allowlist.txt`) |
| Text preprocessing / TF-IDF | `phishguard/text_features.py` |
| SMOTE imbalance handling | `phishguard/balance.py` |
| Random Forest / XGBoost / ensemble | `phishguard/models.py` |
| Training + evaluation (accuracy, precision, recall, F1, AUC-ROC) | `phishguard/train.py` |
| Explainability (SHAP) | `phishguard/explain.py` |
| Decision fusion, real-time scan | `phishguard/predictor.py` |
| REST API | `api/main.py` |
| Dashboard | `app/streamlit_app.py` |
| Concept-drift retraining loop | `phishguard/store.py`, `scripts/retrain.py` |

## Datasets

| Used for | Dataset | Size |
|---|---|---|
| Live URL model | PhiUSIIL Phishing URL Dataset (UCI #967) | 235,795 URLs (100,945 phishing / 134,850 legitimate) |
| Message model | UCI SMS Spam Collection (UCI #228) | 5,574 messages (747 spam / 4,827 ham) |
| Benchmark vs. literature | UCI Phishing Websites (UCI #327) | 11,055 rows, 30 features |

The live URL model is trained on raw URLs (PhiUSIIL) because a pasted URL has to be turned into features on the spot. The UCI Phishing Websites dataset only ships 30 pre-computed features, and some of them need page content or WHOIS data, so it is used for the benchmark comparison only.

`scripts/download_data.py` downloads all three. You can also add your own data: any CSV with `url,label` columns works for the URL model, and any CSV with `text,label` columns (for example a phishing email corpus) works for the message model. Labels such as `phishing/legitimate`, `spam/ham` or `1/0` are all accepted.

## Setup

Python 3.10 or newer (3.11 recommended).

```bash
cd phishguard
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# 1. Get the data (a few tens of MB)
python scripts/download_data.py

# 2. Train and compare models (writes models/ and reports/)
python -m phishguard.train all
#    faster first run: python -m phishguard.train url --max-rows 50000

# 3a. Dashboard
streamlit run app/streamlit_app.py          # http://localhost:8501

# 3b. API
uvicorn api.main:app --reload               # docs at http://127.0.0.1:8000/docs
```

Training on the full URL dataset takes a few minutes, depending on your machine.

Example API call:

```bash
curl -X POST http://127.0.0.1:8000/scan -H "Content-Type: application/json" \
  -d '{"text": "URGENT: your SBI account is blocked. Update KYC at http://sbi-kyc-update.tk/login"}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Loaded models and their test metrics |
| `POST /scan` | `{url?, text?}`: scans both and fuses the results |
| `POST /scan/url`, `POST /scan/text` | Scan a single input |
| `POST /feedback` | `{kind: "url" or "text", input, is_phishing}`: report a wrong verdict |
| `GET /history` | Recent scans |
| `POST /reload` | Load retrained models without a restart |

## Results for your slides

After training, `reports/` contains:

- `summary.md`: comparison tables for every model and task, ready to paste into the Result Analysis slide
- `*_comparison.png`: bar chart of accuracy, precision, recall, F1 and ROC-AUC per model
- `*_confusion_matrix.png`: confusion matrix of the selected model
- `*_metrics.json`, `*_comparison.csv`: raw numbers

Use the numbers from your own run. Don't copy figures from the papers in the literature review into the results slide.

## Results from the first training run

These numbers come from a run on the real datasets in Claude's cloud workspace. XGBoost and imbalanced-learn weren't available there, so HistGradientBoosting and the built-in SMOTE were used. Your own run will regenerate `reports/` with XGBoost.

| Task | Data | Best model (by validation F1) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| SMS scam detection | UCI SMS Spam, 5,574 msgs | Ensemble (LR + RF) | 0.992 | 0.993 | 0.946 | 0.969 | 0.996 |
| Benchmark (30 features) | UCI Phishing Websites, 11,055 | Gradient boosting | 0.977 | 0.978 | 0.969 | 0.974 | 0.997 |
| URL, registered domain only | PhiUSIIL, 171,307 unique domains | Ensemble (LR + NB) | 0.805 | 0.606 | 0.637 | 0.621 | 0.835 |

How to present the URL row: it is low on purpose. It is the honest score once dataset bias is removed (see below). The same data scored on full URLs gives a much higher number, but that model flags Wikipedia and Gmail. In the running system, URLs go through three layers: a trusted-domain allowlist, the ML model for unknown domains, and the heuristic rules. A remaining weakness worth stating: small, unknown legitimate sites (a local cafe's website, a personal Netlify page) can still be flagged by the domain model, because the domain name alone carries limited signal. Page content, domain age (WHOIS) or reputation feeds would help, and they are good future-work points.

## Retraining loop (concept drift)

Every wrong verdict a user flags in the dashboard or through `/feedback` is stored in `phishguard.db`. Running

```bash
python scripts/retrain.py
```

adds that feedback to the training split (never to the test split), retrains both models, rewrites the reports and tells a running API to reload. This is the project's practical answer to the concept-drift gap from the literature review.

## Tests

```bash
pytest -q
```

The tests train tiny models on the hand-made samples in `tests/fixtures/` inside a temp folder. They check that the code works and say nothing about accuracy.

## Docker

```bash
python -m phishguard.train all      # train first, so models/ exists
docker compose up --build           # API on :8000, dashboard on :8501
```

## Evaluation protocol (likely viva questions)

1. Stratified 80/20 train/test split. The test set is used once, at the end.
2. The training part is split again 80/20 into fit and validation sets.
3. SMOTE runs on the fit/training data only, never on validation or test data. Oversampling before the split would leak synthetic copies of test samples into training.
4. Every candidate model trains on the same data. The best validation F1 picks the model, which is then retrained on the full training split and scored once on the test set.
5. Reported: accuracy, precision, recall, F1, ROC-AUC, false-positive rate and the confusion matrix. On imbalanced SMS data, F1 and recall on the scam class matter more than accuracy, because predicting "ham" for everything already scores about 87% accuracy.

**Explanations.** Every verdict is explained by occlusion: remove one word of a message, one part of a domain name, or reset one domain feature to its typical legitimate value, then measure how much the phishing log-odds drop. It works with any model and is easy to explain. URL verdicts also list each heuristic rule that fired. (`explain_tabular` in `explain.py` also supports SHAP TreeExplainer for tree models on numeric features, for the benchmark experiments.)

**Fallbacks.** If `xgboost` is missing, scikit-learn's HistGradientBoosting takes its place and is named as such in every report. If `imbalanced-learn` is missing, a built-in SMOTE with the same algorithm is used. Install everything in `requirements.txt` so the report says XGBoost and imblearn SMOTE.

## Known limitations

- **Dataset bias, and how the design handles it.** In PhiUSIIL, all 134,850 legitimate URLs are bare `https://www.<domain>` homepages. 27% of phishing URLs have a path, only 49% use HTTPS and only 42% start with `www.`. Legitimate URLs also never have a subdomain. A model trained on full URLs (an earlier version of this project) reached F1 0.78, but it had learned "has a path, a subdomain, no HTTPS or no www = phishing", and it flagged en.wikipedia.org and mail.google.com as phishing. So the ML model now sees only the registered domain (`google.com`, `sbi.co.in`), each domain appears once, and no domain sits in both train and test. This gives a lower but honest score. Subdomain, path and scheme tricks (`@`, IP hosts, redirects, shorteners, a brand or keywords in a subdomain, free hosting platforms) are scored by transparent rules in `phishguard/rules.py`, which aren't learned from the biased data. The rules can't be fairly scored on this dataset, because its legitimate URLs never have paths or subdomains for them to fire on. This is a good viva talking point: it is exactly the "weak real-world validation" gap from the literature review.
- **Allowlist.** Well-known domains (and Indian `.gov.in`, `.nic.in`, `.ac.in`, `.edu.in` domains, which have restricted registration) skip the ML model, as in production phishing filters. Rules still run on them, because free subdomains on trusted platforms (sites.google.com) are often abused. Add domains to `data/allowlist.txt`.
- **Lexical features only.** The URL model does not fetch pages or query WHOIS. That keeps it fast and offline, but it cannot see page content or domain age.
- **Old SMS data.** The SMS Spam Collection dates from 2012, and Indian-style KYC, UPI and parcel scams are under-represented. Adding a recent smishing dataset with `--data` improves this.
- **No deep learning branch.** CNN/LSTM and DistilBERT are not included. If your architecture slide shows them, either mark them as future work or remove them.
