"""Train tiny models on the sample fixtures into a temp folder, so tests never
touch the real models, reports or database. The fixtures are small hand-made
examples for testing the code paths only - they say nothing about accuracy."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    from phishguard import predictor, store, train

    tmp = tmp_path_factory.mktemp("pg")
    mp = pytest.MonkeyPatch()
    for mod in (train, predictor):
        mp.setattr(mod, "URL_MODEL_PATH", tmp / "url.joblib")
        mp.setattr(mod, "TEXT_MODEL_PATH", tmp / "text.joblib")
    mp.setattr(train, "REPORTS_DIR", tmp)
    mp.setattr(store, "DB_PATH", tmp / "test.db")
    predictor.load_models(reload=True)

    url_bundle = train.train_url(str(FIX / "sample_urls.csv"))
    text_bundle = train.train_text([str(FIX / "sample_sms.tsv")])
    predictor.load_models(reload=True)
    yield {"tmp": tmp, "url": url_bundle, "text": text_bundle}
    mp.undo()
    predictor.load_models(reload=True)
