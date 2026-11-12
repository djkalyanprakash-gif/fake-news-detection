"""
Fake News Detection — training script.
Student project (BTech CSE, Data Science) by Kalyan.

Trains one TF-IDF + Logistic Regression pipeline on BOTH supplied
Kaggle datasets, combined:

  1. archive_3: Fake.csv (23,481 rows) + True.csv (21,417 rows)
     Columns: title, text, subject, date
  2. archive__1__: 2,260 .txt articles
     - overall/overall/celebrityDataset/{fake,real}/*.txt  (540 + 540)
     - training/training/fakeNewsDataset/{fake,legit}/*.txt (490 + 490)
     - Testing_dataset/testingSet/{fake,real}/*.txt        (50 + 50)
     Catalog CSVs in that zip are skipped — they duplicate the .txt files.

Usage:
    python train_model.py                 # uses data/ if present
    python train_model.py --data data     # explicit data folder

Put the two zip files (or their extracted contents) in data/:
    data/archive_3.zip  /  data/Fake.csv, data/True.csv
    data/archive__1__.zip / extracted .txt trees

Output: model/fake_news_model.joblib and model/metrics.json
If no dataset is found, a small built-in sample set is used instead,
so training never crashes — the app then clearly says it is in
sample/demo mode.
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Small built-in fallback only — used when neither dataset is present.
SAMPLE_DATA = [
    ("Government announces new metro line for Hyderabad", "The state government on Tuesday announced a new metro rail line connecting the airport to the city centre. Officials said construction will begin next year and the project is expected to take three years to complete.", 0),
    ("Scientists confirm water found on the Moon", "Researchers at the national space agency said new readings from the lunar orbiter confirm the presence of water ice in permanently shadowed craters near the Moon's south pole.", 0),
    ("Local college wins national coding competition", "Students from the computer science department won first place at the national coding championship held in Bengaluru, beating 120 teams from across the country.", 0),
    ("RBI keeps repo rate unchanged at 6.5 percent", "The Reserve Bank of India on Friday kept the repo rate unchanged, citing inflation concerns and global economic uncertainty. Economists had widely expected the decision.", 0),
    ("SHOCKING! Scientists prove aliens built the pyramids overnight", "SECRET documents leaked by an insider PROVE that aliens built the pyramids in one night and the government has been hiding it for 5000 years. Share before they delete this!", 1),
    ("You won't BELIEVE what this actor did — doctors hate him", "This one weird trick discovered by a local man will make you rich overnight. Banks are furious and trying to ban it. Click now before it is gone forever!", 1),
    ("BREAKING: Celebrity dies in crash — but nobody can confirm it", "Unverified reports claim a famous actor died in a car crash last night. No hospital, police or family member has confirmed anything, but social media is exploding.", 1),
    ("Miracle cure found: drink this and never get sick again", "Doctors don't want you to know this secret drink cures every disease in 24 hours. Big pharma is hiding it. One glass a day and you will never need a hospital again!", 1),
]


def clean(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_csv_dataset(data_dir: Path) -> pd.DataFrame:
    """Load Fake.csv / True.csv, from extracted files or inside a zip."""
    frames = []
    for name, label in (("Fake.csv", 1), ("True.csv", 0)):
        path = data_dir / name
        df = None
        if path.exists():
            df = pd.read_csv(path, usecols=["title", "text"])
        else:
            for zf in data_dir.glob("*.zip"):
                with zipfile.ZipFile(zf) as z:
                    if name in z.namelist():
                        df = pd.read_csv(z.open(name), usecols=["title", "text"])
                        break
        if df is not None:
            df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean)
            df["label"] = label
            frames.append(df[["text", "label"]])
            print(f"  {name}: {len(df):,} rows (label={label})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text", "label"])


def load_txt_dataset(data_dir: Path) -> pd.DataFrame:
    """Load .txt articles, from extracted trees or inside a zip."""
    rows = []

    def add_file(text: str, label: int):
        text = clean(text)
        if len(text) > 30:
            rows.append({"text": text, "label": label})

    def label_for(parts) -> int | None:
        low = [p.lower() for p in parts]
        if "fake" in low:
            return 1
        if "real" in low or "legit" in low:
            return 0
        return None

    for txt in data_dir.rglob("*.txt"):
        label = label_for(txt.parts)
        if label is not None:
            add_file(txt.read_text(errors="ignore"), label)
    for zf in data_dir.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            for name in z.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                label = label_for(Path(name).parts)
                if label is not None:
                    add_file(z.read(name).decode(errors="ignore"), label)
    print(f"  .txt articles: {len(rows):,} rows")
    return pd.DataFrame(rows, columns=["text", "label"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Folder with the datasets / zips")
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    data_dir = BASE / args.data
    print(f"Loading datasets from {data_dir} ...")
    parts = [load_csv_dataset(data_dir), load_txt_dataset(data_dir)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    source = "kaggle_combined"
    if df.empty:
        print("No dataset found — training on the small built-in sample set.")
        df = pd.DataFrame(SAMPLE_DATA, columns=["title", "text", "label"])
        df["text"] = (df["title"] + " " + df["text"]).map(clean)
        df = df[["text", "label"]]
        source = "sample_fallback"

    df = df.dropna().drop_duplicates(subset=["text"])
    print(f"Total training rows: {len(df):,} "
          f"(fake={int((df.label == 1).sum()):,}, real={int((df.label == 0).sum()):,})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size,
        random_state=42, stratify=df["label"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features,
                                  ngram_range=(1, 2), sublinear_tf=True,
                                  min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   solver="liblinear")),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=["Real", "Fake"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "source": source,
        "rows": int(len(df)),
        "fake": int((df.label == 1).sum()),
        "real": int((df.label == 0).sum()),
        "test_size": args.test_size,
        "accuracy": round(float(acc), 4),
        "model": "TF-IDF (1-2 grams, 20k features) + Logistic Regression",
        "note": "Student project. Accuracy is on a held-out split of these "
                "Kaggle datasets (mostly 2016-2018 US news + celebrity news), "
                "not a general fact-checking score.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
