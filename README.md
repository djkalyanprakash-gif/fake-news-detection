# Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## About

Fake News Detection is a Streamlit web app that estimates whether a news article reads like real or fake news. Paste a headline and article text, and the app gives a verdict with a confidence score and the words that pushed the prediction most. The model is a TF-IDF and logistic regression pipeline trained on two Kaggle datasets combined, about 40,000 articles in total.

## Features

- Paste any headline and article text and get a Real or Fake verdict with confidence
- Shows the words that influenced the result most
- Live news check: searches current coverage for the headline across a curated set of national, regional and international news outlets and shows real recent articles next to the verdict, so recent events the model never saw can still be judged from actual reporting (on by default, untick to turn it off). It also cross-checks the two: when the model says Fake but several outlets are reporting the story, the app says so plainly and points out the coverage is the stronger signal
- Optional Gemini summary: if a Gemini API key is set in Streamlit secrets (`GEMINI_API_KEY`), Gemini briefly says whether the coverage found supports the headline. Without a key the live check still works, showing the sources only
- Warns clearly when the pasted text is too short for a reliable verdict
- Three built-in examples to try without typing anything
- Keeps a short history of recent checks in the session
- Runs the full trained model when it is present, and falls back to a small sample model with a clear on-screen warning if the model file is missing

## Live Demo

Try it here: https://jwxxtkjkizjx5lbuab8nwi.streamlit.app/ (hosted free on Streamlit Community Cloud)

## Dataset

Two Kaggle datasets are combined for training:

1. **Fake and Real News Dataset** — `Fake.csv` (23,481 fake articles) and `True.csv` (21,417 real articles), with columns `title`, `text`, `subject`, `date`. Available at https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. **A second Kaggle text-file collection** — 2,260 articles in plain text files, balanced fake and real, across three groups: celebrity articles (540 fake + 540 real), fakeNewsDataset training articles (490 fake + 490 real), and a testing set (50 fake + 50 real).

The datasets are not included in this repository because of their size. See [data/README.md](data/README.md) for where to get them and how to use them.

## Model & Results

- **Features:** TF-IDF, word 1–2 grams, 20,000 features, `sublinear_tf=True`, `min_df=2`, English stop words removed
- **Classifier:** Logistic Regression, `class_weight="balanced"`, liblinear solver
- **Training data:** 39,907 unique articles after de-duplication (18,441 fake / 21,466 real)
- **Split:** stratified 80/20 train/test
- **Held-out accuracy:** 98.05% on the 20% test split
- **Precision / recall / F1:** about 0.98 for both classes

Confusion matrix on the held-out split (rows = true label, columns = predicted):

```
[[4200,   94],
 [  62, 3626]]
```

That accuracy is only on a held-out split of these datasets, which are mostly 2016–2018 US and celebrity news. It is not a general fact-checking score, and the model should not be used to judge whether a claim is true.

## Project Structure

```
fake-news-detection/
├── app.py                  # Streamlit app
├── train_model.py          # Training script (loads both datasets, saves the model)
├── requirements.txt        # Python dependencies
├── model/
│   ├── fake_news_model.joblib.b64   # Trained model, base64-encoded
│   └── metrics.json        # Training metrics
├── data/
│   └── README.md           # Where to get the datasets (not included in the repo)
├── .streamlit/
│   └── config.toml         # App theme
├── .github/
│   └── workflows/
│       └── ci.yml          # CI: install dependencies, compile the Python files
├── CONTRIBUTING.md
├── LICENSE
├── SECURITY.md
└── README.md
```

## Installation

```bash
git clone https://github.com/djkalyanprakash-gif/fake-news-detection.git
cd fake-news-detection
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the app:

```bash
streamlit run app.py
```

To retrain the model from scratch, put the two dataset zips in `data/` as `archive_3.zip` and `archive__1__.zip` (see [data/README.md](data/README.md)), then run:

```bash
python train_model.py
```

This saves a new `model/fake_news_model.joblib` and `model/metrics.json`. If no dataset is found, training falls back to a small built-in sample set so it never crashes — the app will say it is in demo mode.

The trained model is tracked in this repo as `model/fake_news_model.joblib.b64` (base64 text) rather than as a binary file. The app loads the binary `.joblib` file if it exists, otherwise decodes the `.b64` file in memory.

## How it works

1. **Cleaning:** the headline and article text are combined, lowercased, URLs are removed, and everything except letters is stripped out.
2. **TF-IDF:** the cleaned text is turned into word and two-word features, weighted by how distinctive they are across the training set (20,000 features).
3. **Logistic regression:** the features are fed to a logistic regression classifier, which outputs the probability that the article is fake. Above 0.5 the verdict is Fake, below it Real.
4. **Top words:** because the classifier is linear, the app can show which words in the article pushed the score up or down the most.
5. **Live check:** if the box is ticked, the app also searches current coverage for the headline — restricted to a curated set of about 55 national, regional and international news outlets (the full list of sites is in `TRACKED_SITE_BATCHES` in `app.py`) — and lists the most relevant recent articles, newest first. It then cross-checks the model's verdict against the coverage: if the model says Fake but two or more outlets are reporting the story, the app warns that the coverage is the stronger signal, because the model only judges writing style. With a Gemini API key set in secrets, Gemini adds a short summary of whether the coverage supports the headline.

## Limitations

- The training data is mostly 2016–2018 US and celebrity news, so the model is weakest on recent events, Indian news, and non-English text. The live news check is there to cover exactly that gap, but it only shows coverage — it does not decide what is true.
- It judges writing style, not facts. A well-written false article can score as real, and a badly written true article can score as fake.
- Short headlines alone give unreliable model results; the app warns about this, and for a headline on its own it shows only the live news check.
- The live check needs internet access and Google News to be reachable; if the search fails, the app says so and still shows the model verdict.

## Privacy & Security

- The model runs locally inside the app — pasted articles are processed in memory only and are never stored or saved to disk.
- There is no login, no analytics, and no tracking.
- The live news check is the only external call the app makes, and only when its box is ticked (it is on by default): it sends only the headline to the news search (Google News RSS, restricted to the tracked outlets) and shows the matching coverage. The article text is never sent. Untick the box and nothing leaves the app.
- If a Gemini API key is set in Streamlit secrets, the headline and the titles of the coverage found are also sent to the Gemini API for a short summary. No key is set by default, and the key is never stored in this repository.
- The only cookies are the ones Streamlit itself needs to keep the app running.
- Recent checks are kept only in your browser session (in memory) and disappear when the session ends. You can clear them any time with the Clear history button.
- This repository contains no secrets, tokens, passwords, or personal data. See [SECURITY.md](SECURITY.md).

## Future work

- Add a second dataset of recent news so the model handles current events better
- Try a transformer-based classifier and compare it with the current model on the same split
- Add a simple API endpoint so the model can be called from other tools

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and small improvements are welcome.

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Kalyan** — BTech CSE (Data Science), 3rd year

- GitHub: https://github.com/djkalyanprakash-gif
- LinkedIn: https://www.linkedin.com/in/dj-kalyan-prakash-534297329/
